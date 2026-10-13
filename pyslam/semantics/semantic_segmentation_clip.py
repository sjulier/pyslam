"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import cv2
from einops import rearrange
import numpy as np
import os
import sys
import platform

import torch
from PIL import Image
import packaging
import packaging.version
import types

from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose

# f3rm's CLIP code does `from pkg_resources import packaging`. setuptools >= 81 no longer ships
# pkg_resources, and recent builds that still do may not expose the vendored `packaging`. In that
# case register a minimal in-process stand-in that provides only `packaging` (nothing is installed).
try:
    import pkg_resources
except ImportError:
    pkg_resources = types.ModuleType("pkg_resources")
    sys.modules["pkg_resources"] = pkg_resources
if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = packaging

import torch.nn.functional as F

# Two implementations of CLIP with dense (per-patch) features, the same OpenAI weights and the same maths:
# - open_clip (the conda package open-clip-torch, part of the semantics level, also on native Windows),
#   with the dense features computed here (OpenClipDense);
# - f3rm's CLIP module, unpacked into thirdparty/f3rm_pkg by scripts/install_extra.sh (models-semantics).
# open_clip is used when it is installed; PYSLAM_CLIP_BACKEND=f3rm or =open_clip chooses one.
import pyslam.config as config

try:
    import open_clip
except ImportError:
    open_clip = None

try:
    config.cfg.set_lib("f3rm")
    from f3rm.features.clip import clip as f3rm_clip
    from f3rm.features.clip import tokenize as f3rm_tokenize
except ImportError:
    f3rm_clip = f3rm_tokenize = None

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_utils import (
    similarity_heatmap_image,
    labels_color_map_factory,
    labels_to_image,
)
from .semantic_labels_factory import semantic_labels_factory

from pyslam.utilities.logging import Printer


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# The open_clip names of the OpenAI CLIP models (the OpenAI weights were trained with QuickGELU)
kOpenClipModels = {
    "ViT-B/32": "ViT-B-32-quickgelu",
    "ViT-B/16": "ViT-B-16-quickgelu",
    "ViT-L/14": "ViT-L-14-quickgelu",
    "ViT-L/14@336px": "ViT-L-14-336-quickgelu",
}


def clip_backend(encoder_name):
    """The CLIP implementation to use: "open_clip" or "f3rm" (see the imports above)."""
    wanted = os.environ.get("PYSLAM_CLIP_BACKEND", "")
    if wanted == "f3rm" or (wanted != "open_clip" and (open_clip is None or encoder_name not in kOpenClipModels)):
        if f3rm_clip is None:
            raise ImportError(
                "CLIP needs open_clip (the conda package open-clip-torch) or f3rm (pixi run -e semantics models-semantics)"
            )
        return "f3rm"
    if open_clip is None:
        raise ImportError("PYSLAM_CLIP_BACKEND=open_clip, but open_clip is not installed (open-clip-torch)")
    if encoder_name not in kOpenClipModels:
        raise ValueError(f"open_clip backend: {encoder_name} is not one of {list(kOpenClipModels)}")
    return "open_clip"


def interpolate_positional_embedding(positional_embedding, x, patch_size, h, w):
    """The positional embedding of a ViT, interpolated to an image of h x w pixels, i.e. (h // patch_size) x
    (w // patch_size) patches (bicubic). The same computation as f3rm's interpolate_positional_embedding,
    itself from DINO's interpolate_pos_encoding:
    https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L174
    """
    num_patches = x.shape[1] - 1
    num_og_patches = positional_embedding.shape[0] - 1
    if num_patches == num_og_patches and h == w:
        return positional_embedding.to(x.dtype)
    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]
    patch_pos_embed = positional_embedding[1:]
    h0, w0 = h // patch_size, w // patch_size
    assert h0 * w0 == num_patches, "Number of patches does not match"
    h0, w0 = h0 + 0.1, w0 + 0.1  # against floating point error in the interpolation (DINO issue 8)
    patch_per_ax = int(np.sqrt(num_og_patches))
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, patch_per_ax, patch_per_ax, dim).permute(0, 3, 1, 2),
        scale_factor=(h0 / patch_per_ax, w0 / patch_per_ax),
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1], "Interpolation error."
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(-1, dim)
    return torch.cat([class_pos_embed, patch_pos_embed], dim=0).to(x.dtype)


class OpenClipDense(torch.nn.Module):
    """An open_clip ViT CLIP model with the interface this module uses from f3rm: encode_text(),
    get_patch_encodings() and visual.patch_size. The patch encodings follow MaskCLIP
    (https://arxiv.org/abs/2112.01071), as f3rm does: the last transformer block is replaced by its value
    projection alone (no attention between the patches, no MLP), then ln_post and the projection to the
    space of the text embeddings."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        patch_size = model.visual.patch_size
        self.visual = types.SimpleNamespace(
            patch_size=patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        )

    def encode_text(self, tokens):
        return self.model.encode_text(tokens)

    def get_patch_encodings(self, image):
        v = self.model.visual
        _, _, h, w = image.shape
        x = v.conv1(image.to(v.conv1.weight.dtype))  # (B, width, h / patch, w / patch)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, patches, width)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + interpolate_positional_embedding(v.positional_embedding, x, self.visual.patch_size, h, w)
        x = v.ln_pre(x)
        batch_first = getattr(v.transformer, "batch_first", True)
        if not batch_first:
            x = x.permute(1, 0, 2)
        *blocks, last = v.transformer.resblocks
        for block in blocks:
            x = block(x)
        dim = last.attn.embed_dim
        x = F.linear(last.ln_1(x), last.attn.in_proj_weight[-dim:], last.attn.in_proj_bias[-dim:])
        x = F.linear(x, last.attn.out_proj.weight, last.attn.out_proj.bias)
        if not batch_first:
            x = x.permute(1, 0, 2)
        x = v.ln_post(x[:, 1:, :])  # the patches, not the class token
        if v.proj is not None:
            x = x @ v.proj
        return x


class SemanticSegmentationCLIP(SemanticSegmentationBase):
    # CLIP available models: https://github.com/f3rm/f3rm/tree/main/f3rm/features/clip
    available_configs = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ]

    supported_feature_types = [
        SemanticFeatureType.LABEL,
        SemanticFeatureType.PROBABILITY_VECTOR,
        SemanticFeatureType.FEATURE_VECTOR,
    ]

    # TODO(dvdmc): take the text query parameter out
    def __init__(
        self,
        device=None,
        encoder_name="ViT-L/14@336px",
        model_path="",
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 1024),
        semantic_feature_type=SemanticFeatureType.LABEL,
        custom_set_labels=None,
        sim_text_query="clock",
        skip_center_crop=True,
        **kwargs,
    ):

        device = self.init_device(device)

        self.encoder_name = encoder_name  # Keep the encoder name to infer if it's ViT or RN
        self.backend = clip_backend(encoder_name)
        if self.backend == "open_clip":
            self.tokenize = open_clip.get_tokenizer(kOpenClipModels[encoder_name])
        else:
            self.tokenize = f3rm_tokenize
        tokenize = self.tokenize

        # NOTE: transform is called preprocess in the original code
        model, transform = self.init_model(
            device, encoder_name, semantic_dataset_type, image_size, model_path
        )

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        # Config the dataset type
        if semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
            if custom_set_labels is None:
                raise ValueError(
                    "custom_set_labels must be provided if semantic_dataset_type is CUSTOM_SET"
                )
            self.semantic_color_map = labels_color_map_factory(
                semantic_dataset_type, num_classes=len(custom_set_labels)
            )
        elif semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            if semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
                raise ValueError(
                    "semantic_feature_type must be FEATURE_VECTOR if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            if sim_text_query == "":
                raise ValueError(
                    "sim_text_query must be provided if semantic_dataset_type is FEATURE_SIMILARITY"
                )
            self.semantic_color_map = None
            self.sim_scale = 3.0  # NOTE: This is for visualization
        else:
            self.semantic_color_map = labels_color_map_factory(semantic_dataset_type)

        self.semantic_dataset_type = semantic_dataset_type

        # Config the text encodings
        if semantic_dataset_type == SemanticDatasetType.CUSTOM_SET:
            self.label_names = custom_set_labels
            self.tokens = [tokenize(text_query).to(device) for text_query in custom_set_labels]
        elif semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            # We already checked that semantic_feature_type is SemanticFeatureType.FEATURE_VECTOR
            self.label_names = [sim_text_query]
            self.tokens = [
                tokenize(sim_text_query).to(device)
            ]  # We will only work with a single text query
        else:
            self.label_names = semantic_labels_factory(semantic_dataset_type)
            self.tokens = torch.stack(
                [tokenize(text_query).to(device) for text_query in self.label_names]
            )  # Shape: (N, token_dim)

        self.text_embs = torch.stack([model.encode_text(token).squeeze(0) for token in self.tokens])

        self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)

        # Patch the preprocess if we want to skip center crop
        if skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, CenterCrop) for t in transform.transforms]
            assert sum(is_center_crop) == 1, "There should be exactly one CenterCrop transform"
            # Create new transform without center crop
            transform = Compose([t for t in transform.transforms if not isinstance(t, CenterCrop)])
            print("Skipping center crop")

        super().__init__(model, transform, device, semantic_feature_type)

    def init_model(self, device, encoder_name, semantic_dataset_type, image_size, model_path):

        # Check if selected config is available
        if encoder_name not in self.available_configs:
            raise ValueError(
                f"Segformer does not support {encoder_name} model with size {image_size} and dataset {semantic_dataset_type}"
            )

        if model_path == "" and self.backend == "open_clip":  # the OpenAI weights, through open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                kOpenClipModels[encoder_name], pretrained="openai", device=device
            )
            model = OpenClipDense(model.eval())
            print(f"SemanticSegmentationCLIP: open_clip {kOpenClipModels[encoder_name]} (OpenAI weights)")
        elif model_path == "":  # Load pre-trained models
            model, preprocess = f3rm_clip.load(encoder_name, device)
            print(f"SemanticSegmentationCLIP: f3rm {encoder_name}")
        else:
            raise NotImplementedError(
                "Segformer only supports pre-trained model for now"
            )  # TODO(dvdmc): allow to load a custom model
        model = model.eval()

        return model, preprocess

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            # Convert string to torch.device object
            device = torch.device(device)
        # At this point, device should be a torch.device object
        if device.type == "cuda":
            print("SemanticSegmentationCLIP: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():  # Should return True for MPS availability
                raise Exception("SemanticSegmentationCLIP: MPS is not available")
            print("SemanticSegmentationCLIP: Using MPS")
        else:
            print("SemanticSegmentationCLIP: Using CPU")
        return device

    def set_query_word(self, query_word):
        if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.label_names = [query_word]
            self.tokens = [self.tokenize(query_word).to(self.device)]
            self.text_embs = torch.stack(
                [self.model.encode_text(token).squeeze(0) for token in self.tokens]
            )
            self.text_embs /= self.text_embs.norm(dim=-1, keepdim=True)
        else:
            Printer.red(
                "Setting the query word will have no effect since semantic_dataset_type is not FEATURE_SIMILARITY"
            )

    def get_output_dims(self, h_in, w_in):
        """Compute output dimensions."""
        # from https://github.com/f3rm/f3rm/blob/main/f3rm/features/clip_extract.py
        if self.encoder_name.startswith("ViT"):
            h_out = h_in // self.model.visual.patch_size
            w_out = w_in // self.model.visual.patch_size
            return h_out, w_out

        if self.encoder_name.startswith("RN"):
            h_out = max(h_in / w_in, 1.0) * self.model.visual.attnpool.spacial_dim
            w_out = max(w_in / h_in, 1.0) * self.model.visual.attnpool.spacial_dim
            return int(h_out), int(w_out)

        raise ValueError(f"unknown clip model: {self.encoder_name}")

    def num_classes(self):
        return len(self.label_names)

    @torch.no_grad()
    def infer(self, image) -> SemanticSegmentationOutput:
        prev_width = image.shape[1]
        prev_height = image.shape[0]
        recover_size = transforms.Resize(
            (prev_height, prev_width), interpolation=transforms.InterpolationMode.NEAREST
        )
        image_pil = Image.fromarray(image)
        img_t = self.transform(image_pil).unsqueeze(0).to(self.device)

        # We use a batch inference approach for using the implementation in f3rm
        embeddings = []
        embeddings.append(self.model.get_patch_encodings(img_t))

        embeddings = torch.cat(embeddings, dim=0)

        h_out, w_out = self.get_output_dims(img_t.shape[-2], img_t.shape[-1])

        embeddings = rearrange(embeddings, "b (h w) c -> b h w c", h=h_out, w=w_out)  # (H, W, D)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.squeeze(0)

        if self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Nothing to do except recover the size
            # We need to permute channels to do the recover size correctly TODO(dvdmc): can we improve this?
            self.semantics = (
                recover_size(embeddings.permute(2, 0, 1)).permute(1, 2, 0).cpu().numpy()
            )
            return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

        # Compute similarities
        sims = embeddings @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)

        # NOTE: Careful with channel ordering here. It differs from other sem seg modules!

        # Normalize to get "probabilities"
        probs = (sims / sims.norm(dim=-1, keepdim=True)).permute(2, 0, 1)
        probs = recover_size(probs)

        if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            self.semantics = probs.permute(1, 2, 0).cpu().numpy()
            return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

        # Get the label
        pred = probs.argmax(dim=0)
        # if self.semantic_feature_type == SemanticFeatureType.LABEL: # NOT NECESSARY FOR NOW
        self.semantics = pred.cpu().numpy()
        return SemanticSegmentationOutput(semantics=self.semantics, instances=None)

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        return self.sem_img_to_rgb(semantics, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantic_img, self.semantic_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(
                np.argmax(semantic_img, axis=-1), self.semantic_color_map, bgr=bgr
            )
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Transform semantic to tensor
            # TODO(dvdmc): check if doing these operations (and functions below) in CPU is more efficient (it probably is)
            semantics = torch.from_numpy(semantic_img).to(self.device)
            # Compute similarity
            sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                return similarity_heatmap_image(
                    sims.cpu().detach().numpy(),
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                pred = sims.argmax(dim=-1)
                return labels_to_image(
                    pred.cpu().detach().numpy(), self.semantic_color_map, bgr=bgr
                )

    def features_to_sims(self, semantics):
        """Public interface to compute similarity

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """

        if self.semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_sims()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        return sims.cpu().detach().numpy()

    def features_to_labels(self, semantics):
        """Public interface to compute labels

        Args:
            semantics (np.ndarray): Semantic features of generic shape ([1] or [H, W], D)
        """
        if self.semantic_feature_type != SemanticFeatureType.FEATURE_VECTOR:
            print(
                "WARNING: if you computed semantics from this module, they shouldn't be used with features_to_labels()"
            )
        # Transform semantic to tensor
        semantics = torch.from_numpy(semantics).to(self.device)
        # Compute similarity
        sims = semantics @ self.text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
        pred = sims.argmax(dim=-1)
        return pred.cpu().detach().numpy()
