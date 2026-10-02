#!/usr/bin/env bash
# This file is part of https://github.com/luigifreda/pyslam
#
# Install optional pySLAM components ("extras") on request, on top of the core installation
# (pyenv-conda-create.sh + the core builds), instead of installing everything at once.
# Each extra fetches only the git submodules it needs, applies pySLAM's patches, downloads the
# model weights and then checks every component on the bundled test images (scripts/extras_check.py).
# It can be re-run safely: patches already applied and files already downloaded are skipped.
#
# usage: scripts/install_extra.sh --list
#        scripts/install_extra.sh <extra> [<extra> ...]      e.g. scripts/install_extra.sh features vpr

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils
. "$ROOT_DIR"/bash_utils.sh

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

# ====================================================

PYTHON_EXE=$(get_python_exe)

EXTRAS_ORDER="features vpr depth semantics scene3d tf"

# extra_description <extra>: print the description, or nothing for an unknown extra
# (a function rather than an associative array, which needs bash >= 4; macOS ships bash 3.2)
function extra_description() {
    case "$1" in
        features) echo "Learned local features and matchers: SuperPoint, LightGlue, XFeat, DISK, ALIKED, D2-Net, R2D2, Key.Net, HardNet, SOSNet, TFeat, L2-Net, LoFTR" ;;
        vpr) echo "Visual place recognition loop detectors: NetVLAD, CosPlace, EigenPlaces, MegaLoc, AlexNet" ;;
        depth) echo "Depth and stereo estimation: Depth Anything V2, Depth Pro, RAFT-Stereo, CREStereo (PyTorch), Depth Anything V3 [pixi level: depth]" ;;
        semantics) echo "Semantic segmentation and object detection: DeepLabV3, SegFormer, YOLO, RF-DETR, CLIP, Detic, EOV-Seg, ODISE [pixi level: semantics]" ;;
        scene3d) echo "3D representations: MASt3R, DUSt3R, MV-DUSt3R, VGGT, Robust VGGT, Fast3R, Gaussian splatting (need an NVIDIA GPU) [pixi level: full]" ;;
        tf) echo "TensorFlow-based features: DELF, LF-Net, ContextDesc, GeoDesc, and the HDC-DELF place recognition [pixi level: tf]" ;;
    esac
}

function list_extras() {
    echo "Available extras:"
    for e in $EXTRAS_ORDER; do
        printf "  %-10s %s\n" "$e" "$(extra_description "$e")"
    done
}

# init_submodules <path> ...: fetch only the given git submodules (recursively)
function init_submodules() {
    print_blue "Fetching submodules: $*"
    git submodule update --init --recursive -- "$@" || { print_red "ERROR: could not fetch submodules: $*"; exit 1; }
}

# apply_patch <thirdparty dir> <patch file in thirdparty/>: apply unless it is already applied
function apply_patch() {
    local dir="$ROOT_DIR/thirdparty/$1" patch="$ROOT_DIR/thirdparty/$2"
    if git -C "$dir" apply --reverse --check "$patch" &>/dev/null; then
        echo "patch $2 already applied"
    elif git -C "$dir" apply --check "$patch" &>/dev/null; then
        git -C "$dir" apply "$patch" && echo "patch $2 applied" || { print_red "ERROR: could not apply $2"; exit 1; }
    else
        print_red "ERROR: $2 does not apply to thirdparty/$1 (local changes?)"
        exit 1
    fi
}

# clone_repo <thirdparty dir> <url> [<commit>]: clone into thirdparty/<dir> unless it is already there,
# and check out <commit> when given (the version pySLAM's patch was made for). An existing clone at
# another commit (e.g. from an older install) is moved to <commit> if it has no local changes.
function clone_repo() {
    local dir="$ROOT_DIR/thirdparty/$1" url="$2" commit="$3" head
    if [ -e "$dir/.git" ]; then
        if [ -n "$commit" ]; then
            head=$(git -C "$dir" rev-parse HEAD 2>/dev/null)
            if [ "$head" != "$commit" ]; then
                if [ -z "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
                    print_blue "thirdparty/$1 is at ${head:0:7}: checking out ${commit:0:7} ..."
                    { git -C "$dir" checkout -q "$commit" 2>/dev/null \
                        || { git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$commit"; }; } \
                        || { print_red "ERROR: could not check out $commit in thirdparty/$1"; exit 1; }
                else
                    print_yellow "thirdparty/$1 is at ${head:0:7} with local changes; pySLAM's patch was made for ${commit:0:7}."
                    print_yellow "  If the next step fails, remove thirdparty/$1 and re-run this script."
                fi
            fi
        fi
        echo "thirdparty/$1 already cloned"
        return 0
    fi
    print_blue "Cloning $url into thirdparty/$1 ..."
    rm -rf "$dir"
    git clone "$url" "$dir" || { print_red "ERROR: could not clone $url"; exit 1; }
    if [ -n "$commit" ]; then
        git -C "$dir" checkout -q "$commit" || { print_red "ERROR: could not check out $commit in thirdparty/$1"; exit 1; }
    fi
}

# download_file <url> <file>: download unless the file is already there
function download_file() {
    if [ ! -s "$2" ]; then
        print_blue "Downloading $(basename "$2") ..."
        mkdir -p "$(dirname "$2")"
        # resumable: an interrupted download leaves <file>.part, which the next run continues
        wget -q --show-progress --tries=5 --timeout=60 --continue -O "$2.part" "$1" && mv "$2.part" "$2" \
            || { print_red "ERROR: could not download $1 (run this script again to resume)"; exit 1; }
    fi
}

# gdrive_download_file <Google Drive file id> <file>: download from Google Drive unless the file is
# already there. gdown alone has no timeout and writes straight to the final name; this goes through
# pySLAM's downloader, which restarts (and resumes) a stalled download and renames only on success.
function gdrive_download_file() {
    if [ ! -s "$2" ]; then
        print_blue "Downloading $(basename "$2") ..."
        ensure_python_package "$PYTHON_EXE" gdown gdown || exit 1
        mkdir -p "$(dirname "$2")"
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_EXE" -c '
import sys
from pyslam.utilities.file_management import gdrive_download_with_retry
gdrive_download_with_retry("https://drive.google.com/uc?id=" + sys.argv[1], sys.argv[2])' "$1" "$2" \
            || { print_red "ERROR: could not download $(basename "$2") from Google Drive (run this script again to resume)"; exit 1; }
    fi
}

# component <name> <function>: install one component in a subshell, so that a failure (a download
# that breaks, a build error) is reported and the other components of the extra are still installed
COMPONENT_FAILURES=""
function component() {
    local name="$1"
    shift
    if ! ( "$@" ); then
        print_red "ERROR: $name was not installed (see above); continuing with the other components"
        COMPONENT_FAILURES="$COMPONENT_FAILURES${COMPONENT_FAILURES:+, }$name"
    fi
}

function install_features() {
    init_submodules thirdparty/superpoint thirdparty/LightGlue thirdparty/accelerated_features \
        thirdparty/disk thirdparty/d2net thirdparty/r2d2 thirdparty/keynet thirdparty/hardnet \
        thirdparty/SOSNet thirdparty/tfeat thirdparty/logpolar
    apply_patch d2net d2net.patch
    apply_patch r2d2 r2d2.patch
    apply_patch keynet keynet.patch
    apply_patch LightGlue lightglue.patch
    # D2-Net weights (the other models are downloaded by the check below, on first creation)
    component "D2-Net" install_d2net_model
}

function install_d2net_model() {
    if [ ! -f thirdparty/d2net/models/d2_ots.pth ]; then
        gdrive_download_file "12Uk95TjBT7VZSEitvm3B3XNK37Q_uU8T" thirdparty/d2net/models/d2net.tar.xz
        ( cd thirdparty/d2net/models && tar -xf d2net.tar.xz && rm d2net.tar.xz ) \
            || { print_red "ERROR: could not unpack the D2-Net model"; exit 1; }
    fi
}

function install_netvlad_model() {
    # NetVLAD weights: pySLAM uses configs/netvlad_extract.ini, i.e. the Mapillary model with 512
    # principal components (92 MB). Without this file Patch-NetVLAD downloads all its seven models
    # (2.9 GB) at the first use, with a downloader that cannot resume.
    download_file "https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/mapillary_WPCA512.pth.tar?download=true" \
        thirdparty/patch_netvlad/patchnetvlad/pretrained_models/mapillary_WPCA512.pth.tar
}

function install_vpr() {
    init_submodules thirdparty/vpr thirdparty/patch_netvlad
    apply_patch vpr vpr.patch
    apply_patch patch_netvlad patch_netvlad.patch
    component "NetVLAD" install_netvlad_model
    # torch >= 2.13 asks "Do you trust this repository?" on the first torch.hub.load of a repo, which
    # a loop-detection child process cannot answer: trust the repos used by the VPR detectors.
    "$PYTHON_EXE" - <<'EOF' || exit 1
import os, torch
path = os.path.join(torch.hub.get_dir(), "trusted_list")
os.makedirs(os.path.dirname(path), exist_ok=True)
trusted = set(open(path).read().split()) if os.path.exists(path) else set()
new = [r for r in ("gmberton_cosplace", "gmberton_eigenplaces", "gmberton_MegaLoc") if r not in trusted]
if new:
    with open(path, "a") as f:
        f.write("".join(r + "\n" for r in new))
print("torch.hub trusted repos:", ", ".join(sorted(trusted | set(new))))
EOF
}

function install_depth_pro() {
    # Depth Pro (monocular, metric)
    clone_repo ml_depth_pro https://github.com/apple/ml-depth-pro.git
    apply_patch ml_depth_pro ml_depth_pro.patch
    download_file https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt thirdparty/ml_depth_pro/checkpoints/depth_pro.pt   # 1.9 GB
}

function install_depth_anything_v2() {
    # Depth Anything V2 (monocular, metric indoor/outdoor models)
    clone_repo depth_anything_v2 https://github.com/DepthAnything/Depth-Anything-V2.git
    apply_patch depth_anything_v2 depth_anything_v2.patch
    local dataset size model
    for dataset in Hypersim VKITTI; do
        for size in Small:vits Base:vitb Large:vitl; do
            model="depth_anything_v2_metric_$(echo "$dataset" | tr '[:upper:]' '[:lower:]')_${size#*:}.pth"
            download_file "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-$dataset-${size%:*}/resolve/main/$model?download=true" \
                "thirdparty/depth_anything_v2/metric_depth/checkpoints/$model"
        done
    done
}

function install_raft_stereo() {
    # (thirdparty/raft_stereo.patch only fixed the model links, which upstream has fixed since)
    clone_repo raft_stereo https://github.com/princeton-vl/RAFT-Stereo.git 6e93ed2169bd858dbb43033988563f3b0bb49506
    if [ ! -f thirdparty/raft_stereo/models/raftstereo-middlebury.pth ]; then
        # the link is the one of RAFT-Stereo's download_models.sh
        download_file "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1" \
            thirdparty/raft_stereo/models/models.zip
        ( cd thirdparty/raft_stereo/models && unzip -o -q models.zip && rm -f models.zip ) \
            || { print_red "ERROR: could not unpack the RAFT-Stereo models"; exit 1; }
    fi
}

function install_crestereo_pytorch() {
    # CREStereo, PyTorch port (the original needs MegEngine, which is not supported here)
    clone_repo crestereo_pytorch https://github.com/ibaiGorordo/CREStereo-Pytorch.git
    apply_patch crestereo_pytorch crestereo_pytorch.patch
    gdrive_download_file "1pNVdaSvkgCK9NuU1i66nZVNg87VsGxox" thirdparty/crestereo_pytorch/models/crestereo_eth3d.pth
}

function install_depth_anything_v3() {
    # Depth Anything V3 (monocular); its weights are downloaded by the check below, on first creation
    clone_repo depth_anything_v3 https://github.com/ByteDance-Seed/depth-anything-3 ed6989a23cd389e975ed9f7cbd7385396e6d867e
    apply_patch depth_anything_v3 depth_anything_v3.patch
}

function install_depth() {
    component "Depth Pro" install_depth_pro
    component "Depth Anything V2" install_depth_anything_v2
    component "RAFT-Stereo" install_raft_stereo
    component "CREStereo" install_crestereo_pytorch
    component "Depth Anything V3" install_depth_anything_v3
}

# unpack_pypi_package <requirement> <thirdparty dir>: download the wheel of a pure-Python package from
# PyPI and unpack it into thirdparty/<dir>, WITHOUT installing it or its dependencies (for packages
# whose declared dependencies would replace the environment's packages). pySLAM adds the folder to
# sys.path through config_libs.yaml.
function unpack_pypi_package() {
    local dir="$ROOT_DIR/thirdparty/$2" tmp_dir
    print_blue "Fetching $1 into thirdparty/$2 (without its dependencies) ..."
    tmp_dir=$(mktemp -d)
    "$PYTHON_EXE" -m pip download --no-deps --only-binary :all: -q -d "$tmp_dir" "$1" \
        && rm -rf "$dir" && mkdir -p "$dir" \
        && "$PYTHON_EXE" -m zipfile -e "$tmp_dir"/*.whl "$dir" \
        || { rm -rf "$tmp_dir"; print_red "ERROR: could not fetch $1 from PyPI"; exit 1; }
    rm -rf "$tmp_dir"
}

function install_rf_detr() {
    clone_repo rf_detr https://github.com/roboflow/rf-detr.git fd1295b8ccacba0fad2b4a40c8a35b67bc62c335
    apply_patch rf_detr rf_detr.patch
}

function install_detic() {
    # Detic (with its CenterNet2 submodule)
    if [ ! -d thirdparty/detic/.git ]; then
        clone_repo detic https://github.com/facebookresearch/Detic.git 436cda2a2347df60a7c66daca0e8c59f93dc5e79
        git -C thirdparty/detic submodule update --init --recursive || { print_red "ERROR: could not fetch Detic's submodules"; exit 1; }
    fi
    apply_patch detic detic.patch
    apply_patch detic/third_party/CenterNet2 detic/third_party/centernet2.patch  # created by detic.patch
    download_file https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        thirdparty/detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
}

function install_eov_seg() {
    clone_repo eov_segmentation https://github.com/nhw649/EOV-Seg.git 6f5e93e9aca6ccae89fe492b24018f8530075fc4
    apply_patch eov_segmentation eov_segmentation.patch
    gdrive_download_file "1dVfHpzmCOlV6hLfUpd3nHXz62wdB7RY2" thirdparty/eov_segmentation/checkpoints/convnext-l.pth   # 2.1 GB
}

function install_odise() {
    clone_repo odise https://github.com/NVlabs/ODISE.git 2b187e4b2ff4c3d5da342aec2cc234b537720a65
    apply_patch odise odise.patch
    # ODISE is imported from its source folder (no pip install). Its model zoo looks for the configs
    # inside the package, where its setup.py would have linked them.
    if [ ! -e thirdparty/odise/odise/model_zoo/configs ]; then
        ln -s ../../configs thirdparty/odise/odise/model_zoo/configs
    fi
    # ODISE uses the `ldm` module of stable-diffusion-sdkit. That package pins old versions of
    # OpenCV, kornia, transformers, ... so only the module is unpacked, without its dependencies.
    if [ ! -f thirdparty/stable_diffusion_sdkit/ldm/models/diffusion/ddpm.py ]; then
        unpack_pypi_package "stable-diffusion-sdkit==2.1.5" stable_diffusion_sdkit
        # pytorch-lightning >= 1.8 moved rank_zero_only
        sed -i.bak 's/from pytorch_lightning.utilities.distributed import rank_zero_only/from pytorch_lightning.utilities.rank_zero import rank_zero_only/' \
            thirdparty/stable_diffusion_sdkit/ldm/models/diffusion/ddpm.py && rm -f thirdparty/stable_diffusion_sdkit/ldm/models/diffusion/ddpm.py.bak
    fi
}

function install_clip_segmentation() {
    # CLIP segmentation uses the CLIP module of f3rm. The package depends on nerfstudio, which is
    # not needed for that module, so it is unpacked without its dependencies as well.
    if [ ! -f thirdparty/f3rm_pkg/f3rm/features/clip/clip.py ]; then
        unpack_pypi_package "f3rm==0.0.6" f3rm_pkg
    fi
}

function install_semantics() {
    # DeepLabV3, SegFormer and YOLO need no code here: their weights are downloaded by the check below.
    component "RF-DETR" install_rf_detr
    component "Detic" install_detic
    component "EOV-Seg" install_eov_seg
    component "ODISE" install_odise
    component "CLIP segmentation" install_clip_segmentation
}

# cuda_home: the CUDA toolkit that torch's cpp_extension should use (it takes nvcc from $CUDA_HOME/bin):
# CUDA_HOME if set, otherwise the prefix of the nvcc on the PATH (the environment under conda/pixi
# with cuda-nvcc, /usr or /usr/local/cuda-X for a system toolkit). The path is not resolved: conda's
# bin/nvcc may be a link into targets/.
function cuda_home() {
    if [ -n "$CUDA_HOME" ]; then
        echo "$CUDA_HOME"
    else
        dirname "$(dirname "$(command -v nvcc)")"
    fi
}

# has_cuda: true if torch can use an NVIDIA GPU
function has_cuda() {
    "$PYTHON_EXE" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

# build_curope <dir>: build the CUDA extension of CroCo's RoPE (used by DUSt3R/MASt3R/MV-DUSt3R) in place.
# It needs the CUDA compiler of the environment (nvcc; cuda-nvcc in a conda/pixi environment).
function build_curope() {
    local dir="$ROOT_DIR/thirdparty/$1"
    if ls "$dir"/curope*.so &>/dev/null; then
        echo "curope already built in thirdparty/$1"
        return 0
    fi
    if ! command -v nvcc &>/dev/null; then
        print_yellow "No CUDA compiler (nvcc): not building curope in thirdparty/$1. The model still works, with a slower PyTorch implementation of RoPE."
        return 0
    fi
    print_blue "Building curope in thirdparty/$1 ..."
    ( cd "$dir" && CUDA_HOME="$(cuda_home)" MAX_JOBS="$(get_build_jobs)" "$PYTHON_EXE" setup.py build_ext --inplace ) \
        || { print_red "ERROR: could not build curope in thirdparty/$1 (is the CUDA compiler nvcc available?)"; exit 1; }
}

# build_cuda_extension_inplace <thirdparty dir> <package>: build a torch CUDA extension with
# `setup.py build_ext --inplace`, so that <package>/_C*.so ends up next to the sources
function build_cuda_extension_inplace() {
    local dir="$ROOT_DIR/thirdparty/$1"
    if ls "$dir/$2"/_C*.so &>/dev/null; then
        echo "$2 already built"
        return 0
    fi
    print_blue "Building $2 in thirdparty/$1 ..."
    ( cd "$dir" && CUDA_HOME="$(cuda_home)" MAX_JOBS="$(get_build_jobs)" "$PYTHON_EXE" setup.py build_ext --inplace ) \
        || { print_red "ERROR: could not build $2 (is the CUDA compiler nvcc available?)"; exit 1; }
}

function install_mast3r() {
    # MASt3R (with DUSt3R and CroCo as submodules)
    if [ ! -d thirdparty/mast3r/.git ]; then
        clone_repo mast3r https://github.com/naver/mast3r f5209afc300cec36239a7ac992263f36847bbba0
        git -C thirdparty/mast3r submodule update --init --recursive || { print_red "ERROR: could not fetch MASt3R's submodules"; exit 1; }
    fi
    apply_patch mast3r mast3r.patch
    apply_patch mast3r/dust3r mast3r-dust3r.patch
    apply_patch mast3r/dust3r/croco mast3r-dust3r-croco.patch
    build_curope mast3r/dust3r/croco/models/curope
    download_file https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        thirdparty/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
}

function install_mvdust3r() {
    clone_repo mvdust3r https://github.com/facebookresearch/mvdust3r.git 430ca6630b07567cfb2447a4dcee9747b132d5c7
    apply_patch mvdust3r mvdust3r.patch
    build_curope mvdust3r/croco/models/curope
    # pySLAM uses the MV-DUSt3R model MVD.pth (6.6 GB). The other checkpoints of MV-DUSt3R (MV-DUSt3R+
    # stages 1 and 2 and DUSt3R 224, 18 GB more) are not used: for those, see
    # thirdparty/mvdust3r_scripts/download_models.py.
    download_file https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/MVD.pth thirdparty/mvdust3r/checkpoints/MVD.pth
}

function install_vggt() {
    # VGGT and Robust VGGT (their weights are downloaded by the check below, on first creation)
    clone_repo vggt https://github.com/facebookresearch/vggt.git a288dd0f14786c93483e45524328726ab7b1b4ce
    clone_repo vggt_robust https://github.com/cvlab-kaist/RobustVGGT.git 0763ed6484b1e91a2b8bd5072d317745743492cc
}

function install_fast3r() {
    clone_repo fast3r https://github.com/facebookresearch/fast3r.git 33104d4b5b8df43795ecded236194958bbdac572
    apply_patch fast3r fast3r.patch
}

function install_gaussian_splatting() {
    # Gaussian splatting (MonoGS): three CUDA extensions, built in the source tree and found through
    # config_libs.yaml (nothing is installed into the environment, and no sudo is needed)
    if ! command -v nvcc &>/dev/null; then
        print_yellow "No CUDA compiler (nvcc): skipping the Gaussian splatting extensions (simple-knn, diff-gaussian-rasterization, lietorch)."
        print_yellow "  The pixi level 'full' provides nvcc. In a conda environment, install the compiler matching torch's CUDA version, e.g.:"
        print_yellow "  conda install -c conda-forge --override-channels cuda-nvcc cuda-cudart-dev cuda-libraries-dev \"cuda-version=$("$PYTHON_EXE" -c 'import torch; print(torch.version.cuda)' 2>/dev/null)\""
        return 0
    fi
    # The MonoGS GUI imports the Python binding of GLFW. The pixi level `full` has it; elsewhere install
    # it into the active environment (conda-forge's `pyglfw`; its `glfw` is only the C library).
    if ! "$PYTHON_EXE" -c "import glfw" &>/dev/null; then
        if [[ -n "$PIXI_PROJECT_NAME" ]]; then
            print_red "ERROR: the Python module glfw is missing: use the pixi level 'full' (pixi run -e full models-scene3d)"
            exit 1
        elif [[ -n "$CONDA_PREFIX" ]] && command -v conda &>/dev/null; then
            print_blue "Installing pyglfw (conda-forge) into the conda environment ..."
            conda install -y -p "$CONDA_PREFIX" -c conda-forge --override-channels pyglfw \
                || { print_red "ERROR: could not install pyglfw"; exit 1; }
        else
            ensure_python_package "$PYTHON_EXE" glfw glfw || exit 1
        fi
    fi
    build_cuda_extension_inplace monogs/submodules/simple-knn simple_knn
    build_cuda_extension_inplace monogs/submodules/diff-gaussian-rasterization diff_gaussian_rasterization
    if [ ! -f thirdparty/lietorch/install/lietorch_backends.so ]; then
        print_blue "Building lietorch ..."
        ( cd thirdparty/lietorch && rm -rf build install \
            && cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DSITE_PACKAGES_DIR="$ROOT_DIR/thirdparty/lietorch/install" \
            && cmake --build build -j "$(get_build_jobs)" && cmake --install build ) \
            || { print_red "ERROR: could not build lietorch (are nvcc, cmake and ninja available?)"; exit 1; }
    else
        echo "lietorch already built"
    fi
}

function install_scene3d() {
    if ! has_cuda; then
        print_yellow "The scene3d models (MASt3R, DUSt3R, MV-DUSt3R, VGGT, Fast3R) need an NVIDIA GPU with CUDA: skipping them on this machine."
        return 0
    fi
    component "MASt3R and DUSt3R" install_mast3r
    component "MV-DUSt3R" install_mvdust3r
    component "VGGT and Robust VGGT" install_vggt
    component "Fast3R" install_fast3r
    component "Gaussian splatting" install_gaussian_splatting
}

function install_contextdesc() {
    # ContextDesc (its code is part of this repository)
    local dir=thirdparty/contextdesc/pretrained name
    for name in "contextdesc++:1TQIjijkyd3fNvEivPPpnxHKaSqFxu5TE" "retrieval_model:1_J_aDSdKcUUk0ZXhn9bTqV6zuyzUixLD"; do
        if [ ! -d "$dir/${name%:*}" ]; then
            gdrive_download_file "${name#*:}" "$dir/${name%:*}.tar.xz"
            ( cd "$dir" && tar -xf "${name%:*}.tar.xz" && rm "${name%:*}.tar.xz" ) \
                || { print_red "ERROR: could not unpack the ContextDesc model ${name%:*}"; exit 1; }
        fi
    done
}

function install_lfnet() {
    apply_patch lfnet lfnet.patch
    [ -f thirdparty/lfnet/__init__.py ] || touch thirdparty/lfnet/__init__.py
    if [ ! -d thirdparty/lfnet/pretrained/lfnet-norotaug ]; then
        download_file https://cs.ubc.ca/research/kmyi_data/files/2018/lf-net/lfnet-norotaug.tar.gz thirdparty/lfnet/pretrained/lfnet-norotaug.tar.gz
        tar -C thirdparty/lfnet/pretrained -xf thirdparty/lfnet/pretrained/lfnet-norotaug.tar.gz \
            || { print_red "ERROR: could not unpack the LF-Net model"; exit 1; }
    fi
}

function install_geodesc() {
    # GeoDesc (its code is part of this repository)
    download_file https://raw.githubusercontent.com/lzx551402/geodesc/master/model/geodesc.pb thirdparty/geodesc/model/geodesc.pb
}

function install_delf() {
    # DELF (also used by the HDC-DELF place recognition): compile its protocol buffers with the
    # environment's protoc, so that the generated code matches the installed protobuf
    local delf_dir=thirdparty/tensorflow_models/research/delf
    if ! command -v protoc &>/dev/null; then
        print_red "ERROR: protoc not found (it comes with libprotobuf in a conda/pixi environment)"
        exit 1
    fi
    if [ ! -f "$delf_dir/delf/protos/.protoc_version" ] || [ "$(cat "$delf_dir/delf/protos/.protoc_version")" != "$(protoc --version)" ]; then
        print_blue "Compiling DELF's protocol buffers with $(protoc --version) ..."
        ( cd "$delf_dir" && protoc delf/protos/*.proto --python_out=. && protoc --version > delf/protos/.protoc_version ) \
            || { print_red "ERROR: could not compile DELF's protocol buffers"; exit 1; }
    fi
    if [ ! -d "$delf_dir/delf/python/examples/parameters/delf_gld_20190411" ]; then
        download_file http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz "$delf_dir/delf/python/examples/parameters/delf_gld_20190411.tar.gz"
        tar -C "$delf_dir/delf/python/examples/parameters" -xf "$delf_dir/delf/python/examples/parameters/delf_gld_20190411.tar.gz" \
            || { print_red "ERROR: could not unpack the DELF model"; exit 1; }
    fi
}

function install_tf() {
    if ! "$PYTHON_EXE" -c "import tensorflow" &>/dev/null; then
        print_yellow "TensorFlow is not installed in this environment: skipping the TensorFlow-based features."
        print_yellow "  With pixi, use the level 'tf' (pixi run -e tf models-tf)."
        return 0
    fi

    init_submodules thirdparty/lfnet thirdparty/tensorflow_models thirdparty/vpr
    component "ContextDesc" install_contextdesc
    component "LF-Net" install_lfnet
    component "GeoDesc" install_geodesc
    component "DELF" install_delf

    # HDC-DELF place recognition
    apply_patch vpr vpr.patch
}

if [[ $# -eq 0 || "$1" == "--list" || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "usage: $0 --list | <extra> [<extra> ...]"
    list_extras
    cd "$STARTING_DIR"
    [[ $# -eq 0 ]] && exit 1 || exit 0
fi

for extra in "$@"; do
    if [[ -z "$(extra_description "$extra")" ]]; then
        print_red "ERROR: unknown extra '$extra'"
        list_extras
        exit 1
    fi
done

FAILED=""
for extra in "$@"; do
    print_blue '================================================'
    print_blue "Installing extra '$extra': $(extra_description "$extra")"
    print_blue '================================================'
    install_$extra
    print_blue "Checking '$extra' and downloading its model weights (first run can take a while) ..."
    "$PYTHON_EXE" "$SCRIPTS_DIR/extras_check.py" "$extra" || FAILED="$FAILED $extra"
done

cd "$STARTING_DIR"
if [[ -n "$COMPONENT_FAILURES" ]]; then
    print_red "Not installed: $COMPONENT_FAILURES (see the errors above). Run this script again: it skips what is"
    print_red "already installed and resumes interrupted downloads."
fi
if [[ -n "$FAILED" ]]; then
    print_red "Some components failed the check in:$FAILED (see above)"
fi
if [[ -n "$COMPONENT_FAILURES$FAILED" ]]; then
    exit 1
fi
print_green "Installed: $*"
