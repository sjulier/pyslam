"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
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

import os
import time
import math
import numpy as np
import platform

from pyslam.utilities.logging import Printer

from pyslam.config_parameters import Parameters

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

from pyslam.utilities.serialization import Serializable, register_class

kVerbose = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# NOTE: At present, under mac, boost serialization is very slow, we use txt files instead.
def dbow2_orb_vocabulary_factory(*args, **kwargs):
    use_text_vocabulary = platform.system() == "Darwin"
    if use_text_vocabulary:
        return DBowOrbVocabularyDataTxt(*args, **kwargs)
    else:
        return DBow2OrbVocabularyData(*args, **kwargs)


# NOTE: at present, under mac, boost serialization is very slow, we use txt files instead.
def dbow3_orb_vocabulary_factory(*args, **kwargs):
    use_text_vocabulary = platform.system() == "Darwin"
    if use_text_vocabulary:
        return DBowOrbVocabularyDataTxt(*args, **kwargs)
    else:
        return DBow3OrbVocabularyData(*args, **kwargs)


# Sizes in bytes of the vocabulary files that pySLAM downloads. A file with another size is an
# interrupted download (older versions wrote directly to the final name).
kVocabularyFileSizes = {
    "ORBvoc.txt": 145250924,
    "ORBvoc.dbow2": 105381767,
    "ORBvoc.dbow3": 105381669,
}


# The ORB vocabularies are also published, xz-compressed (about 35 MB each), with pySLAM's prebuilt
# native modules: downloaded from there first, from Google Drive if that fails
kVocabularyReleaseUrl = os.environ.get(
    "PYSLAM_VOCABULARY_URL", "https://github.com/sjulier/pyslam/releases/download/native-bundles"
)


@register_class
class VocabularyData(Serializable):
    def __init__(
        self,
        vocab_file_path=None,
        descriptor_type=None,
        descriptor_dimension=None,
        url_vocabulary=None,
        url_type=None,
    ):
        self.vocab_file_path = vocab_file_path
        self.descriptor_type = descriptor_type
        self.descriptor_dimension = descriptor_dimension
        self.url_vocabulary = url_vocabulary
        self.url_type = url_type

    def set_aside_incomplete_file(self):
        """Rename a downloaded vocabulary file that is incomplete, so that it is downloaded again."""
        if self.url_vocabulary is None or self.vocab_file_path is None:
            return
        expected_size = kVocabularyFileSizes.get(os.path.basename(self.vocab_file_path))
        if expected_size is None or not os.path.exists(self.vocab_file_path):
            return
        size = os.path.getsize(self.vocab_file_path)
        if size != expected_size:
            incomplete_path = self.vocab_file_path + ".incomplete"
            Printer.yellow(
                f"VocabularyData: {self.vocab_file_path} has {size} bytes instead of {expected_size}: "
                f"it is an incomplete download. Moving it to {incomplete_path} and downloading it again."
            )
            os.replace(self.vocab_file_path, incomplete_path)

    def copy_from_mirror(self):
        """With PYSLAM_MODEL_MIRROR (a local copy of the model files, see scripts/install_extra.sh),
        take the vocabulary from <mirror>/checkout/data/ instead of downloading it."""
        mirror = os.environ.get("PYSLAM_MODEL_MIRROR")
        if not mirror or self.vocab_file_path is None or os.path.exists(self.vocab_file_path):
            return
        src = os.path.join(mirror, "checkout", "data", os.path.basename(self.vocab_file_path))
        if os.path.isfile(src):
            import shutil

            Printer.blue(f"VocabularyData: copying {os.path.basename(src)} from the local model mirror")
            os.makedirs(os.path.dirname(self.vocab_file_path), exist_ok=True)
            shutil.copyfile(src, self.vocab_file_path + ".part")
            os.replace(self.vocab_file_path + ".part", self.vocab_file_path)

    def download_from_release(self):
        """Download <name>.xz from kVocabularyReleaseUrl and unpack it. False if that fails."""
        name = os.path.basename(self.vocab_file_path)
        expected_size = kVocabularyFileSizes.get(name)
        if expected_size is None or not kVocabularyReleaseUrl:
            return False
        import lzma
        import shutil
        import urllib.request

        url = f"{kVocabularyReleaseUrl.rstrip('/')}/{name}.xz"
        part = self.vocab_file_path + ".part"
        Printer.blue(f"VocabularyData: downloading vocabulary {name} from: {url}")
        try:
            os.makedirs(os.path.dirname(self.vocab_file_path), exist_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "pyslam-vocabulary"})
            with urllib.request.urlopen(request, timeout=60) as response:
                with lzma.open(response) as unpacked, open(part, "wb") as f:
                    shutil.copyfileobj(unpacked, f, 1 << 20)
            if os.path.getsize(part) != expected_size:
                raise ValueError(f"{os.path.getsize(part)} bytes instead of {expected_size}")
            os.replace(part, self.vocab_file_path)
            return True
        except Exception as e:  # noqa: BLE001  (network errors, a corrupt or missing file)
            Printer.yellow(f"VocabularyData: download from {url} failed ({e}): trying {self.url_type}")
            if os.path.exists(part):
                os.remove(part)
            return False

    def check_download(self):
        self.set_aside_incomplete_file()
        self.copy_from_mirror()
        if self.url_vocabulary is not None and not os.path.exists(self.vocab_file_path):
            self.download_from_release()
        if self.url_vocabulary is not None and not os.path.exists(self.vocab_file_path):
            if self.url_type == "gdrive":
                gdrive_url = self.url_vocabulary
                Printer.blue(
                    f"VocabularyData: downloading vocabulary {self.descriptor_type.name} from: {gdrive_url}"
                )
                try:
                    gdrive_download_lambda(url=gdrive_url, path=self.vocab_file_path)
                except Exception as e:
                    Printer.red(f"VocabularyData: cannot download vocabulary from {gdrive_url}")
                    raise e
        if self.vocab_file_path is not None and not os.path.exists(self.vocab_file_path):
            Printer.red(f"VocabularyData: cannot find vocabulary file: {self.vocab_file_path}")
            raise FileNotFoundError


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBowOrbVocabularyDataTxt(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.txt"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBow2OrbVocabularyData(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.dbow2"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=1pvBERLLSUV4IcaInNJMURTb8p-r9-5Xf",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBow3OrbVocabularyData(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.dbow3"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


@register_class
class VladOrbVocabularyData(VocabularyData):
    kVladVocabFile = kDataFolder + "/VLADvoc_orb.txt"

    def __init__(
        self,
        vocab_file_path=kVladVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/file/d/1u6AJEa2aZg7u5aS6vFX2qXiKKePQ_6t2",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )
