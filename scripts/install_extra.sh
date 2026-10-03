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

EXTRAS_ORDER="features-core features vpr-core vpr depth semantics scene3d"

# extra_description <extra>: print the description, or nothing for an unknown extra
# (a function rather than an associative array, which needs bash >= 4; macOS ships bash 3.2)
function extra_description() {
    case "$1" in
        features-core) echo "The recommended learned features: SuperPoint, and SuperPoint with the LightGlue matcher" ;;
        features) echo "All learned local features and matchers: SuperPoint, LightGlue, XFeat, DISK, ALIKED, D2-Net, R2D2, Key.Net, HardNet, SOSNet, TFeat, L2-Net, LoFTR" ;;
        vpr-core) echo "The recommended visual place recognition loop detector: CosPlace" ;;
        vpr) echo "All visual place recognition loop detectors: NetVLAD, CosPlace, EigenPlaces, MegaLoc, AlexNet" ;;
        depth) echo "Depth and stereo estimation: Depth Anything V2, Depth Pro, RAFT-Stereo, CREStereo (PyTorch), Depth Anything V3 [pixi level: depth]" ;;
        semantics) echo "Semantic segmentation and object detection: DeepLabV3, SegFormer, YOLO, RF-DETR, CLIP, Detic, EOV-Seg, ODISE [pixi level: semantics]" ;;
        scene3d) echo "3D representations: MASt3R, DUSt3R, MV-DUSt3R, VGGT, Robust VGGT, Fast3R, Gaussian splatting (need an NVIDIA GPU) [pixi level: full]" ;;
    esac
}

function list_extras() {
    echo "Available extras:"
    for e in $EXTRAS_ORDER; do
        printf "  %-14s %s\n" "$e" "$(extra_description "$e")"
    done
}

# init_submodules <path> ...: fetch only the given git submodules (recursively), each at its recorded
# commit and without its history (--depth 1; git fetches the commit directly when it is not at the tip
# of a branch). If that fails, e.g. with a server that does not allow fetching a commit directly, the
# whole history is fetched.
function init_submodules() {
    print_blue "Fetching submodules: $*"
    git submodule update --init --recursive --depth 1 -- "$@" \
        || git submodule update --init --recursive -- "$@" \
        || { print_red "ERROR: could not fetch submodules: $*"; exit 3; }
}

# apply_patch <thirdparty dir> <patch file in thirdparty/>: apply unless it is already applied
# (--whitespace=nowarn: several patches add lines with trailing spaces; git's warnings about them are
# harmless but look like errors)
function apply_patch() {
    local dir="$ROOT_DIR/thirdparty/$1" patch="$ROOT_DIR/thirdparty/$2"
    if git -C "$dir" apply --reverse --check "$patch" &>/dev/null; then
        echo "patch $2 already applied"
    elif git -C "$dir" apply --check "$patch" &>/dev/null; then
        git -C "$dir" apply --whitespace=nowarn "$patch" && echo "patch $2 applied" || { print_red "ERROR: could not apply $2"; exit 1; }
    elif [[ -n "$PYSLAM_RESET_CLONES" && -e "$dir/.git" ]]; then
        # the clone has other changes, typically an older version of pySLAM's patch: discard the changes
        # to tracked files and the files this patch creates (downloaded weights are left alone)
        print_yellow "thirdparty/$1 has local changes: discarding them to apply $2 (PYSLAM_RESET_CLONES is set)"
        git -C "$dir" checkout -q -- . \
            && git -C "$dir" apply --summary "$patch" | awk '/^ create mode/ {print $4}' | while read -r new_file; do rm -f "$dir/$new_file"; done
        git -C "$dir" apply --whitespace=nowarn "$patch" && echo "patch $2 applied" || { print_red "ERROR: could not apply $2 to thirdparty/$1"; exit 1; }
    else
        print_red "ERROR: $2 does not apply to thirdparty/$1: the folder has local changes, for example an older version of this patch."
        print_red "  To discard them and apply the current patch, run this script again with PYSLAM_RESET_CLONES=1"
        print_red "  (it resets the files tracked by git in the cloned model folders; downloaded weights are kept)."
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
                        || { git -C "$dir" fetch -q --depth 1 origin "$commit" && git -C "$dir" checkout -q "$commit"; } \
                        || { git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$commit"; }; } \
                        || { print_red "ERROR: could not check out $commit in thirdparty/$1"; exit 1; }
                elif [[ -n "$PYSLAM_RESET_CLONES" ]]; then
                    print_yellow "thirdparty/$1 is at ${head:0:7} with local changes: discarding them and checking out ${commit:0:7} (PYSLAM_RESET_CLONES is set)"
                    git -C "$dir" checkout -q -- . \
                        && { git -C "$dir" checkout -q "$commit" 2>/dev/null \
                            || { git -C "$dir" fetch -q --depth 1 origin "$commit" && git -C "$dir" checkout -q "$commit"; } \
                            || { git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$commit"; }; } \
                        || { print_red "ERROR: could not check out $commit in thirdparty/$1"; exit 1; }
                else
                    print_yellow "thirdparty/$1 is at ${head:0:7} with local changes; pySLAM's patch was made for ${commit:0:7}."
                    print_yellow "  If the next step fails, run this script again with PYSLAM_RESET_CLONES=1."
                fi
            fi
        fi
        echo "thirdparty/$1 already cloned"
        return 0
    fi
    # Only the needed commit is fetched, without the history (a shallow clone): some of these
    # repositories are over 100 MB with it. Three attempts, for poor connections.
    print_blue "Cloning $url into thirdparty/$1 ..."
    local attempt
    for attempt in 1 2 3; do
        rm -rf "$dir"
        if [ -n "$commit" ]; then
            git init -q "$dir" && git -C "$dir" remote add origin "$url" \
                && git -C "$dir" fetch -q --depth 1 origin "$commit" && git -C "$dir" checkout -q FETCH_HEAD && return 0
        else
            git clone -q --depth 1 "$url" "$dir" && return 0
        fi
        print_yellow "could not clone $url: attempt $attempt/3"
        sleep 5
    done
    rm -rf "$dir"
    print_red "ERROR: could not clone $url"
    exit 3
}

# update_submodules <thirdparty dir>: fetch the submodules of a cloned repository (shallow if possible)
function update_submodules() {
    local dir="$ROOT_DIR/thirdparty/$1"
    if [[ -n "$PYSLAM_RESET_CLONES" ]]; then
        # submodules that carry an older patch cannot be moved to their recorded commits: discard the
        # changes to their tracked files first (their patches are applied again afterwards)
        git -C "$dir" submodule foreach --quiet --recursive 'git checkout -q -- .'
    fi
    git -C "$dir" submodule update --init --recursive --depth 1 \
        || git -C "$dir" submodule update --init --recursive \
        || { print_red "ERROR: could not fetch the submodules of thirdparty/$1."
             print_red "  If they have local changes (for example an older patch), run this script again with PYSLAM_RESET_CLONES=1."
             exit 1; }
}

# file_size <file>: its size in bytes (GNU and BSD stat)
function file_size() {
    stat -c %s "$1" 2>/dev/null || stat -f %z "$1"
}

# set_aside_incomplete <file> <size>: a file that is present with another size than expected is an
# interrupted download of an older downloader, which wrote to the final name. Make it <file>.part,
# so that the download continues from it.
function set_aside_incomplete() {
    if [ -n "$2" ] && [ -s "$1" ] && [ "$(file_size "$1")" != "$2" ]; then
        print_yellow "$(basename "$1") is incomplete ($(file_size "$1") of $2 bytes): continuing its download"
        if [ "$(file_size "$1")" -lt "$2" ]; then mv "$1" "$1.part"; else rm -f "$1"; fi
    fi
}

# check_downloaded_size <file> <size>: fail if the downloaded <file>.part does not have the expected size
function check_downloaded_size() {
    if [ -n "$2" ] && [ "$(file_size "$1.part")" != "$2" ]; then
        print_red "ERROR: $(basename "$1") has $(file_size "$1.part") bytes after the download, expected $2"
        rm -f "$1.part"
        return 1
    fi
}

# download_file <url> <file> [<size in bytes>]: download unless the file is already there (and, if
# the size is given, complete)
# A local model mirror (for testing, or a lab without internet): PYSLAM_MODEL_MIRROR=<folder> with
#   <folder>/checkout/  model files at their place in a pySLAM checkout (data/..., thirdparty/...)
#   <folder>/home/      the caches of PyTorch, Hugging Face, CLIP, ... at their place in a home folder
# Files found there are copied instead of downloaded. Code (git repositories) still comes from GitHub.
# from_mirror <path>: copy <path> (a file or a folder, relative to the checkout) from the mirror if it
# is there and missing here
function from_mirror() {
    local src="${PYSLAM_MODEL_MIRROR:-}/checkout/$1"
    [[ -n "$PYSLAM_MODEL_MIRROR" && -e "$src" ]] || return 1
    if [[ -d "$src" ]]; then
        [[ -n "$(ls -A "$1" 2>/dev/null)" ]] && return 1
    elif [[ -s "$1" ]]; then
        return 1
    fi
    print_blue "Copying $1 from the local model mirror ..."
    mkdir -p "$(dirname "$1")"
    cp -a "$src" "$(dirname "$1")/" || { print_red "ERROR: could not copy $src"; exit 1; }
}

# with a mirror: its caches go into the home folder once (what is already there is kept)
function caches_from_mirror() {
    if [[ -n "$PYSLAM_MODEL_MIRROR" && -d "$PYSLAM_MODEL_MIRROR/home" ]]; then
        print_blue "Copying the model caches from the local model mirror into $HOME (existing files are kept) ..."
        rsync -a --ignore-existing "$PYSLAM_MODEL_MIRROR/home/" "$HOME/" \
            || { print_red "ERROR: could not copy the caches of $PYSLAM_MODEL_MIRROR"; exit 1; }
    fi
}

function download_file() {
    set_aside_incomplete "$2" "$3"
    [ -s "$2" ] || from_mirror "$2" || true
    set_aside_incomplete "$2" "$3"
    if [ ! -s "$2" ]; then
        print_blue "Downloading $(basename "$2") ..."
        mkdir -p "$(dirname "$2")"
        # resumable: an interrupted download leaves <file>.part, which the next attempt or the next
        # run continues. A connection without data for 60 s is dropped and retried. curl is used
        # where there is no wget (macOS).
        local attempt ok=""
        for attempt in 1 2 3 4 5; do
            if [[ -z "$PYSLAM_DOWNLOAD_WITH_CURL" ]] && command -v wget &>/dev/null; then
                wget -q --show-progress --tries=1 --timeout=60 --continue -O "$2.part" "$1" && ok=1 && break
            else
                curl -fL --connect-timeout 60 --speed-limit 1 --speed-time 60 -C - -o "$2.part" "$1" && ok=1 && break
            fi
            print_yellow "download of $(basename "$2") interrupted: attempt $attempt/5"
            sleep 2
        done
        [ -n "$ok" ] && check_downloaded_size "$2" "$3" && mv "$2.part" "$2" \
            || { print_red "ERROR: could not download $1 (run this script again to resume)"; exit 3; }
    fi
}

# gdrive_download_file <Google Drive file id> <file> [<size in bytes>]: download from Google Drive
# unless the file is already there (and, if the size is given, complete). gdown alone has no timeout
# and writes straight to the final name; this goes through pySLAM's downloader, which restarts (and
# resumes) a stalled download and renames only on success.
function gdrive_download_file() {
    [ -s "$2" ] || from_mirror "$2" || true
    if [ -n "$3" ] && [ -s "$2" ] && [ "$(file_size "$2")" != "$3" ]; then
        print_yellow "$(basename "$2") is incomplete ($(file_size "$2") of $3 bytes): downloading it again"
        rm -f "$2"
    fi
    if [ ! -s "$2" ]; then
        print_blue "Downloading $(basename "$2") ..."
        ensure_python_package "$PYTHON_EXE" gdown gdown || exit 1
        mkdir -p "$(dirname "$2")"
        PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_EXE" -c '
import sys
from pyslam.utilities.file_management import gdrive_download_with_retry
gdrive_download_with_retry("https://drive.google.com/uc?id=" + sys.argv[1], sys.argv[2])' "$1" "$2" \
            || { print_red "ERROR: could not download $(basename "$2") from Google Drive (run this script again to resume)"; exit 3; }
        if [ -n "$3" ] && [ "$(file_size "$2")" != "$3" ]; then
            print_red "ERROR: $(basename "$2") has $(file_size "$2") bytes after the download, expected $3"
            rm -f "$2"
            exit 1
        fi
    fi
}

# unpack_archive <archive> <folder>: unpack and remove the archive. An archive that cannot be unpacked
# (e.g. truncated by an interrupted download) is removed too, so that the next run downloads it again.
function unpack_archive() {
    local rc=0
    case "$1" in
        *.zip) unzip -o -q "$1" -d "$2" || rc=1 ;;
        *) tar -C "$2" -xf "$1" || rc=1 ;;
    esac
    rm -f "$1"
    [ $rc -eq 0 ] || { print_red "ERROR: could not unpack $(basename "$1") (removed: run this script again to download it)"; exit 1; }
}

# component <name> <function>: install one component in a subshell, so that a failure (a download
# that breaks, a build error) is reported and the other components of the extra are still installed
# Exit code 3 of a component (or of the checks) means that a download failed: the component is
# "untried" (run the script again later), not broken.
COMPONENT_FAILURES=""
COMPONENT_UNTRIED=""
function component() {
    local name="$1" rc
    shift
    ( "$@" )
    rc=$?
    if [[ $rc -eq 3 ]]; then
        print_yellow "$name was not installed: a download failed; continuing with the other components"
        COMPONENT_UNTRIED="$COMPONENT_UNTRIED${COMPONENT_UNTRIED:+, }$name"
    elif [[ $rc -ne 0 ]]; then
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
    [ -f thirdparty/d2net/models/d2_ots.pth ] || from_mirror thirdparty/d2net/models || true
    if [ ! -f thirdparty/d2net/models/d2_ots.pth ]; then
        gdrive_download_file "12Uk95TjBT7VZSEitvm3B3XNK37Q_uU8T" thirdparty/d2net/models/d2net.tar.xz
        unpack_archive thirdparty/d2net/models/d2net.tar.xz thirdparty/d2net/models
    fi
}

function install_netvlad_model() {
    # NetVLAD weights: pySLAM uses configs/netvlad_extract.ini, i.e. the Mapillary model with 512
    # principal components (92 MB). Without this file Patch-NetVLAD downloads all its seven models
    # (2.9 GB) at the first use, with a downloader that cannot resume.
    download_file "https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/mapillary_WPCA512.pth.tar?download=true" \
        thirdparty/patch_netvlad/patchnetvlad/pretrained_models/mapillary_WPCA512.pth.tar 92488008
}

# The recommended learned features (pixi task models-features): SuperPoint, and SuperPoint with the
# LightGlue matcher. Their weights are downloaded by the check, on first creation.
function install_features_core() {
    init_submodules thirdparty/superpoint thirdparty/LightGlue
    apply_patch LightGlue lightglue.patch
}

# The recommended place recognition (pixi task models-vpr): CosPlace, from torch.hub
function install_vpr_core() {
    init_submodules thirdparty/vpr
    apply_patch vpr vpr.patch
    trust_vpr_hub_repos
}

function install_vpr() {
    init_submodules thirdparty/vpr thirdparty/patch_netvlad
    apply_patch vpr vpr.patch
    apply_patch patch_netvlad patch_netvlad.patch
    component "NetVLAD" install_netvlad_model
    trust_vpr_hub_repos
}

# torch >= 2.13 asks "Do you trust this repository?" on the first torch.hub.load of a repo, which
# a loop-detection child process cannot answer: trust the repos used by the VPR detectors.
function trust_vpr_hub_repos() {
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
    download_file https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt thirdparty/ml_depth_pro/checkpoints/depth_pro.pt 1904446787
}

function install_depth_anything_v2() {
    # Depth Anything V2 (monocular, metric indoor/outdoor models)
    clone_repo depth_anything_v2 https://github.com/DepthAnything/Depth-Anything-V2.git
    apply_patch depth_anything_v2 depth_anything_v2.patch
    # <dataset>:<size>:<encoder>:<bytes>
    local entry dataset size encoder bytes model
    for entry in Hypersim:Small:vits:99222290 Hypersim:Base:vitb:389965138 Hypersim:Large:vitl:1341401882 \
                 VKITTI:Small:vits:99221808 VKITTI:Base:vitb:389964656 VKITTI:Large:vitl:1341401064; do
        IFS=: read -r dataset size encoder bytes <<< "$entry"
        model="depth_anything_v2_metric_$(echo "$dataset" | tr '[:upper:]' '[:lower:]')_$encoder.pth"
        download_file "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-$dataset-$size/resolve/main/$model?download=true" \
            "thirdparty/depth_anything_v2/metric_depth/checkpoints/$model" "$bytes"
    done
}

function install_raft_stereo() {
    # (thirdparty/raft_stereo.patch only fixed the model links, which upstream has fixed since)
    clone_repo raft_stereo https://github.com/princeton-vl/RAFT-Stereo.git 6e93ed2169bd858dbb43033988563f3b0bb49506
    [ -f thirdparty/raft_stereo/models/raftstereo-middlebury.pth ] || from_mirror thirdparty/raft_stereo/models || true
    if [ ! -f thirdparty/raft_stereo/models/raftstereo-middlebury.pth ]; then
        # the link is the one of RAFT-Stereo's download_models.sh
        download_file "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1" \
            thirdparty/raft_stereo/models/models.zip
        unpack_archive thirdparty/raft_stereo/models/models.zip thirdparty/raft_stereo/models
    fi
}

function install_crestereo_pytorch() {
    # CREStereo, PyTorch port (the original needs MegEngine, which is not supported here)
    clone_repo crestereo_pytorch https://github.com/ibaiGorordo/CREStereo-Pytorch.git
    apply_patch crestereo_pytorch crestereo_pytorch.patch
    gdrive_download_file "1pNVdaSvkgCK9NuU1i66nZVNg87VsGxox" thirdparty/crestereo_pytorch/models/crestereo_eth3d.pth 21763979
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
        || { rm -rf "$tmp_dir"; print_red "ERROR: could not fetch $1 from PyPI"; exit 3; }
    rm -rf "$tmp_dir"
}

function install_rf_detr() {
    clone_repo rf_detr https://github.com/roboflow/rf-detr.git fd1295b8ccacba0fad2b4a40c8a35b67bc62c335
    apply_patch rf_detr rf_detr.patch
}

function install_detic() {
    # Detic (with its CenterNet2 submodule)
    clone_repo detic https://github.com/facebookresearch/Detic.git 436cda2a2347df60a7c66daca0e8c59f93dc5e79
    update_submodules detic
    apply_patch detic detic.patch
    apply_patch detic/third_party/CenterNet2 detic/third_party/centernet2.patch  # created by detic.patch
    download_file https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        thirdparty/detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth 702411632
}

function install_eov_seg() {
    clone_repo eov_segmentation https://github.com/nhw649/EOV-Seg.git 6f5e93e9aca6ccae89fe492b24018f8530075fc4
    apply_patch eov_segmentation eov_segmentation.patch
    # its weights (2.3 GB) are only of use with an NVIDIA GPU: the check skips EOV-Seg without one
    if has_cuda; then
        gdrive_download_file "1dVfHpzmCOlV6hLfUpd3nHXz62wdB7RY2" thirdparty/eov_segmentation/checkpoints/convnext-l.pth 2267750855
    else
        print_yellow "EOV-Seg needs an NVIDIA GPU with CUDA: not downloading its model (2.3 GB) on this machine."
    fi
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

# have_nvcc: true if there is a CUDA compiler that the environment's C++ compiler can work with.
# In a conda or pixi environment that is an nvcc of the environment itself (conda-forge's cuda-nvcc):
# a system nvcc would be run with the environment's gcc, which does not search /usr/include and is
# usually too new for the system's CUDA.
function have_nvcc() {
    local nvcc
    nvcc=$(command -v nvcc) || return 1
    if [[ -n "$CONDA_PREFIX" && "$nvcc" != "$CONDA_PREFIX"/* ]]; then
        print_yellow "The CUDA compiler found ($nvcc) is outside the environment and cannot be used with its C++ compiler."
        return 1
    fi
}

# print_nvcc_hint: how to get a CUDA compiler
function print_nvcc_hint() {
    print_yellow "  The pixi level 'full' provides nvcc. In a conda environment, install the compiler matching torch's CUDA version, e.g.:"
    print_yellow "  conda install -c conda-forge --override-channels cuda-nvcc cuda-cudart-dev cuda-libraries-dev \"cuda-version=$("$PYTHON_EXE" -c 'import torch; print(torch.version.cuda)' 2>/dev/null)\""
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
    if ! have_nvcc; then
        print_yellow "No CUDA compiler (nvcc): not building curope in thirdparty/$1. The model still works, with a slower PyTorch implementation of RoPE."
        print_nvcc_hint
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
    clone_repo mast3r https://github.com/naver/mast3r f5209afc300cec36239a7ac992263f36847bbba0
    update_submodules mast3r
    apply_patch mast3r mast3r.patch
    apply_patch mast3r/dust3r mast3r-dust3r.patch
    apply_patch mast3r/dust3r/croco mast3r-dust3r-croco.patch
    build_curope mast3r/dust3r/croco/models/curope
    download_file https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        thirdparty/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth 2754910614
}

function install_mvdust3r() {
    clone_repo mvdust3r https://github.com/facebookresearch/mvdust3r.git 430ca6630b07567cfb2447a4dcee9747b132d5c7
    apply_patch mvdust3r mvdust3r.patch
    build_curope mvdust3r/croco/models/curope
    # pySLAM uses the MV-DUSt3R model MVD.pth (6.6 GB). The other checkpoints of MV-DUSt3R (MV-DUSt3R+
    # stages 1 and 2 and DUSt3R 224, 18 GB more) are not used: for those, see
    # thirdparty/mvdust3r_scripts/download_models.py.
    download_file https://huggingface.co/Zhenggang/MV-DUSt3R/resolve/main/checkpoints/MVD.pth thirdparty/mvdust3r/checkpoints/MVD.pth 6577062741
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
    if ! have_nvcc; then
        print_yellow "No CUDA compiler (nvcc): skipping the Gaussian splatting extensions (simple-knn, diff-gaussian-rasterization, lietorch)."
        print_nvcc_hint
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

caches_from_mirror
FAILED=""
UNTRIED=""
for extra in "$@"; do
    print_blue '================================================'
    print_blue "Installing extra '$extra': $(extra_description "$extra")"
    print_blue '================================================'
    install_${extra//-/_}
    print_blue "Checking '$extra' and downloading its model weights (first run can take a while) ..."
    "$PYTHON_EXE" "$SCRIPTS_DIR/extras_check.py" "$extra"
    case $? in
        0) ;;
        3) UNTRIED="$UNTRIED $extra" ;;
        *) FAILED="$FAILED $extra" ;;
    esac
done

cd "$STARTING_DIR"
if [[ -n "$COMPONENT_FAILURES" ]]; then
    print_red "Not installed: $COMPONENT_FAILURES (see the errors above)."
fi
if [[ -n "$FAILED" ]]; then
    print_red "Some components failed the check in:$FAILED (see above)"
fi
if [[ -n "$COMPONENT_UNTRIED$UNTRIED" ]]; then
    [[ -n "$COMPONENT_UNTRIED" ]] && print_yellow "Not installed because a download failed: $COMPONENT_UNTRIED."
    [[ -n "$UNTRIED" ]] && print_yellow "Some components could not be checked because a download failed, in:$UNTRIED (see above)."
    print_yellow "This is a network or server problem, not a broken installation: run the same command again"
    print_yellow "later. It skips what is already installed and resumes interrupted downloads."
fi
if [[ -n "$COMPONENT_FAILURES$FAILED" ]]; then
    exit 1
fi
if [[ -n "$COMPONENT_UNTRIED$UNTRIED" ]]; then
    exit 3
fi
print_green "Installed: $*"
