#!/usr/bin/env python3
# This file is part of https://github.com/luigifreda/pyslam
"""
The counterpart of scripts/install_extra.sh on Windows (native), where there is no bash: install
optional pySLAM components ("extras") on top of the core installation. Only the recommended learned
models are ported: SuperPoint, SuperPoint with the LightGlue matcher, and CosPlace.

Each extra fetches the git submodules it needs, applies pySLAM's patches and then checks every
component on the bundled test images (scripts/extras_check.py), which downloads the model weights.
It can be re-run safely: what is already done is skipped.

usage: python scripts/install_extra_windows.py --list
       python scripts/install_extra_windows.py <extra> [<extra> ...]

Exit code: 0 if everything is installed, 1 if something failed, 3 if only downloads failed (run the
same command again later).
"""
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# name: (description, submodules, (thirdparty folder, patch file in thirdparty/) ..., torch.hub repos to trust)
EXTRAS = {
    "features-core": (
        "The recommended learned features: SuperPoint, and SuperPoint with the LightGlue matcher",
        ["thirdparty/superpoint", "thirdparty/LightGlue"],
        [("LightGlue", "lightglue.patch")],
        [],
    ),
    "vpr-core": (
        "The recommended visual place recognition loop detector: CosPlace",
        ["thirdparty/vpr"],
        [("vpr", "vpr.patch")],
        # torch >= 2.13 asks "Do you trust this repository?" on the first torch.hub.load of a repo,
        # which a loop-detection child process cannot answer
        ["gmberton_cosplace", "gmberton_eigenplaces", "gmberton_MegaLoc"],
    ),
}
NOT_PORTED = ["features", "vpr", "depth", "semantics", "scene3d"]


def git(*args, cwd=ROOT_DIR, quiet=False):
    """Run git; True if it succeeded."""
    out = subprocess.DEVNULL if quiet else None
    return subprocess.run(["git", *args], cwd=cwd, stdout=out, stderr=out).returncode == 0


def init_submodules(paths):
    """Fetch only the given submodules, each at its recorded commit and without its history; with
    the whole history if the server does not allow that. A fetch that fails is tried again after a
    pause (5, 15 and 30 s: four attempts in all), since most failures are a connection to GitHub
    that drops for a moment; PYSLAM_SUBMODULE_RETRY_PAUSES="<s> <s> ..." sets other pauses ("" for a
    single attempt). The same as init_submodules() of install_extra.sh."""
    print(f"Fetching submodules: {' '.join(paths)}", flush=True)
    pauses = [float(p) for p in os.environ.get("PYSLAM_SUBMODULE_RETRY_PAUSES", "5 15 30").split()]
    num_attempts = len(pauses) + 1
    update = ["git", "submodule", "update", "--init", "--recursive"]
    for attempt in range(1, num_attempts + 1):
        # git's output is captured: on a failure git repeats the same error for its own retry and
        # again for the fallback, about eight times per attempt, which buries the messages below
        output = ""
        for command in (update + ["--depth", "1", "--", *paths], update + ["--", *paths]):
            result = subprocess.run(command, cwd=ROOT_DIR, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace")
            output += result.stdout or ""
            if result.returncode == 0:
                print(output, end="", flush=True)
                return
        lines = output.splitlines()
        errors = sorted({line for line in lines if line.startswith(("fatal:", "error:"))})
        for line in errors or lines[-5:]:
            print(f"    git: {line}", flush=True)
        if attempt < num_attempts:
            pause = pauses[attempt - 1]
            print(f"Fetching the submodules failed (attempt {attempt} of {num_attempts}; git's message "
                  f"is above): trying again in {pause:g} s ...", flush=True)
            time.sleep(pause)
    print(f"ERROR: could not fetch submodules after {num_attempts} attempt(s): {' '.join(paths)}")
    print("Check the network connection and run the same command again: what is already installed is kept.")
    sys.exit(3)


def apply_patch(folder, patch_name):
    """Apply thirdparty/<patch_name> to thirdparty/<folder> unless it is already applied."""
    cwd = os.path.join(ROOT_DIR, "thirdparty", folder)
    patch = os.path.join(ROOT_DIR, "thirdparty", patch_name)
    # --ignore-whitespace: a clone made with core.autocrlf (the default of Git for Windows) has
    # other line ends than the patch was made with
    apply = ["apply", "--ignore-whitespace", "--whitespace=nowarn"]
    if git(*apply, "--reverse", "--check", patch, cwd=cwd, quiet=True):
        print(f"patch {patch_name} already applied")
        return
    if not git(*apply, "--check", patch, cwd=cwd, quiet=True):
        if not os.environ.get("PYSLAM_RESET_CLONES"):
            print(f"ERROR: {patch_name} does not apply to thirdparty/{folder}: the folder has local "
                  "changes, for example an older version of this patch.\n"
                  "  To discard them and apply the current patch, run this command again with "
                  "PYSLAM_RESET_CLONES=1.")
            sys.exit(1)
        print(f"thirdparty/{folder} has local changes: discarding them to apply {patch_name} "
              "(PYSLAM_RESET_CLONES is set)")
        git("checkout", "-q", "--", ".", cwd=cwd)
    if not git(*apply, patch, cwd=cwd):
        print(f"ERROR: could not apply {patch_name} to thirdparty/{folder}")
        sys.exit(1)
    print(f"patch {patch_name} applied")


def trust_hub_repos(repos):
    import torch

    path = os.path.join(torch.hub.get_dir(), "trusted_list")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trusted = set(open(path).read().split()) if os.path.exists(path) else set()
    new = [r for r in repos if r not in trusted]
    if new:
        with open(path, "a") as f:
            f.write("".join(r + "\n" for r in new))
    print("torch.hub trusted repos:", ", ".join(sorted(trusted | set(new))))


def list_extras():
    print("Available extras on Windows:")
    for name, (description, _, _, _) in EXTRAS.items():
        print(f"  {name:14s} {description}")
    print(f"Not on Windows yet: {', '.join(NOT_PORTED)}")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--list", "-h", "--help"):
        print(f"usage: {sys.argv[0]} --list | <extra> [<extra> ...]")
        list_extras()
        sys.exit(0 if args else 1)
    for extra in args:
        if extra not in EXTRAS:
            reason = "is not on Windows yet" if extra in NOT_PORTED else "is unknown"
            print(f"ERROR: the extra '{extra}' {reason}")
            list_extras()
            sys.exit(1)

    failed, untried = [], []
    for extra in args:
        description, submodules, patches, hub_repos = EXTRAS[extra]
        print("================================================")
        print(f"Installing extra '{extra}': {description}")
        print("================================================", flush=True)
        init_submodules(submodules)
        for folder, patch_name in patches:
            apply_patch(folder, patch_name)
        if hub_repos:
            trust_hub_repos(hub_repos)
        print(f"Checking '{extra}' and downloading its model weights (first run can take a while) ...",
              flush=True)
        check = os.path.join(ROOT_DIR, "scripts", "extras_check.py")
        rc = subprocess.run([sys.executable, check, extra], cwd=ROOT_DIR).returncode
        if rc == 3:
            untried.append(extra)
        elif rc != 0:
            failed.append(extra)

    if failed:
        print(f"Some components failed the check in: {' '.join(failed)} (see above)")
    if untried:
        print(f"Some components could not be checked because a download failed, in: {' '.join(untried)} "
              "(see above).\nThis is a network or server problem, not a broken installation: run the "
              "same command again later.")
    if failed:
        sys.exit(1)
    if untried:
        sys.exit(3)
    print(f"Installed: {' '.join(args)}")


if __name__ == "__main__":
    main()
