This folder contains lightweight **Python import shims** that make a few
third-party dependencies happier on macOS (and other environments where
their expected top-level modules are missing or packaged differently).

- `pkg_resources/`:
  - Provides just enough of `pkg_resources` for code that does:
    - `from pkg_resources import packaging`
  - Internally it simply re-exports the standard `packaging` module.
  - Used to unblock:
    - Older releases of OpenAI's CLIP package (e.g. `openai-clip` 1.0.1 on
      PyPI and conda-forge), which call `pkg_resources.packaging.version`.
      Detic installs OpenAI's CLIP from GitHub, which no longer needs it.

This directory is **not** imported directly from the repo root. Instead,
it is copied into the active Python environment's `site-packages` by the
`scripts/install_mac_shims.sh` helper when:

- `import pkg_resources` fails.

On Linux or properly configured environments, you typically don't need this
shim at all; the script simply detects that the real module is available
and does nothing.
