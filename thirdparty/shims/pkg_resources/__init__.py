"""
Lightweight compatibility shim for third‑party code that does:

    from pkg_resources import packaging

We don't need the full `pkg_resources` API here; older releases of OpenAI's
CLIP package only use `packaging.version.parse`.  Importing and re‑exporting the
`packaging` module is therefore sufficient and avoids depending on how
`setuptools` bundles `pkg_resources` on a given platform/Python version.
"""

import packaging as packaging  # re-export for consumers expecting `pkg_resources.packaging`
import packaging.version  # noqa: F401  (a submodule: `import packaging` alone does not load it)

