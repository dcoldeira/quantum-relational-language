"""pytest configuration for the QRL test suite.

Installs the local ``platform/`` package into ``sys.modules['platform']``
while preserving stdlib fall-through.  Must live in the *tests/* directory
(not the repo root) so it is loaded during collection — after
``pytest_sessionstart`` has already called ``platform.python_version()``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLAT_DIR = os.path.join(_REPO, "platform")

_stdlib_platform = sys.modules.get("platform")

# Only swap if stdlib's platform is a flat module (not already our package)
if _PLAT_DIR not in getattr(_stdlib_platform, "__path__", []):

    class _PlatformPkg(types.ModuleType):
        """Our local platform/ package with stdlib fall-through.

        Any attribute not set by platform/__init__.py (e.g. python_version,
        system, …) is transparently delegated to the stdlib platform module
        so pytest internals keep working after the swap.
        """

        def __getattr__(self, name: str):
            if _stdlib_platform is not None:
                try:
                    return getattr(_stdlib_platform, name)
                except AttributeError:
                    pass
            raise AttributeError(f"module 'platform' has no attribute {name!r}")

    pkg = _PlatformPkg("platform")
    pkg.__path__ = [_PLAT_DIR]  # type: ignore[assignment]
    pkg.__file__ = os.path.join(_PLAT_DIR, "__init__.py")
    pkg.__package__ = "platform"
    spec = importlib.util.spec_from_file_location(
        "platform",
        pkg.__file__,
        submodule_search_locations=[_PLAT_DIR],
    )
    pkg.__spec__ = spec
    sys.modules["platform"] = pkg

    # Make repo root available so relative submodule imports resolve
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    assert spec is not None
    spec.loader.exec_module(pkg)  # type: ignore[union-attr]
