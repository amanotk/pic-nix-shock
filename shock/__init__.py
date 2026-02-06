#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python Data Analysis Tool for Collisionless Shock Simulations."""

from . import utils as _utils


def _reexport(module, names):
    for name in names:
        globals()[name] = getattr(module, name)


_utils_public = [name for name in dir(_utils) if not name.startswith("_")]
_reexport(_utils, _utils_public)

# Keep package-level API compatible with older imports when optional
# runtime dependencies for summary helpers are available.
try:
    from . import summary as _summary
except ModuleNotFoundError:
    _summary_public = []
else:
    _summary_public = [name for name in dir(_summary) if not name.startswith("_")]
    _reexport(_summary, _summary_public)

__all__ = ["__version__", *_utils_public, *_summary_public]
__version__ = "0.1.0"
