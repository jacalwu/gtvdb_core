"""The one place the driver's version is written.

`pyproject.toml` reads it back with `dynamic = ["version"]` / `attr = ...`, and
setuptools resolves that attribute by parsing this file rather than importing
it, so the build does not need `grpcio` installed. Keep this module free of
imports and of anything but a plain string assignment.
"""

__version__ = "0.1.0"
