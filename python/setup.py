"""Wheel packaging shim for a distribution that embeds a native engine.

`crates/gtvpy-local` is a pyo3 cdylib built with `abi3-py39`; `build_native.py`
copies the artefact into `gtvdb/_local/`. Two things setuptools cannot infer:

* the distribution is **not** pure Python — the extension has no `ext_modules`
  entry (cargo built it), so `has_ext_modules()` returns False and the wheel is
  laid out as pure: the package lands under `*.data/purelib/` and the tag
  becomes `py3-none-any`, which on a non-Windows machine means pip would install
  a Windows binary it cannot load;
* the tag must be `cp39-abi3`, or pip refuses the wheel on 3.12 and below and
  silently resolves an older pure-Python release instead of this one.

Build through `python/build_native.py --wheel`, which puts the artefact in place
before setuptools ever looks at it.
"""

from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Declare that this distribution ships compiled code."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - see module docstring
        return True


class platform_bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # Belt and braces: BinaryDistribution already implies this.
        self.root_is_pure = False

    def get_tag(self):
        """Stamp the abi3 tag the extension was actually built for."""
        _, _, plat = super().get_tag()
        return "cp39", "abi3", plat


setup(cmdclass={"bdist_wheel": platform_bdist_wheel}, distclass=BinaryDistribution)
