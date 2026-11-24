"""Wheel packaging shim.

The compiled extension is copied in by `cargo build -p gtvpy-local --release`
as `pygtvdb_local/_gtv_native.abi3.so`. We must mark the wheel as **non-pure**,
otherwise setuptools stamps `py3-none-any` and macOS/Windows users would pip
install a Linux binary. Build with:

    python -m build --wheel --no-isolation \
        --config-setting=--global-option=bdist_wheel \
        --config-setting=--plat-name=manylinux_2_34_x86_64

or simply `python setup.py bdist_wheel --plat-name manylinux_2_34_x86_64`.
"""

from setuptools import Distribution, setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Tell setuptools this distribution ships compiled code.

    Without it, `has_ext_modules()` is False (the package declares no
    `ext_modules`, because the `.pyd`/`.so` is prebuilt by cargo), and the wheel
    is laid out as pure Python — the package lands under
    `*.data/purelib/` instead of the wheel root. It still installs, but any
    tool that reads the wheel by path (`check_contents`, audit tools, some
    indexes) cannot find the extension.
    """

    def has_ext_modules(self) -> bool:  # noqa: D102 - see class docstring
        return True


class platform_bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # This distribution embeds a native extension: it is NOT pure Python.
        self.root_is_pure = False

    def get_tag(self):
        """Stamp the abi3 tag the extension was actually built for.

        `crates/gtvpy-local` enables pyo3's `abi3-py39`, so one wheel is
        supposed to cover CPython 3.9+. Left to setuptools, the tag becomes the
        *building* interpreter's exact version (`cp313-cp313`), and pip then
        refuses the wheel on 3.12 and below — silently defeating abi3.
        """
        _, _, plat = super().get_tag()
        return "cp39", "abi3", plat


setup(cmdclass={"bdist_wheel": platform_bdist_wheel}, distclass=BinaryDistribution)
