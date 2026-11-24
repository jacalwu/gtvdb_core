# Publishing `pygtvdb-local` to PyPI

This distribution ships a **compiled engine**, so unlike `pygtvdb` it cannot be
one universal file: PyPI holds one wheel per platform, and pip picks the right
one. A release is therefore three builds and one upload, not one build and one
upload.

## First release, in order

### 1. Accounts (once)

- [ ] Register at <https://pypi.org> **and** <https://test.pypi.org> — separate
      account databases, separate tokens.
- [ ] Enable two-factor authentication. Uploading without 2FA is refused, and
      the error ("invalid or non-existent authentication information") does not
      mention 2FA at all.
- [ ] Verify the email address; an unverified account cannot upload.

### 2. Trusted publishing (once, recommended)

No token is stored anywhere: GitHub mints a short-lived credential per run.

- [ ] PyPI → *Your account* → *Publishing* → *Add a pending publisher*:

      | field | value |
      |:--|:--|
      | PyPI project name | `pygtvdb-local` |
      | Owner | `jacalwu` |
      | Repository | `gtvdb_core` |
      | Workflow name | `publish-python-local.yml` |
      | Environment | `pypi` |

- [ ] Repeat on TestPyPI with environment `testpypi`.
- [ ] GitHub → *Settings* → *Environments* → create `pypi` (and `testpypi`).
      Add yourself as a required reviewer on `pypi`: a mis-tagged commit then
      needs a human click before it can burn a version number.

A *pending* publisher works for the very first upload, so no token is needed
even for `0.1.0`. Note the distribution name is `pygtvdb-local` while the
import name is `pygtvdb_local` — the publisher field wants the distribution.

### 3. Rehearse on TestPyPI

- [ ] GitHub → *Actions* → *Publish Python local driver* → *Run workflow*,
      `target: testpypi`.
- [ ] Install from TestPyPI on each OS you claim to support. TestPyPI has no
      `pyarrow`, hence the extra index:

      ```bash
      pip install --index-url https://test.pypi.org/simple/ \
          --extra-index-url https://pypi.org/simple pygtvdb-local
      ```

- [ ] `python -c "import pygtvdb_local as g; print(g.connect().cursor().execute('SELECT 1 AS v').fetchall())"`

### 4. Ship

- [ ] `git tag pyl-v0.1.0 && git push origin pyl-v0.1.0` — the tag must equal
      `pygtvdb_local/_version.py` or the workflow stops at the version check.
- [ ] Confirm <https://pypi.org/project/pygtvdb-local/#files> lists a wheel for
      every platform you built.

## Every release

```bash
# 1. Bump the version — the only place it is written.
$EDITOR python_local/pygtvdb_local/_version.py     # __version__ = "0.2.0"

# 2. Build for this machine and prove the wheel works.
python python_local/build_native.py --wheel
python -m unittest discover -s python_local/tests -t python_local/tests
python python_local/check_wheel.py

# 3a. Tag route: CI builds all three platforms and publishes via OIDC.
git commit -am "release pygtvdb-local 0.2.0"
git tag pyl-v0.2.0 && git push origin pyl-v0.2.0

# 3b. Local route: upload the one platform wheel you just built.
python -m twine upload --username __token__ python_local/dist/*.whl
```

`check_wheel.py` deliberately re-installs the wheel into a temp directory and
queries through it from outside the source tree. Importing from the repository
root finds `python_local/pygtvdb_local/` and would pass even if the wheel were
missing its `.pyd` — the exact failure this package is prone to.

## The platform matrix

| runner | wheel tag | covers |
|:--|:--|:--|
| `ubuntu-latest` | `manylinux_2_34_x86_64` | glibc ≥ 2.34 (Ubuntu 22.04+) |
| `windows-latest` | `win_amd64` | Windows 10+ x64 |
| `macos-latest` | `macosx_*_arm64` | Apple Silicon |

Every wheel is `cp39-abi3`: one artefact per platform covers Python 3.9–3.13.
`build_native.py` is the single entry point that produces them, and it is what
CI runs — so a green pipeline proves the documented command works.

Intel Macs need `macos-13` added to the matrix; the tag comes from the runner,
so both architectures can coexist in one release. Linux aarch64 would need an
`ubuntu-24.04-arm` runner and a `manylinux_*_aarch64` tag.

## Rules PyPI will not let you break

- **A version number is burned forever.** A botched `0.1.0` means `0.1.1`; the
  filename is immutable and a release can be yanked, never deleted.
- **Metadata is frozen at upload.** A typo in `README.md` needs a new version.
- **Never publish a partial matrix.** The `publish` job refuses unless all
  three wheels are present: pip silently resolves an *older* version on a
  platform with no matching wheel, which looks like a working install.
- **`manylinux_2_34` is a floor, not a description.** It claims the wheel runs
  on any glibc ≥ 2.34. Building on a newer image and tagging it `manylinux_2_28`
  would produce wheels that crash on load.
