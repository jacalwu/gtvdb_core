# Publishing `gtvdb` to PyPI

The driver carries the in-process engine, so it is no longer pure Python: a
release is **three platform builds and one upload**. There is no universal
artefact — each wheel contains a compiled engine for one OS and architecture.

## First release, in order

Nothing here is code, and none of it can be done by a script — it is account
setup on two websites plus one tag push. Do it once; afterwards a release is
the three-step loop below.

### 1. Accounts (once)

- [ ] Register at <https://pypi.org> **and** <https://test.pypi.org>. They are
      separate account databases; a token from one does not work on the other.
- [ ] Enable two-factor auth on PyPI. Uploading without 2FA is refused, and the
      error message ("invalid or non-existent authentication information") does
      not say so.
- [ ] Verify the email address. An unverified account cannot upload.

### 2. Trusted publishing (once, recommended)

No token is stored anywhere; GitHub mints a short-lived credential per run.

- [ ] PyPI → *Your account* → *Publishing* → *Add a pending publisher*:
      | field | value |
      |:--|:--|
      | PyPI project name | `pygtvdb` |
      | Owner | `jacalwu` |
      | Repository | `gtvdb_core` |
      | Workflow name | `publish-python.yml` |
      | Environment | `pypi` |
- [ ] Repeat on TestPyPI with environment `testpypi`.
- [ ] GitHub → *Settings* → *Environments* → create `pypi` (and `testpypi`).
      Add yourself as a required reviewer on `pypi`: a mis-tagged commit then
      needs a human click before it can burn a version number.

A *pending* publisher works for the very first upload — the project does not
have to exist yet. That is why no API token is needed even for `0.1.0`. Use a
token only if you would rather not put OIDC in the path:

- [ ] PyPI → *Account settings* → *API tokens* → *Add token*. The first release
      needs an account-scoped token; afterwards re-scope it to the `pygtvdb`
      project.

### 3. Rehearse on TestPyPI (once per new packaging change)

- [ ] GitHub → *Actions* → *Publish Python driver* → *Run workflow*,
      `target: testpypi`.
- [ ] `pip install --index-url https://test.pypi.org/simple/
      --extra-index-url https://pypi.org/simple gtvdb` — the extra index is
      required because `grpcio` / `pyarrow` are not on TestPyPI.
- [ ] `python -c "import gtvdb; print(gtvdb.__version__, gtvdb.paramstyle)"`.

### 4. Ship

- [ ] `git tag py-v0.1.0 && git push origin py-v0.1.0` — the tag must equal
      `gtvdb/_version.py` or the workflow stops at the version check.
- [ ] Confirm <https://pypi.org/project/pygtvdb/> renders the README, the licence,
      and the Python 3.9–3.13 classifiers.

## What the repository still has to change before `0.1.0` means something

The packaging is done. Two things a first user used to hit within ten minutes:

1. **Engine distribution — now solved.** `pip install pygtvdb` is still a client
   only (it cannot embed the engine), but the server half no longer requires a
   Rust toolchain and a git clone. The `v*` tag workflow (`.github/workflows/
   publish-engine.yml`) publishes Linux binaries to the GitHub release and a
   container image to GHCR; the root `Dockerfile` builds the same image locally:

       docker run --rm -p 50051:50051 -e GTV_ADDR=0.0.0.0:50051 \
         ghcr.io/jacalwu/gtvdb-core:0.1.0 gtv-demo-server
       # then: python -c "import gtvdb; ...connect('gtv://127.0.0.1:50051')"

   The image's `gtv-demo-server` serves one immutable snapshot for protocol
   development; the production store-backed server remains in `gtvdb_ext`.
   Practise the image tag against TestPyPI's client before announcing either.
2. **`gtvdb` is still an unclaimed name.** It was free at the time of writing,
   but names are first-come. `check_release.py` re-checks it on every run.

(`gtvdb_ext`'s `gtv-server` used to answer protocol v0 only, which made a v1
driver pointless against the production server. Its core pin now carries v1 —
`ExecuteQuery`, `Negotiate`, and `Execute` for compatibility — so that gap is
closed; what remains is the engine not being distributable, above.)

## The platform matrix

| runner | wheel tag | covers |
|:--|:--|:--|
| `ubuntu-latest` | `manylinux_2_34_x86_64` | glibc ≥ 2.34 (Ubuntu 22.04+) |
| `windows-latest` | `win_amd64` | Windows 10+ x64 |
| `macos-latest` | `macosx_*_arm64` | Apple Silicon |

Every wheel is `cp39-abi3`, so one artefact per platform covers Python 3.9–3.13.
`build_native.py` produces them and CI runs that same script, which is why a
green pipeline proves the documented local command works.

Intel Macs need `macos-13` added to the matrix; the tag comes from the runner, so
both architectures can coexist in one release. Linux aarch64 would need an
`ubuntu-24.04-arm` runner and a `manylinux_*_aarch64` tag.

**No sdist is uploaded.** An sdist carries no compiled engine, so pip on a
machine without a Rust toolchain would try to build one and fail. Per-platform
wheels mean every install is complete. `check_release.py` still builds an sdist
to prove it is self-sufficient, but the publish job uploads `*.whl` only.

A platform missing from a release is worse than a failed release: pip there
silently resolves an **older, remote-only** version instead of reporting that
the new one has no wheel for it.

## Every release

Two equivalent routes. The tag route is the one that ends with a CI-built
artefact and a GitHub release; the local route is what you use when the
upload itself is the thing being debugged.

```bash
# 1. Bump the version — the only place it is written.
$EDITOR python/gtvdb/_version.py          # __version__ = "0.2.0"

# 2. Run the gate. Builds in a temp dir, runs the suite, checks metadata,
#    and refuses if PyPI already has this version.
python python/check_release.py

# 3a. Tag route: CI builds, verifies the tag against _version.py, publishes
#     via OIDC, and attaches the artefacts to a GitHub release.
git commit -am "release python driver 0.2.0"
git tag py-v0.2.0 && git push origin py-v0.2.0

# 3b. Local route: build, rehearse on TestPyPI, then upload.
python python/check_release.py --keep-dist
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<testpypi-token> \
  python -m twine upload --repository testpypi python/dist/*

python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple gtvdb
python -c "import gtvdb; print(gtvdb.__version__)"

python -m twine upload python/dist/*     # the real thing
```

`check_release.py` deliberately cannot upload: a check that runs on a laptop
should not be able to push a release. Uploading stays an explicit command.

## Rules PyPI will not let you break

- **A version number is burned forever.** You can `yank` a release but not
  delete it, and the filename is immutable — so a botched `0.1.0` means `0.1.1`.
  That is why the suite and `twine check` run before the upload, not after.
- **Metadata is frozen at upload.** Fixing a typo in `README.md` needs a new
  version; the project page for an old release will keep showing the old text.
- **`python/LICENSE.md` must match `LICENSE.md`.** PEP 639 forbids `..` in
  `license-files`, so the build root needs its own copy. `check_release.py`
  fails if the two ever diverge.

## Versioning policy

`0.x` while the wire protocol is v1 and `docs/api-compat.md` still lists breaking
changes. The driver version tracks the *client* only — it does not have to match
the engine's crate versions, because protocol compatibility is negotiated at
connect time (`Negotiate`), not inferred from a version string.
