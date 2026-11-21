# Publishing `gtvdb` to PyPI

The driver is pure Python, so a release is: bump one string, run one gate, upload
two files. There is no wheel matrix and no platform build.

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
      | PyPI project name | `gtvdb` |
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
      needs an account-scoped token; afterwards re-scope it to the `gtvdb`
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
- [ ] Confirm <https://pypi.org/project/gtvdb/> renders the README, the licence,
      and the Python 3.9–3.13 classifiers.

## What the repository still has to change before `0.1.0` means something

The packaging is done; these are the gaps a first user will hit within ten
minutes, and none of them are fixed by a metadata field:

1. **There is no engine on PyPI.** `pip install gtvdb` installs a client. Its
   README says to start `gtv-server` or `cargo run -p gtv-engine --example
   serve_demo`, which requires a Rust toolchain and a git clone. A "Getting
   started" that ends in `cargo build` is a real drop-off; shipping a container
   image (or publishing the engine binary) is the missing half of the story.
2. **`gtvdb` is still an unclaimed name.** It was free at the time of writing,
   but names are first-come. `check_release.py` re-checks it on every run.

(`gtvdb_ext`'s `gtv-server` used to answer protocol v0 only, which made a v1
driver pointless against the production server. Its core pin now carries v1 —
`ExecuteQuery`, `Negotiate`, and `Execute` for compatibility — so that gap is
closed; what remains is the engine not being distributable, above.)

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
