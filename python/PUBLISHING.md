# Publishing `gtvdb` to PyPI

The driver is pure Python, so a release is: bump one string, run one gate, upload
two files. There is no wheel matrix and no platform build.

## One-time setup

1. **Claim the name.** Register on <https://pypi.org> and add a verified email.
   `gtvdb` was unclaimed when this document was written; `check_release.py`
   re-checks it on every run.
2. **Pick an auth method.**
   - *Trusted publishing (preferred, no secret in the repo or on disk).* On PyPI →
     *Your projects* → *Publishing* → add a pending publisher with:
     workflow `publish-python.yml`, environment `pypi`. Then do the same for
     TestPyPI with environment `testpypi`. Only works once the project exists, so
     the very first release has to be uploaded with a token — or create a
     *pending* publisher before the first upload, which is the cleaner path.
   - *API token.* PyPI → *Account settings* → *API tokens* → scope it to the
     `gtvdb` project once it exists. Keep it in `~/.pypirc` or an environment
     variable; never in this repository.

## Every release

```bash
# 1. Bump the version — the only place it is written.
$EDITOR python/gtvdb/_version.py          # __version__ = "0.1.0"

# 2. Run the gate. Builds in a temp dir, runs the suite, checks metadata,
#    and refuses if PyPI already has this version.
python python/check_release.py

# 3. Dry run on TestPyPI, install from it, and only then go to PyPI.
python python/check_release.py --keep-dist
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<testpypi-token> \
  python -m twine upload --repository testpypi python/dist/*

python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple gtvdb
python -c "import gtvdb; print(gtvdb.__version__)"

# 4. The real thing.
python -m twine upload python/dist/*
```

`check_release.py` deliberately cannot upload: a check that runs on a laptop
should not be able to push a release. Uploading stays an explicit command.

### Or: tag-driven, with no token anywhere

```bash
git tag py-v0.1.0 && git push origin py-v0.1.0
```

`.github/workflows/publish-python.yml` then builds, verifies the tag matches
`gtvdb/_version.py`, publishes to PyPI via OIDC, and attaches the artefacts to a
GitHub release. Use *Run workflow* with `target: testpypi` to rehearse.

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
