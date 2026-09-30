# Publishing modal-spindle

The distribution name is `modal-spindle`; Python imports remain `import spindle`
and the command remains `spindle`. The package supports Python 3.11 and 3.12;
deploying Modal apps requires Python 3.12.

## Before the first release

Two prerequisites still need to be resolved:

1. **Publish the compatible Stitch dependency.** Spindle currently depends on
   `stitch @ git+https://github.com/modal-projects/stitch.git@375a9396a7b05770dc4ed9cc5fe34fc4d5a472d5`.
   [PyPI rejects direct URL dependencies](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#direct-url-dependencies).
   The existing PyPI project named `stitch` is unrelated. Publish Modal's library
   under a distinct name such as `modal-stitch`, keeping its `stitch` import
   package. Use a release containing the API at the pinned commit: Spindle imports
   `stitch.types`, `stitch.publish`, `stitch.pools`, `stitch.service`, and
   `stitch.stores`. A different Stitch revision needs compatibility testing.
   Then replace the Git dependency in `pyproject.toml` and `STITCH_PACKAGE` in
   `src/spindle/providers/modal/image_dependencies.py` with that verified PyPI
   release, and regenerate both the root and Codegolf example lockfiles.
   Do not replace it with the unrelated PyPI package or omit this required dependency.
2. **Choose a license.** Add the approved text as `LICENSE`, set the SPDX
   expression in `[project].license`, and set `license-files = ["LICENSE"]`.
   The current repository does not declare a license; the release check requires
   one before upload.

As checked on 2026-09-29, `modal-spindle` and `modal-stitch` did not have public
PyPI projects. This does not guarantee that either name can be registered later.

Keep the Git installation command in the README until the first release is
available. After publishing, change it to `uv add modal-spindle` and add the pip
equivalent, `python -m pip install modal-spindle`.

## One-time account setup

1. Sign into the intended owning [PyPI account](https://pypi.org/account/login/)
   with a verified email address and two-factor authentication enabled.
2. Create a GitHub Actions environment named `pypi` in
   [modal-projects/spindle](https://github.com/modal-projects/spindle/settings/environments).
   Configure release tag restrictions and any desired required reviewers there.
3. Add a [pending Trusted Publisher](https://pypi.org/manage/account/publishing/):

   | Field | Value |
   | --- | --- |
   | PyPI project name | `modal-spindle` |
   | GitHub owner | `modal-projects` |
   | Repository | `spindle` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

   The workflow filename is the basename, not `.github/workflows/publish.yml`.
   If the project already exists under your account, configure its publisher
   from that project's Publishing settings instead.

[Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
creates the project on the first successful upload. A pending publisher does not
reserve the name. No long-lived PyPI token needs to be stored in GitHub secrets.
Set up `modal-stitch` separately in its own repository before releasing Spindle.

## Validate a release

Use a clean checkout to avoid mixing old artifacts with the current version:

```bash
uv sync --locked --python 3.12 --default-index https://pypi.org/simple
uv pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync pytest -q
uv build --default-index https://pypi.org/simple
uvx --default-index https://pypi.org/simple --from twine twine check --strict dist/*
uv run --no-project --with packaging python scripts/check_distribution.py --for-pypi --tag v0.1.0
```

Until the two prerequisites above are resolved, the final check intentionally
fails and explains what remains. Omit `--for-pypi` to check the current Git-based
development distributions without claiming they are ready for PyPI.

The Package workflow builds a wheel from the source archive, checks metadata and
package contents, and installs each artifact in fresh Python 3.11 and 3.12
environments. It runs `pip check`, imports the public API, and generates and
validates a deployment config outside a source checkout. It does not deploy apps
or allocate GPUs. GPU runtime dependencies are installed in Modal images.

To refresh the public lockfile after changing dependencies:

```bash
uv lock --default-index https://pypi.org/simple
```

## Publish

1. Set the same version in `pyproject.toml` and `src/spindle/__init__.py`, then
   refresh `uv.lock`. The initial version is `0.1.0`.
2. Merge the release changes and verify that Core CPU tests and Package pass.
3. Create and publish a GitHub release with a tag matching `v<version>`, such as
   `v0.1.0`, on the reviewed commit. Publishing the GitHub release starts the
   PyPI upload workflow; creating a draft release does not.
4. The workflow reruns core tests, builds and validates the artifacts, and waits
   for the `pypi` environment's rules before uploading those same artifacts via
   [PyPA's publishing action](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
5. Confirm the release at [PyPI](https://pypi.org/project/modal-spindle/), then
   install it in a fresh Python 3.12 environment and run `spindle --help`.

PyPI release files cannot be overwritten. If the uploaded code needs changing,
use a new version. To rehearse an upload on TestPyPI, configure its own account,
publisher, and environment and point a separate publishing job at
`https://test.pypi.org/legacy/`; a PyPI publisher does not authorize TestPyPI.
