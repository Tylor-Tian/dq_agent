# Publishing to PyPI (Trusted Publishing)

This repository publishes with **PyPI Trusted Publishing (OIDC)** through GitHub Actions.
No PyPI API token, username, or password is used.

## Build locally

```bash
python -m pip install -U pip
python -m pip install build twine
python -m build
twine check dist/*
```

Expected outputs:

- `dist/*.whl`
- `dist/*.tar.gz`

## Configure Trusted Publisher (one-time)

In PyPI project settings, add a Trusted Publisher pointing to this GitHub repository.
Use the workflow file name:

- `publish-pypi.yml`

## Release-driven publish

The workflow `.github/workflows/publish-pypi.yml` triggers on:

- GitHub Release `published`
- Manual `workflow_dispatch`

After a release is published on GitHub, the workflow builds and checks distributions,
then publishes using `pypa/gh-action-pypi-publish@release/v1` via OIDC.

## Optional: TestPyPI dry run

Before publishing to PyPI, you can test the same flow against TestPyPI by creating a
separate Trusted Publisher configuration there and running a release in a staging repo/tag.
