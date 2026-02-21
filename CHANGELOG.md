# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- PyPI Trusted Publishing workflow via GitHub Actions OIDC (`.github/workflows/publish-pypi.yml`).
- Open source release assets: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and publishing guide.
- README updates for positioning, installation, CI integration, and release links.
- Demo architecture image at `docs/assets/demo.svg`.

### Changed
- Refreshed `examples/report.json` and `examples/report.md` to match current schema and CLI behavior.
- Improved package metadata in `pyproject.toml` for PyPI release readiness.

## [v0.1.0] - 2026-02-21

### Added
- CLI commands for runnable data-quality workflows, including `dq demo` and `dq run`.
- Typed error contract for deterministic, machine-readable failure handling.
- Stable artifact schemas with `schema_version: 1`.
- Dual report outputs: `report.json` (machine) and `report.md` (human).
- Structured trace output (`trace.jsonl`) for run-stage observability.
- Replay and resume support using `run_record.json` and `checkpoint.json`.
- Shadow execution mode for baseline vs candidate comparison and regression gating.
- Benchmark harness and committed benchmark outputs for labeled dirty/clean datasets.
