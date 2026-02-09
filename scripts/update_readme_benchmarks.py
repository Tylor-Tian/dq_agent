#!/usr/bin/env python3
"""Update README benchmark section from committed benchmark artifacts.

This script keeps the README benchmark section in sync with the latest
committed results in `benchmarks/`.

It replaces the content between:
  <!-- BENCHMARKS:START -->
  <!-- BENCHMARKS:END -->

with a generated markdown block.
"""

from __future__ import annotations

from pathlib import Path


START = "<!-- BENCHMARKS:START -->"
END = "<!-- BENCHMARKS:END -->"


def extract_first_table(md: str, header_prefix: str) -> str:
    """Extract the first markdown table starting with a given header line prefix."""
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header_prefix):
            start = i
            break
    if start is None:
        return ""
    out: list[str] = []
    for j in range(start, len(lines)):
        if not lines[j].strip().startswith("|"):
            break
        out.append(lines[j])
    return "\n".join(out).strip()


def find_line(md: str, prefix: str) -> str:
    """Find the first line that starts with prefix (after stripping)."""
    for line in md.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


def build_block(repo_root: Path) -> str:
    parts: list[str] = []

    # Raha summary
    raha_compare = repo_root / "benchmarks" / "raha_compare.md"
    if raha_compare.exists():
        table = extract_first_table(raha_compare.read_text(encoding="utf-8"), "| profile |")
        parts.append(
            "### Raha (7 datasets; dirty vs clean profiles)\n"
            "\n"
            + (table or "(No summary table found in `benchmarks/raha_compare.md`.)")
            + "\n\n"
            "Full breakdown: `benchmarks/raha_compare.md`."
        )
    else:
        parts.append(
            "### Raha (dirty vs clean profiles)\n\n"
            "Missing `benchmarks/raha_compare.md`. Reproduce with:\n\n"
            "```bash\n"
            "bash scripts/run_raha_and_save.sh\n"
            "```"
        )

    # Raha string noise union comparison
    raha_noise_compare = repo_root / "benchmarks" / "raha_noise_union" / "compare.md"
    if raha_noise_compare.exists():
        noise_md = raha_noise_compare.read_text(encoding="utf-8")
        table = extract_first_table(noise_md, "| metric |")

        tax_row = find_line(noise_md, "| raha/tax |")
        tax_block = ""
        if tax_row:
            tax_block = (
                "\n\nLargest per-dataset gain (from the committed compare file):\n\n"
                "| dataset | base cell_f1 | union cell_f1 | Δ | base row_f1 | union row_f1 | Δ |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n"
                f"{tax_row}"
            )
        parts.append(
            "### Raha string-noise ablation (patterns: `*`, `''`)\n"
            "\n"
            + (table or "(No summary table found in `benchmarks/raha_noise_union/compare.md`.)")
            + tax_block
            + "\n\n"
            "Full breakdown: `benchmarks/raha_noise_union/compare.md`."
        )
    else:
        parts.append(
            "### Raha string-noise ablation (patterns: `*`, `''`)\n\n"
            "Missing `benchmarks/raha_noise_union/compare.md`. Reproduce with:\n\n"
            "```bash\n"
            "bash scripts/bench_raha_noise_union.sh\n"
            "```"
        )

    # PED summary (optional)
    ped_compare = repo_root / "benchmarks" / "ped_compare.md"
    if ped_compare.exists():
        table = extract_first_table(ped_compare.read_text(encoding="utf-8"), "| profile |")
        parts.append(
            "### PED (additional dirty/clean datasets)\n"
            "\n"
            + (table or "(No summary table found in `benchmarks/ped_compare.md`.)")
            + "\n\n"
            "Full breakdown: `benchmarks/ped_compare.md`."
        )
    else:
        parts.append(
            "### PED (additional dirty/clean datasets)\n\n"
            "Not generated yet. Run:\n\n"
            "```bash\n"
            "bash scripts/run_ped_and_save.sh\n"
            "```"
        )

    return "\n\n".join(parts).strip() + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    readme = repo_root / "README.md"

    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(
            "README markers not found. Add the following markers to README.md:\n"
            f"  {START}\n  {END}\n"
        )

    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    block = build_block(repo_root)

    new_text = before + START + "\n" + block + END + after
    readme.write_text(new_text, encoding="utf-8")
    print(f"updated: {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
