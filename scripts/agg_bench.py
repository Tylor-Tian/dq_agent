#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def safe_f1(tp: int, fp: int, fn: int, empty_policy: str = "perfect") -> Optional[float]:
    tp = int(tp); fp = int(fp); fn = int(fn)
    if tp == 0 and fp == 0 and fn == 0:
        if empty_policy == "skip":
            return None
        if empty_policy == "zero":
            return 0.0
        return 1.0  # perfect
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def mean(xs: List[Optional[float]]) -> Optional[float]:
    ys = [x for x in xs if x is not None]
    if not ys:
        return None
    return sum(ys) / len(ys)


def load_profile(root: Path, empty_policy: str) -> Dict:
    paths = sorted(glob.glob(str(root / "*" / "metrics.json")))
    if not paths:
        raise SystemExit(f"no metrics.json under {root}")
    rows = []
    tot_tp = tot_fp = tot_fn = 0
    tot_rtp = tot_rfp = tot_rfn = 0
    cell_f1s: List[Optional[float]] = []
    row_f1s: List[Optional[float]] = []

    for p in paths:
        d = json.load(open(p, "r", encoding="utf-8"))
        name = (d.get("dataset") or {}).get("name") or p
        c = d.get("cell") or {}
        r = d.get("row") or {}
        tp, fp, fn = int(c.get("tp", 0)), int(c.get("fp", 0)), int(c.get("fn", 0))
        rtp, rfp, rfn = int(r.get("tp", 0)), int(r.get("fp", 0)), int(r.get("fn", 0))

        f1c = safe_f1(tp, fp, fn, empty_policy=empty_policy)
        f1r = safe_f1(rtp, rfp, rfn, empty_policy=empty_policy)

        rows.append((name, tp, fp, fn, f1c, rtp, rfp, rfn, f1r))

        tot_tp += tp; tot_fp += fp; tot_fn += fn
        tot_rtp += rtp; tot_rfp += rfp; tot_rfn += rfn

        cell_f1s.append(f1c)
        row_f1s.append(f1r)

    out = {
        "datasets": len(rows),
        "macro_cell_f1": mean(cell_f1s),
        "macro_row_f1": mean(row_f1s),
        "micro_cell_f1": safe_f1(tot_tp, tot_fp, tot_fn, empty_policy="perfect"),
        "micro_row_f1": safe_f1(tot_rtp, tot_rfp, tot_rfn, empty_policy="perfect"),
        "cell_tp_fp_fn": (tot_tp, tot_fp, tot_fn),
        "row_tp_fp_fn": (tot_rtp, tot_rfp, tot_rfn),
        "rows": rows,
    }
    return out


def render_md(
    clean: Dict,
    dirty: Dict,
    out_md: Path,
    *,
    title: str = "Raha benchmark summary (dirty vs clean profiles)",
    topn: int = 10,
) -> None:
    def fmt(x):
        return "" if x is None else f"{x:.6f}"

    # worst by cell f1
    def worst(profile: Dict) -> List[Tuple]:
        xs = profile["rows"]
        xs2 = sorted(xs, key=lambda t: (t[4] if t[4] is not None else -1e9))
        return xs2[:topn]

    md = []
    md.append(f"# {title}\n")
    md.append("| profile | datasets | macro_cell_f1 | micro_cell_f1 | macro_row_f1 | micro_row_f1 | cell_tp/fp/fn | row_tp/fp/fn |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.append(f"| clean | {clean['datasets']} | {fmt(clean['macro_cell_f1'])} | {fmt(clean['micro_cell_f1'])} | {fmt(clean['macro_row_f1'])} | {fmt(clean['micro_row_f1'])} | {clean['cell_tp_fp_fn']} | {clean['row_tp_fp_fn']} |")
    md.append(f"| dirty | {dirty['datasets']} | {fmt(dirty['macro_cell_f1'])} | {fmt(dirty['micro_cell_f1'])} | {fmt(dirty['macro_row_f1'])} | {fmt(dirty['micro_row_f1'])} | {dirty['cell_tp_fp_fn']} | {dirty['row_tp_fp_fn']} |")

    md.append("\n## Worst datasets by cell_f1\n")
    md.append("### clean\n")
    md.append("| dataset | cell tp/fp/fn | cell_f1 | row tp/fp/fn | row_f1 |")
    md.append("|---|---:|---:|---:|---:|")
    for name,tp,fp,fn,f1c,rtp,rfp,rfn,f1r in worst(clean):
        md.append(f"| {name} | {(tp,fp,fn)} | {fmt(f1c)} | {(rtp,rfp,rfn)} | {fmt(f1r)} |")

    md.append("\n### dirty\n")
    md.append("| dataset | cell tp/fp/fn | cell_f1 | row tp/fp/fn | row_f1 |")
    md.append("|---|---:|---:|---:|---:|")
    for name,tp,fp,fn,f1c,rtp,rfp,rfn,f1r in worst(dirty):
        md.append(f"| {name} | {(tp,fp,fn)} | {fmt(f1c)} | {(rtp,rfp,rfn)} | {fmt(f1r)} |")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="bench out root for clean profile")
    ap.add_argument("--dirty", required=True, help="bench out root for dirty profile")
    ap.add_argument("--out-md", required=True, help="output markdown file")
    ap.add_argument("--out-json", default=None, help="optional output json file")
    ap.add_argument(
        "--title",
        default="Raha benchmark summary (dirty vs clean profiles)",
        help="Markdown title (H1) for the rendered compare file.",
    )
    ap.add_argument("--empty-policy", choices=["perfect","skip","zero"], default="perfect")
    args = ap.parse_args()

    clean = load_profile(Path(args.clean), empty_policy=args.empty_policy)
    dirty = load_profile(Path(args.dirty), empty_policy=args.empty_policy)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    render_md(clean, dirty, out_md, title=str(args.title))

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out = {"clean": {k: v for k, v in clean.items() if k != "rows"},
               "dirty": {k: v for k, v in dirty.items() if k != "rows"}}
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
