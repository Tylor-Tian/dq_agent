from __future__ import annotations


def to_markdown(res: dict) -> str:
    meta = res["meta"]
    score = res["score"]
    rules = sorted(res["rules"], key=lambda x: x["hits"], reverse=True)
    top = [rule for rule in rules if rule["hits"] > 0][:5]
    advice = []
    for rule in rules:
        if rule["hits"] > 0 and rule["advice"] not in advice:
            advice.append(rule["advice"])

    lines = [
        "# Data Quality Report",
        "",
        f"- Input: {meta['input']}",
        f"- Rows: {meta['rows']}",
        f"- Cols: {meta['cols']}",
        f"- Score: **{score}**/100",
        "",
        "## Top Violations",
    ]
    if top:
        for rule in top:
            detail = f" ({rule['detail']})" if rule["detail"] else ""
            lines.append(f"- {rule['id']}: {rule['hits']} hits{detail}")
    else:
        lines.append("- No rule violations")

    lines.extend(["", "## Samples", "```text"])
    lines.extend(res.get("samples", []))
    lines.extend(["```", "", "## Advice"])
    if advice:
        for item in advice:
            lines.append(f"- {item}")
    else:
        lines.append("- Keep monitoring quality metrics over time.")
    return "\n".join(lines) + "\n"
