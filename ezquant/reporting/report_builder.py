from __future__ import annotations

from html import escape


def build_report_html(manifest, qc_report, outputs, watermark_failed: bool = False) -> str:
    watermark = "<h2 style='color:#b00;'>FAILED QC</h2>" if watermark_failed else ""

    qc_rows = "".join(
        f"<tr><td>{escape(c.id)}</td><td>{escape(c.status.value)}</td><td>{escape(c.message)}</td></tr>"
        for c in qc_report.checks
    )

    results_section = "<p>No final results exported.</p>"
    if outputs is not None and outputs.tables:
        table = outputs.tables[0]
        header = "".join(f"<th>{escape(k)}</th>" for k in (table.rows[0].keys() if table.rows else []))
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row.values()) + "</tr>"
            for row in table.rows[:20]
        )
        results_section = (
            f"<h3>{escape(table.name)}</h3><table border='1'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        )

    return f"""
<html>
<head><meta charset='utf-8'><title>EZQUANT Report</title></head>
<body>
<h1>EZQUANT Analysis Report</h1>
{watermark}
<h2>Recipe</h2>
<p>{escape(manifest.recipe.name if manifest.recipe else 'N/A')} v{escape(manifest.recipe.version if manifest.recipe else 'N/A')}</p>
<h2>Policy</h2>
<p>{escape(manifest.lab_policy.policy_name if manifest.lab_policy else 'N/A')} ({escape(manifest.lab_policy.policy_version if manifest.lab_policy else 'N/A')})</p>
<p>Policy hash: {escape(manifest.lab_policy.policy_hash if manifest.lab_policy else 'N/A')}</p>
<h2>QC Summary ({escape(qc_report.overall_status.value)})</h2>
<table border='1'><thead><tr><th>ID</th><th>Status</th><th>Message</th></tr></thead><tbody>{qc_rows}</tbody></table>
<h2>Results</h2>
{results_section}
</body>
</html>
"""
