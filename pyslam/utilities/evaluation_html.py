import pandas as pd
from pathlib import Path
from jinja2 import Template
import webbrowser
import html
from datetime import datetime
import getpass


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: #f9f9f9;
            padding: 2rem;
            color: #333;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }
        .table-wrapper {
            margin-bottom: 2.5rem;
        }
        .caption {
            font-weight: 600;
            margin-bottom: 0.2rem;
            color: #222;
        }
        .description {
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            color: #444;
            max-width: 60rem;
        }
        .direction {
            font-weight: 400;
            font-size: 0.85rem;
            color: #555;
        }
        .note {
            font-size: 0.85rem;
            margin: 0 auto 2rem auto;
            max-width: 60rem;
            color: #444;
        }
        td.best {
            font-weight: 700;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border-radius: 6px;
            overflow: hidden;
            font-size: 0.8rem;  /* Smaller font for table content */
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.5rem 0.6rem;
            text-align: left;
        }
        thead th {
            background: #007acc;
            color: white;
            font-weight: 600;
        }
        tbody tr:nth-child(even) {
            background: #f5faff;
        }
    </style>
</head>
<body>

<h1>{{ title }}</h1>
<div style="text-align: center; font-size: 0.85rem;  margin-bottom: 2rem;">
    {{ user }} &nbsp;|&nbsp; {{ date }} &nbsp;|&nbsp; Commit: <code>{{ git_hash }}</code>
</div>

{% if note %}<div class="note">{{ note }}</div>{% endif %}

{% for table in tables %}
<div class="table-wrapper">
    <div class="caption">{{ table.caption }}{% if table.direction %} <span class="direction">({{ table.direction }})</span>{% endif %}</div>
    {% if table.description %}<div class="description">{{ table.description }}</div>{% endif %}
    <table>
        <thead>
            <tr>
                {% for col in table.columns %}
                <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in table.data %}
            <tr>
                {% for cell in row %}
                <td{% if cell.best %} class="best"{% endif %}>{{ cell.text }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endfor %}

</body>
</html>
"""


# What the tables of main_slam_evaluation.py hold, by the metric in their file name (table_<metric>.csv):
# (caption, description, higher_is_better)
kMetricInfo = {
    "rmse": (
        "ATE RMSE (m)",
        "Absolute trajectory error: the root mean square of the position errors of the tracked frames, after "
        "aligning the estimated trajectory to the ground truth (with a scale factor for a monocular camera). In the "
        "units of the ground truth: metres for TUM, KITTI and EuRoC.",
        False,
    ),
    "max": (
        "ATE max (m)",
        "The largest position error of a tracked frame, after the same alignment.",
        False,
    ),
    "percent_tracked": (
        "Tracked (% of the frames)",
        "The frames with a pose in the final trajectory. The trajectory errors above are over these frames only: "
        "a run that lost track early has a short trajectory, and can have a small error.",
        True,
    ),
    "percent_lost": (
        "Lost (% of the frames)",
        "The frames in which tracking was lost or relocalizing.",
        False,
    ),
}

# The rows of a table which are not one value per preset to compare
kRowsNotCompared = ("Std Dev", "Best (Average) Preset", "Best (Average) Metric")

kNote = (
    "Each value is the mean over the runs of a preset (a column) on a dataset (a row); Average and Std Dev are over "
    "all the runs of a preset. Bold: the better value in a row. A difference smaller than the Std Dev is not a "
    "difference: run more times (--runs) before reading it as one."
)


def escape_html_df(df: pd.DataFrame) -> pd.DataFrame:
    def escape(x):
        return html.escape(str(x)) if pd.notna(x) else ""

    # DataFrame.map since pandas 2.1, where applymap is deprecated (a FutureWarning per table)
    return df.map(escape) if hasattr(df, "map") else df.applymap(escape)


def metric_of(csv_path):
    """The metric of a table_<metric>.csv file, or None."""
    stem = Path(csv_path).stem
    return stem[len("table_") :] if stem.startswith("table_") else None


def best_columns(row, higher_is_better):
    """The indices of the cells of a row (after its label) which hold its best value: none when the row is not
    one value per preset, or has fewer than two numbers to compare."""
    if str(row[0]) in kRowsNotCompared:
        return set()
    values = {}
    for i, cell in enumerate(row[1:], start=1):
        try:
            value = float(cell)
        except (TypeError, ValueError):
            continue  # N/A, or an empty cell
        if value == value:  # not NaN
            values[i] = value
    if len(values) < 2:
        return set()
    best = max(values.values()) if higher_is_better else min(values.values())
    if all(v == best for v in values.values()):
        return set()  # all equal: nothing is better
    return {i for i, v in values.items() if v == best}


def csv_list_to_html(
    csv_paths, output_html_path, title="Evaluation Report", git_commit_hash=None, open_browser=True
):
    tables = []
    author_name = getpass.getuser().capitalize()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
        if df.empty:
            continue
        escaped_df = escape_html_df(df)
        info = kMetricInfo.get(metric_of(csv_path))
        higher_is_better = info[2] if info else None
        data = []
        for raw, escaped in zip(df.values.tolist(), escaped_df.values.tolist()):
            best = best_columns(raw, higher_is_better) if info else set()
            data.append([{"text": text, "best": i in best} for i, text in enumerate(escaped)])
        table_data = {
            "caption": info[0] if info else Path(csv_path).stem.replace("_", " ").title(),
            "description": info[1] if info else "",
            "direction": ("higher is better" if higher_is_better else "lower is better") if info else "",
            "columns": escaped_df.columns.tolist(),
            "data": data,
        }
        tables.append(table_data)

    template = Template(html_template)
    rendered_html = template.render(
        tables=tables,
        note=kNote if any(t["direction"] for t in tables) else "",
        title=title,
        user=author_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        git_hash=git_commit_hash,
    )

    Path(output_html_path).write_text(rendered_html, encoding="utf-8")
    print(f"✅ HTML report saved to: {output_html_path}")

    if open_browser:
        webbrowser.open(f"file://{Path(output_html_path).resolve()}")
