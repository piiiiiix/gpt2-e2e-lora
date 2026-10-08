'''python -m streamlit run src/11_CIDEr_analysis.py'''

"""Render the six FFT runs, relative differences, and descriptive variability."""
import csv
import itertools
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "outputs/benchmarks"
BASE = ROOT / "fft_benchmark_comparison_20261006_1852-1944"
METRICS = ("CIDEr", "ROUGE-L", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "NIST", "METEOR")


def relative_difference(a, b):
    average = (a + b) / 2
    return abs(a - b) / average if average != 0 else None


def threshold_color(value):
    if value is None or value < 0.05:
        return None
    if value <= 0.10:
        return "#fff0b3"
    if value <= 0.20:
        return "#ffe1dc"
    return "#f2a59b"


def percent(value):
    return "N/A" if value is None else f"{value:.2%}"


def write_csv(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def styled_table(ax, headers, rows, widths=None, grouped=False, fontsize=11):
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                     cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#203a5f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10)
        else:
            color = ["#edf4fc", "#f2f7ef", "#fff5e9"][(r - 1) // 2] if grouped else (
                "#f4f7fb" if r % 2 else "white")
            cell.set_facecolor(color)
    return table


def main():
    with BASE.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as stream:
        runs = list(csv.DictReader(stream))
    runs.sort(key=lambda r: (int(r["Seed"]), r["Model training timestamp"]))
    if [int(r["Seed"]) for r in runs] != [42, 42, 43, 43, 44, 44]:
        raise ValueError("Expected two runs for each seed: 42, 43, 44")
    differences, summaries, rows, colors = [], [], [], []
    for metric in METRICS:
        values = [float(r[metric]) for r in runs]
        record = {"metric": metric, "min": min(values), "max": max(values),
                  "range": max(values) - min(values)}
        row = [metric] + [f"{record[k]:.6f}" for k in ("min", "max", "range")]
        relative_values = []
        for seed, i in zip((42, 43, 44), (0, 2, 4)):
            absolute = abs(values[i] - values[i + 1])
            relative = relative_difference(values[i], values[i + 1])
            record[f"seed{seed}_delta"] = absolute
            record[f"seed{seed}_relative_difference"] = relative
            row.extend([f"{absolute:.6f}", percent(relative)])
            relative_values.append(relative)
        pairs = [(relative_difference(values[i], values[j]), i, j)
                 for i, j in itertools.combinations(range(6), 2)
                 if runs[i]["Seed"] != runs[j]["Seed"]]
        # Select the largest relative difference, keeping its paired absolute difference.
        relative, i, j = max(pairs, key=lambda p: float("-inf") if p[0] is None else p[0])
        absolute = abs(values[i] - values[j])
        record.update(different_seed_max_relative_difference=relative,
                      different_seed_delta_at_max_relative=absolute,
                      different_seed_pair=f"{runs[i]['Seed']}/{runs[j]['Seed']}",
                      different_seed_run_a=runs[i]["Model training timestamp"],
                      different_seed_run_b=runs[j]["Model training timestamp"])
        row.extend([f"{absolute:.6f}", percent(relative), record["different_seed_pair"]])
        relative_values.append(relative)
        differences.append(record)
        rows.append(row)
        colors.append(relative_values)
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        summaries.append({"metric": metric, "n_runs": len(values), "mean": mean,
                          "sample_SD": sd, "CV": sd / mean if mean != 0 else None})

    fig = plt.figure(figsize=(26, 15), dpi=180, facecolor="white")
    fig.text(.03, .96, "FFT benchmark comparison | 2026-10-06 18:52 - 19:44",
             fontsize=22, weight="bold", color="#172b4d")
    fig.text(.03, .928, "Relative difference = |a-b| / ((a+b)/2) | <5%: no color | 5%-10%: yellow | >10%: red | >20%: dark red",
             fontsize=12, color="#52647a")
    original = [[r["Seed"], r["Model training timestamp"].replace("_", "\n", 1)] +
                [f"{float(r[k]):.6f}" for k in METRICS] for r in runs]
    styled_table(fig.add_axes([.03, .67, .94, .225]),
                 ["Seed", "Model training timestamp"] + list(METRICS), original,
                 [.05, .19] + [.095] * 8, grouped=True)
    fig.text(.03, .635, "Per-metric range and pairwise differences", fontsize=17,
             weight="bold", color="#172b4d")
    headers = ["Metric", "Min (6 runs)", "Max (6 runs)", "Max - Min"]
    for seed in (42, 43, 44):
        headers.extend([f"Seed {seed}\nAbsolute diff.", f"Seed {seed}\nRelative diff."])
    headers.extend(["Different seeds\nAbsolute diff.*", "Different seeds\nMax relative diff.", "Seeds of max\nrelative diff."])
    table = styled_table(fig.add_axes([.03, .365, .94, .245]), headers, rows, fontsize=10)
    for r, relative_values in enumerate(colors, 1):
        for c, value in zip((5, 7, 9, 11), relative_values):
            color = threshold_color(value)
            if color:
                table[r, c].set_facecolor(color)
                table[r, c].get_text().set_weight("bold")
                table[r, c].get_text().set_color("#172b4d" if value <= .10 else "#a82318")
    fig.text(.03, .332, "Run-to-run variability | 6 runs per metric", fontsize=17,
             weight="bold", color="#172b4d")
    summary_rows = [[r["metric"], str(r["n_runs"]), f"{r['mean']:.6f}",
                     f"{r['sample_SD']:.6f}", percent(r["CV"])] for r in summaries]
    styled_table(fig.add_axes([.03, .105, .94, .205]),
                 ["Metric", "Runs", "Mean", "Sample SD (n-1)", "CV = SD / mean"], summary_rows)
    fig.text(.03, .074, "Colors indicate relative difference thresholds, not statistical significance. Zero denominators are shown as N/A.",
             fontsize=11, color="#52647a")
    fig.text(.03, .05, "Same-seed comparisons use two training runs; different-seed maxima use all 12 cross-seed pairs. *Absolute difference for the pair with maximum relative difference.",
             fontsize=11, color="#52647a")
    fig.text(.03, .026, "Sample SD and CV describe all six runs together, including variation across seeds and repeated training runs.",
             fontsize=11, color="#52647a")
    fig.savefig(BASE.with_suffix(".png"), dpi=180)
    plt.close(fig)
    write_csv(ROOT / "fft_benchmark_differences_20261006_1852-1944.csv", differences)
    write_csv(ROOT / "fft_benchmark_summary_20261006_1852-1944.csv", summaries)
    # Display CV in percent, keeping the metric order used in the comparison table.
    cv_fig, cv_ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    cv_fig.patch.set_facecolor("white")
    cv_values = [r["CV"] * 100 if r["CV"] is not None else 0 for r in summaries]
    bars = cv_ax.barh([r["metric"] for r in summaries], cv_values,
                      color="#385e8d", height=0.64)
    cv_ax.invert_yaxis()
    for bar, record in zip(bars, summaries):
        cv_ax.annotate(percent(record["CV"]),
                       (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                       xytext=(6, 0), textcoords="offset points", va="center",
                       fontsize=11, color="#172b4d")
    cv_ax.set_xlim(0, max(cv_values) * 1.18 if max(cv_values) > 0 else 1)
    cv_ax.set_xlabel("CV (%) = sample SD / mean", fontsize=11, color="#52647a")
    cv_ax.set_axisbelow(True)
    cv_ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    cv_ax.tick_params(axis="both", labelsize=11, length=0, colors="#172b4d")
    for spine in cv_ax.spines.values():
        spine.set_visible(False)
    cv_fig.text(.12, .94, "FFT run-to-run variability | CV", fontsize=19,
                weight="bold", color="#172b4d")
    cv_fig.text(.12, .89, "2026-10-06 18:52 - 19:44 | 6 runs per metric | sample SD (n-1)",
                fontsize=11, color="#52647a")
    cv_fig.text(.12, .04, "CV describes variability across seeds and repeated training runs; it is not a significance test.",
                fontsize=10, color="#52647a")
    cv_fig.subplots_adjust(left=.12, right=.94, top=.83, bottom=.16)
    cv_fig.savefig(ROOT / "fft_benchmark_cv_20261006_1852-1944.png", dpi=180)
    plt.close(cv_fig)
    print(BASE.with_suffix(".png"))


if __name__ == "__main__":
    main()
