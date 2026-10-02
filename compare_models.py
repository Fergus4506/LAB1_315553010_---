"""Merge three recorded runs and draw report figures without retraining."""

import csv
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / "data" / "mplcache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RESULTS = ROOT / "results"
ORDER = ("resnet18", "resnet50", "resnet101")
COLORS = {"resnet18": "#2166ac", "resnet50": "#b2182b", "resnet101": "#1b7837"}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_trends(rows, metric, output):
    fig, ax = plt.subplots(figsize=(9, 5.0))
    for name in ORDER:
        model_rows = [row for row in rows if row["model"] == name]
        epochs = [int(row["epoch"]) for row in model_rows]
        for split, linestyle in (("train", "--"), ("test", "-")):
            values = [float(row[f"{split}_{metric}"]) for row in model_rows]
            if metric == "accuracy":
                values = [100 * value for value in values]
            ax.plot(epochs, values, color=COLORS[name], linestyle=linestyle,
                    linewidth=2.0, marker="o" if split == "test" else None,
                    markersize=3, label=f"{name} {split}")
    ax.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else "Pneumonia F1",
           title=f"Training and test {'accuracy' if metric == 'accuracy' else 'F1'} by epoch")
    ax.set_xticks(range(1, 13))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.16), loc="upper center")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name in ORDER:
        model_rows = [row for row in rows if row["model"] == name]
        epochs = [int(row["epoch"]) for row in model_rows]
        axes[0].plot(epochs, [100 * float(row["validation_accuracy"]) for row in model_rows],
                     color=COLORS[name], marker="o", markersize=3, label=name)
        axes[1].plot(epochs, [float(row["validation_f1"]) for row in model_rows],
                     color=COLORS[name], marker="o", markersize=3, label=name)
    for ax, ylabel, title in zip(axes, ("Accuracy (%)", "Pneumonia F1"),
                                 ("Validation accuracy", "Validation F1")):
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.set_xticks(range(1, 13, 2))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_matrices(rows, summary, output, selection):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, name in zip(axes, ORDER):
        epoch = (summary[name]["highest_observed_test_accuracy"]["epoch"]
                 if selection == "highest" else summary[name]["validation_selected_epoch"])
        row = next(row for row in rows if row["model"] == name and int(row["epoch"]) == epoch)
        matrix = np.array([[int(row["test_tn"]), int(row["test_fp"])],
                           [int(row["test_fn"]), int(row["test_tp"])]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=390)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if matrix[i, j] > 195 else "black")
        ax.set_xticks((0, 1), ("Normal", "Pneumonia"), fontsize=8)
        ax.set_yticks((0, 1), ("Normal", "Pneumonia"), fontsize=8)
        ax.set(xlabel="Predicted", ylabel="Actual" if ax is axes[0] else "",
               title=f"{name}\nepoch {epoch}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_log_excerpt(summary, output):
    display_lines = []
    for name in ORDER:
        log_path = RESULTS / "training.log"
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        epochs = {summary[name]["highest_observed_test_accuracy"]["epoch"],
                  summary[name]["highest_observed_test_f1"]["epoch"]}
        for epoch in sorted(epochs):
            matches = [line for line in lines if line.startswith(f"{name} epoch {epoch:02d}/")]
            if len(matches) != 1:
                raise ValueError(f"Missing actual training log for {name} epoch {epoch}")
            parts = matches[0].split("; ")
            display_lines.extend([parts[0] + ";", "  " + parts[1] + ";", "  " + parts[2]])
    font_paths = (Path(r"C:\Windows\Fonts\consola.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
    font_path = next((path for path in font_paths if path.exists()), None)
    font = ImageFont.truetype(str(font_path), 22) if font_path else ImageFont.load_default(size=22)
    title_font = ImageFont.truetype(str(font_path), 24) if font_path else ImageFont.load_default(size=24)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(850, max(draw_probe.textbbox((0, 0), line, font=font)[2]
                         for line in display_lines) + 70)
    image = Image.new("RGB", (width, 85 + 33 * len(display_lines)), "#152331")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 45), fill="#263c4e")
    draw.text((25, 10), "Training log: highest observed test scores",
              font=title_font, fill="#eaf3f7")
    for index, line in enumerate(display_lines):
        draw.text((25, 62 + 33 * index), line, font=font, fill="#e5f2d4")
    image.save(output)


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=ROOT / "results" / "corrected_group_split_20261002")
    args=parser.parse_args()
    RESULTS=args.results_dir
    rows = load_rows(RESULTS / "epoch_metrics.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    if set(summary) != set(ORDER) or any(sum(row["model"] == name for row in rows) != 12 for name in ORDER):
        raise ValueError("Expected 12 epochs and a completed summary for every model")
    with (RESULTS / "combined_epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_trends(rows, "accuracy", RESULTS / "comparison_accuracy.png")
    plot_trends(rows, "f1", RESULTS / "comparison_f1.png")
    plot_validation(rows, RESULTS / "comparison_validation.png")
    plot_matrices(rows, summary, RESULTS / "comparison_highest_test_confusion.png", "highest")
    plot_matrices(rows, summary, RESULTS / "comparison_final_test_confusion.png", "final")
    plot_log_excerpt(summary, RESULTS / "comparison_log_excerpt.png")
    print("Compared:", ", ".join(ORDER))


if __name__ == "__main__":
    main()
