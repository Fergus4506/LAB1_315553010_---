"""Regenerate accuracy and F1 figures from the recorded epoch CSV."""

import argparse
import csv
import json
from pathlib import Path

from train import plot_confusion, plot_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    with (args.results_dir / "epoch_metrics.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("epoch_metrics.csv is empty")
    history = [{key: (value if key == "model" else int(value) if key == "epoch" else float(value))
                for key, value in row.items()} for row in rows]
    plot_curves(history, args.results_dir / "accuracy_f1_curves.png")
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    for model, info in summary.items():
        highest_epoch = info["highest_observed_test_accuracy"]["epoch"]
        final_epoch = info["validation_selected_epoch"]
        for epoch, suffix, title in (
            (highest_epoch, "highest_test_accuracy_confusion", f"{model}: best test accuracy (epoch {highest_epoch})"),
            (final_epoch, "final_test_confusion", f"{model}: validation-selected (epoch {final_epoch})"),
        ):
            row = next(row for row in history if row["model"] == model and row["epoch"] == epoch)
            metrics = {key.removeprefix("test_"): value for key, value in row.items()
                       if key.startswith("test_")}
            plot_confusion(metrics, args.results_dir / f"{model}_{suffix}.png", title)
    print(f"Updated figures in {args.results_dir}")


if __name__ == "__main__":
    main()
