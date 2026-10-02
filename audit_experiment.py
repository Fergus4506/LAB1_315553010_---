"""Audit the superseded individual-image runs (not the corrected experiment)."""

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import (ChestXrayDataset, collect_samples, counts_to_metrics,
                   create_model, make_transforms, run_epoch, stratified_split)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    summary = json.loads((RESULTS / "comparison_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((RESULTS / "combined_epoch_metrics.csv").open(encoding="utf-8")))
    checks = []
    for row in rows:
        for split in ("train", "validation", "test"):
            counts = [int(row[f"{split}_{key}"]) for key in ("tn", "fp", "fn", "tp")]
            scores = counts_to_metrics(*counts)
            assert sum(counts) == sum(metadata[f"{split}_counts" if split != "validation" else "validation_counts"].values())
            for key in ("accuracy", "precision", "recall", "f1"):
                assert math.isclose(scores[key], float(row[f"{split}_{key}"]), abs_tol=1e-12)
    for name, item in summary.items():
        model_rows = [row for row in rows if row["model"] == name]
        assert [int(row["epoch"]) for row in model_rows] == list(range(1, 13))
        selected = max(model_rows, key=lambda row: float(row["validation_f1"]))
        assert int(selected["epoch"]) == item["validation_selected_epoch"]
        for metric in ("accuracy", "f1"):
            best = max(model_rows, key=lambda row: float(row[f"test_{metric}"]))
            assert int(best["epoch"]) == item[f"highest_observed_test_{metric}"]["epoch"]
            assert math.isclose(float(best[f"test_{metric}"]), item[f"highest_observed_test_{metric}"][metric], abs_tol=1e-12)
        for key, value in item["final_test_at_validation_selected_epoch"].items():
            assert math.isclose(float(selected[f"test_{key}"]), value, abs_tol=1e-12)
    checks.append("All 36 rows, split totals, confusion-derived metrics and summary selections agree")
    print(checks[-1], flush=True)

    data = ROOT / "data" / "chest_xray"
    train, validation = stratified_split(collect_samples(data, "train"), .15, 315553010)
    validation += collect_samples(data, "val")
    splits = {"train": train, "validation": validation, "test": collect_samples(data, "test")}
    for split, samples in splits.items():
        assert {str(k): v for k, v in Counter(label for _, label in samples).items()} == metadata[f"{split}_counts"]
    hash_groups = defaultdict(list)
    all_samples = [(split, path, label) for split, samples in splits.items() for path, label in samples]
    def digest(sample):
        split, path, label = sample
        return hashlib.sha256(path.read_bytes()).hexdigest(), split, str(path.relative_to(data)), label
    with ThreadPoolExecutor(max_workers=8) as executor:
        for digest_value, split, path, label in executor.map(digest, all_samples):
            hash_groups[digest_value].append({"split": split, "path": path, "label": label})
    duplicates = [group for group in hash_groups.values() if len({entry["split"] for entry in group}) > 1]
    people = {}
    for split, samples in splits.items():
        people[split] = {re.match(r"person(\d+)_", path.name).group(1)
                         for path, label in samples if label == 1 and re.match(r"person(\d+)_", path.name)}
    overlaps = {f"{a}__{b}": len(people[a] & people[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    print("Cross-split identical-file groups:", len(duplicates), "Pneumonia filename-person overlaps:", overlaps, flush=True)

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    _, transform = make_transforms(224)
    loaders = {split: DataLoader(ChestXrayDataset(samples, transform), batch_size=32,
                               num_workers=2, pin_memory=True)
               for split, samples in splits.items() if split != "train"}
    class_weights = torch.tensor([4434/(2*1140), 4434/(2*3294)], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    evaluations = {}
    for name, item in summary.items():
        folder = RESULTS / "resnet101" if name == "resnet101" else RESULTS
        checkpoint = torch.load(folder / f"{name}_best_validation.pt", map_location="cpu", weights_only=True)
        model, _ = create_model(name, False)
        model.load_state_dict(checkpoint["state_dict"])
        counts = {"total": sum(p.numel() for p in model.parameters()),
                  "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        assert checkpoint["epoch"] == item["validation_selected_epoch"]
        assert checkpoint["weights"] == item["weights"] and model.fc.out_features == 2
        model.to(device)
        scores = {split: run_epoch(model, loader, criterion, device) for split, loader in loaders.items()}
        exact = all(scores["test"][key] == item["final_test_at_validation_selected_epoch"][key]
                    for key in ("tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1"))
        evaluations[name] = {"checkpoint_epoch": checkpoint["epoch"], "parameters": counts,
                             "evaluation": scores, "test_matches_recorded": exact,
                             "validation_f1_matches_recorded": math.isclose(scores["validation"]["f1"], item["best_validation_f1"], abs_tol=1e-12)}
        assert exact and evaluations[name]["validation_f1_matches_recorded"]
        print(name, "checkpoint metrics reproduced", scores["test"], flush=True)
        del model
        torch.cuda.empty_cache()
    audit = {"audit_date": "2026-10-02", "gpu": torch.cuda.get_device_name(0),
             "history_checks": checks, "counts": {split: dict(Counter(label for _, label in samples)) for split, samples in splits.items()},
             "cross_split_identical_file_groups": duplicates, "pneumonia_filename_person_group_overlap_counts": overlaps,
             "pneumonia_filename_person_group_counts": {split: len(ids) for split, ids in people.items()},
             "checkpoint_reevaluation": evaluations}
    (RESULTS / "audit_20261002.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Audit saved", flush=True)


if __name__ == "__main__":
    main()
