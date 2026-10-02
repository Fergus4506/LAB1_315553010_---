"""Train selected ResNets on the Kaggle chest X-ray dataset.

The public test split is evaluated each epoch because the lab explicitly asks
for test curves. Checkpoint selection uses validation F1 only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import textwrap
import hashlib
import sys
from split_dataset import MANIFEST, load_samples
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
(ROOT / "data" / "mplcache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "mplcache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "torch_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


class ChestXrayDataset(Dataset):
    """Custom image loader with split-specific transforms and explicit labels."""

    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def collect_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    samples = []
    for label, name in enumerate(CLASS_NAMES):
        folder = root / split / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
        samples += [(path, label) for path in sorted(folder.rglob("*"))
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if not samples:
        raise ValueError(f"No images found in {root / split}")
    return samples


def make_transforms(size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=7, translate=(0.03, 0.03),
                                scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(), normalize,
    ])
    return train_transform, eval_transform


def create_model(name: str, pretrained: bool):
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "resnet101":
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {name}")
    model.fc = nn.Linear(model.fc.in_features, 2)  # always reinitialized
    for param_name, parameter in model.named_parameters():
        parameter.requires_grad = param_name.startswith(("layer3.", "layer4.", "fc."))
    return model, str(weights) if weights is not None else "none"


def freeze_batchnorm_in_frozen_stages(model):
    for stage in (model.conv1, model.bn1, model.layer1, model.layer2):
        stage.eval()


def counts_to_metrics(tn: int, fp: int, fn: int, tp: int):
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tn=tn, fp=fp, fn=fn, tp=tp)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batchnorm_in_frozen_stages(model)
    loss_sum = 0.0
    tn = fp = fn = tp = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        loss_sum += loss.item() * len(labels)
        predictions = logits.argmax(dim=1)
        tn += int(((labels == 0) & (predictions == 0)).sum())
        fp += int(((labels == 0) & (predictions == 1)).sum())
        fn += int(((labels == 1) & (predictions == 0)).sum())
        tp += int(((labels == 1) & (predictions == 1)).sum())
    return dict(loss=loss_sum / len(loader.dataset), **counts_to_metrics(tn, fp, fn, tp))


def plot_curves(history: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name in sorted({row["model"] for row in history}):
        rows = [row for row in history if row["model"] == name]
        epochs = [row["epoch"] for row in rows]
        for split, style in (("train", "-"), ("validation", "--"), ("test", ":")):
            axes[0].plot(epochs, [100 * row[f"{split}_accuracy"] for row in rows],
                         style, label=f"{name} {split}")
            axes[1].plot(epochs, [row[f"{split}_f1"] for row in rows],
                         style, label=f"{name} {split}")
    axes[0].set(ylabel="Accuracy (%)", title="Accuracy by epoch")
    axes[1].set(ylabel="F1 score", title="F1 by epoch")
    for ax in axes:
        ax.set(xlabel="Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    for metric, ylabel, title, filename in (
        ("accuracy", "Accuracy (%)", "Training, validation and test accuracy", "accuracy_curve.png"),
        ("f1", "F1 score", "Training, validation and test F1", "f1_curve.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for name in sorted({row["model"] for row in history}):
            rows = [row for row in history if row["model"] == name]
            for split, style in (("train", "-"), ("validation", "--"), ("test", ":")):
                values = [row[f"{split}_{metric}"] for row in rows]
                if metric == "accuracy":
                    values = [100 * value for value in values]
                ax.plot([row["epoch"] for row in rows], values, style,
                        linewidth=1.8, label=f"{name} {split}")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=9, loc="lower right")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(output.with_name(filename), dpi=180)
        plt.close(fig)


def plot_confusion(metrics: dict, output: Path, title: str):
    matrix = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks((0, 1), CLASS_NAMES)
    ax.set_yticks((0, 1), CLASS_NAMES)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title("\n".join(textwrap.wrap(title, width=32)), fontsize=10, pad=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing chest_xray/train, val and test")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "experiment")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=315553010)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--models", nargs="+", choices=("resnet18", "resnet50", "resnet101"),
                        default=("resnet18", "resnet50"))
    parser.add_argument("--split-manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        parser.error("Output already contains a completed run; choose a new output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this assignment run")
    if args.epochs < 1 or not 0 < args.validation_fraction < 1:
        parser.error("epochs must be positive and validation-fraction in (0, 1)")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits, split_info = load_samples(args.data_dir, args.split_manifest)
    if split_info['seed'] != args.seed or split_info['validation_fraction'] != args.validation_fraction:
        raise ValueError("Manifest seed/fraction differ from training arguments")
    train_samples, validation_samples, test_samples = (splits[s] for s in ('train','validation','test'))
    (args.output_dir / 'split_manifest.json').write_bytes(args.split_manifest.read_bytes())
    train_tf, eval_tf = make_transforms(args.image_size)
    train_loader = DataLoader(ChestXrayDataset(train_samples, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ChestXrayDataset(validation_samples, eval_tf),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(ChestXrayDataset(test_samples, eval_tf),
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True)
    train_counts = Counter(label for _, label in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / (2 * train_counts[i]) for i in (0, 1)],
        dtype=torch.float32, device=device)
    metadata = dict(student_id="315553010", student_name="楊敦傑",
                    dataset="paultimothymooney/chest-xray-pneumonia",
                    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    torchvision=__import__("torchvision").__version__,
                    cuda=torch.version.cuda, args={k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
                    python=sys.version, python_executable=sys.executable, venv_prefix=sys.prefix,
                    split_method=split_info["method"],
                    split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
                    seed_reset_per_model=True, cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    train_counts=dict(train_counts),
                    validation_counts=dict(Counter(label for _, label in validation_samples)),
                    test_counts=dict(Counter(label for _, label in test_samples)))
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    training_log = args.output_dir / "training.log"
    training_log.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    all_rows = []
    summary = {}

    for name in args.models:
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, weight_id = create_model(name, not args.no_pretrained)
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda")
        best_val_f1 = -1.0
        best_epoch = 0
        best_test_accuracy_row = None
        best_test_f1_row = None
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            validation = run_epoch(model, val_loader, criterion, device)
            test = run_epoch(model, test_loader, criterion, device)
            row = {"model": name, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            for split, metrics in (("train", train), ("validation", validation), ("test", test)):
                row.update({f"{split}_{key}": value for key, value in metrics.items()})
            all_rows.append(row)
            if validation["f1"] > best_val_f1:
                best_val_f1 = validation["f1"]
                best_epoch = epoch
                torch.save({"model": name, "epoch": epoch,
                            "state_dict": model.state_dict(), "weights": weight_id,
                            "class_names": CLASS_NAMES, "split_manifest_sha256": metadata["split_manifest_sha256"]},
                           args.output_dir / f"{name}_best_validation.pt")
                plot_confusion(test, args.output_dir / f"{name}_final_test_confusion.png",
                               f"{name}: test at validation-selected epoch {epoch}")
            if best_test_accuracy_row is None or test["accuracy"] > best_test_accuracy_row["test_accuracy"]:
                best_test_accuracy_row = row.copy()
                plot_confusion(test, args.output_dir / f"{name}_highest_test_accuracy_confusion.png",
                               f"{name}: highest observed test accuracy (epoch {epoch})")
            if best_test_f1_row is None or test["f1"] > best_test_f1_row["test_f1"]:
                best_test_f1_row = row.copy()
            scheduler.step()
            log_line = (f"{name} epoch {epoch:02d}/{args.epochs}: "
                  f"train acc={train['accuracy']:.4f} f1={train['f1']:.4f}; "
                  f"val acc={validation['accuracy']:.4f} f1={validation['f1']:.4f}; "
                  f"test acc={test['accuracy']:.4f} f1={test['f1']:.4f}")
            print(log_line, flush=True)
            with training_log.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")
            with (args.output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            plot_curves(all_rows, args.output_dir / "accuracy_f1_curves.png")
        summary[name] = dict(weights=weight_id, validation_selected_epoch=best_epoch,
                             best_validation_f1=best_val_f1,
                             final_test_at_validation_selected_epoch=next(
                                 {key.removeprefix("test_"): value for key, value in row.items()
                                  if key.startswith("test_")}
                                 for row in all_rows if row["model"] == name and row["epoch"] == best_epoch),
                             highest_observed_test_accuracy={"epoch": best_test_accuracy_row["epoch"],
                                                             "accuracy": best_test_accuracy_row["test_accuracy"]},
                             highest_observed_test_f1={"epoch": best_test_f1_row["epoch"],
                                                       "f1": best_test_f1_row["test_f1"]},
                             elapsed_seconds=round(time.time() - start, 1))
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    with training_log.open("a", encoding="utf-8") as log_file:
        log_file.write("SUMMARY " + json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
