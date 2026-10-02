"""Verify saved models contain the expected ImageNet feature weights."""

import json
import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "torch_cache" / "hub" / "checkpoints"
MODELS = {
    "resnet18": ("resnet18_best_validation.pt", "resnet18-f37072fd.pth", "ResNet18_Weights.IMAGENET1K_V1"),
    "resnet50": ("resnet50_best_validation.pt", "resnet50-11ad3fa6.pth", "ResNet50_Weights.IMAGENET1K_V2"),
    "resnet101": ("resnet101_best_validation.pt", "resnet101-cd907fc2.pth", "ResNet101_Weights.IMAGENET1K_V2"),
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=ROOT/"results"/"corrected_group_split_20261002")
    args=parser.parse_args()
    results = {}
    for name, (checkpoint_name, reference_name, weight_id) in MODELS.items():
        checkpoint_path=args.results_dir/checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reference = torch.load(CACHE / reference_name, map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        matches = torch.equal(state["conv1.weight"], reference["conv1.weight"])
        output_features = state["fc.weight"].shape[0]
        results[name] = {
            "checkpoint_weights_tag": checkpoint["weights"],
            "expected_weights_tag": weight_id,
            "frozen_conv1_matches_imagenet": bool(matches),
            "classification_outputs": int(output_features),
            "verified": checkpoint["weights"] == weight_id and matches and output_features == 2,
        }
    output = args.results_dir / "pretrained_verification.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if not all(result["verified"] for result in results.values()):
        raise SystemExit("Pretrained verification failed")


if __name__ == "__main__":
    main()
