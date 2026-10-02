"""Predict NORMAL/PNEUMONIA with a validation-selected checkpoint."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from train import CLASS_NAMES, create_model, make_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _ = create_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    _, transform = make_transforms(args.image_size)
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
    print(f"Device: {device}")
    print(f"Prediction: {CLASS_NAMES[int(probabilities[1] > probabilities[0])]}")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {probability:.4f}")


if __name__ == "__main__":
    main()
