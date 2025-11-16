"""YOLOv8 Training Script (GPU Auto-Detection + Dataset Pre-Configured)

This script:
 - Uses GPU automatically if available
 - Uses YOLOv8m as the default model
 - Uses your dataset located at:
      C:/project1/My First Project.v2i.yolov8/data.yaml
 - Saves results under runs/train/

Run:
  python train.py --epochs 50 --batch 16
"""

import argparse
import torch
from pathlib import Path


# ------------------ DEFAULT DATASET PATH ------------------
DEFAULT_DATA_PATH = r"C:/project1/My First Project.v2i.yolov8/data.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolov8m.pt',
                  help='YOLOv8 model or path')
    p.add_argument('--data', default=DEFAULT_DATA_PATH,
                  help='dataset YAML path')
    p.add_argument('--epochs', type=int, default=20,
                  help='number of epochs')
    p.add_argument('--batch', type=int, default=16,
                  help='batch size')
    p.add_argument('--project', default='runs/train',
                  help='project folder for results')
    p.add_argument('--name', default='experiment',
                  help='experiment name')
    p.add_argument('--exist-ok', action='store_true',
                  help='overwrite existing project/name')
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------ GPU CHECK ------------------
    if torch.cuda.is_available():
        device = "0"
        print("🔥 GPU detected → using CUDA")
    else:
        device = "cpu"
        print("⚠ No GPU detected → using CPU")

    # ------------------ IMPORT YOLO ------------------
    try:
        from ultralytics import YOLO
    except Exception:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        raise

    # ------------------ LOAD MODEL ------------------
    print(f"📌 Loading model: {args.model}")
    model = YOLO(args.model)

    # ------------------ TRAINING ------------------
    print("\n🚀 Starting training")
    print(f"   Dataset: {args.data}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Device: {device}\n")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        device=device,
    )

    print("\n✅ Training finished successfully!")
    print("📁 Results saved to: runs/train/")
    print("⭐ Best model saved as: runs/train/<exp>/weights/best.pt")


if __name__ == "__main__":
    main()
