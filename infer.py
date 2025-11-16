"""Simple YOLOv8 inference wrapper.

Usage:
  python infer.py --weights runs/train/exp/weights/best.pt --source test/images --save

This script runs detection and optionally saves annotated images to a folder.
"""
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True, help='path to trained weights .pt')
    p.add_argument('--source', default='test/images', help='image, video or folder')
    p.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    p.add_argument('--save', action='store_true', help='save annotated results')
    p.add_argument('--save-txt', action='store_true', help='save detection labels as txt')
    p.add_argument('--project', default='runs/detect', help='save folder')
    p.add_argument('--name', default=None, help='experiment name')
    return p.parse_args()

def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except Exception:
        print('Failed to import ultralytics. Install requirements from requirements.txt')
        raise

    model = YOLO(args.weights)
    print(f"Running inference: weights={args.weights} source={args.source}")

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
    )

    # Print a short summary
    print(f"Completed inference. {len(results)} result sets saved to {args.project}/{args.name or 'exp'}")

if __name__ == '__main__':
    main()
