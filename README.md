# YOLOv8 training and inference helper

This small helper includes `train.py` and `infer.py` for training and running YOLOv8 on a Roboflow-style dataset (a `data.yaml` in the project root referencing `train/`, `val/`, and `names`).

Setup (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train example:

```powershell
python train.py --model yolov8n.pt --data data.yaml --epochs 50 --batch 16 --project runs/train --name pothole_exp --exist-ok
```

Infer example:

```powershell
python infer.py --weights runs/train/pothole_exp/weights/best.pt --source test/images --save --project runs/detect --name pothole_infer
```

Notes:

- The scripts are intentionally minimal wrappers around the `ultralytics.YOLO` API.
- If you exported from Roboflow, your `data.yaml` should already be in the expected format. Verify the `train` and `val` paths are correct.
- For larger datasets or GPU training, pick a larger model (yolov8s/m/l/x) and adjust batch/epochs accordingly.
