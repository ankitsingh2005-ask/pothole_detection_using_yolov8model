import cv2
import numpy as np
import argparse
import time
from ultralytics import YOLO

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", default="output_polyline.mp4", help="saved video file name")
    return p.parse_args()

def main():
    args = get_args()

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Error: Cannot open video/webcam")
        return

    # ---- VIDEO WRITER (SAVE VIDEO) ----
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_save = int(cap.get(cv2.CAP_PROP_FPS))
    if fps_save == 0:
        fps_save = 30  # fallback for webcams

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.output, fourcc, fps_save, (width, height))
    print(f"🎥 Saving output video to: {args.output}")

    # ---- FPS Counter ----
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Fix incorrect color channels
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame, conf=0.20)[0]

        # Draw detections
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)

            # Green polyline
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)

            label = f"{model.names[int(cls)]} {float(conf):.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # FPS display
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ---- SAVE FRAME TO VIDEO ----
        out.write(frame)

        # ---- SHOW WINDOW ----
        cv2.imshow("Pothole Polyline Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Video saved successfully!")

if __name__ == "__main__":
    main()
