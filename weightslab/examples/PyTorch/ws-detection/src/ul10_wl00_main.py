"""Stage ul10_wl00 — pure Ultralytics, zero WeightsLab.

Anchor reference for the integration ladder. Each subsequent
ulXX_wlYY_main.py replaces a slice of Ultralytics conveniences
with explicit WL-aware equivalents, ending at main.py (the
current production script, considered ul04_wl06).
"""
import os

from ultralytics import YOLO


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.normpath(os.path.join(HERE, "..", "data", "data.yaml"))
MODEL = "yolo11s.pt"
IMG_SIZE = 1024
EPOCHS = 1
BATCH = 4


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
    )


if __name__ == "__main__":
    main()
