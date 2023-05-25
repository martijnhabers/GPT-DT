from ultralytics import YOLO
import os

model = YOLO("models/yolov8ncustom.pt")


def yolo_tri_crop(image):
    if not os.path.isfile(image):
        raise Exception(f"Input image path {image} is not a file!")

    results = model.predict(image, save_crop=True, conf=0.5, project="tri-crop")
    return results
