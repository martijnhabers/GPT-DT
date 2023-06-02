import torch
import pandas as pd

model = torch.hub.load(
    "yolov5", "custom", path="models/Yolov5_orientation", source="local"
)

def vehicle_detection(image):
    results = model(image)

    # Results
    #    results.print()
    #    results.save()
    x = results.xyxy[0]

    if torch.cuda.is_available():
        return x.cpu()
    else:
        return x
