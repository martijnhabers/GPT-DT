# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:33:27 2023

@author: Mees
"""

import torch
import pandas as pd


def vehicle_detection(image):
    model = torch.hub.load(
        "yolov5", "custom", path="models/Yolov5_orientation", source="local"
    )
    results = model(image)

    # Results
    #    results.print()
    #    results.save()
    x = results.xyxy[0]

    return x
