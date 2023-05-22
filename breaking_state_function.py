# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:34:19 2023

@author: Mees
"""

import numpy as np
import cv2
import pandas as pd
import torch
from tensorflow import keras
from PIL import Image, ImageDraw

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

classes_rem = {
    0: "Niks",
    1: "remt",
}


def Breaking(row, df):
    bord_crop = df.iloc[row]["foto_naam"]
    model = keras.models.load_model("models/model_rem.keras")  # juiste plek aangeven!

    data = []

    image = cv2.imread(bord_crop)
    image_fromarray = Image.fromarray(image, "RGB")
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    data.append(np.array(resize_image))

    X = np.array(data)
    X = X / 255

    pred = np.argmax(model.predict(X), axis=1)
    print(classes_rem[int(pred)])
    df.loc[row, "state"] = classes_rem[int(pred)] + df.loc[row, "state"]
