# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:57:32 2023

@author: Mees
"""
import torch
from PIL import Image
import os

# import numpy as np
import pandas as pd

repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)


def create_depth_map(image_input):
    filename, extension = os.path.splitext(image_input)
    image_depth_map = filename + ".csv"
    file_path = "Depth_map_csv/" + image_depth_map

    # Zoe_NK

    ##### sample prediction
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)

    # Local file
    image = Image.open("images/" + image_input).convert("RGB")  # load

    # Create depth estimation, save as csv and return as a dataframe
    depth = zoe.infer_pil(image)
    depth_df = pd.DataFrame(depth)
    depth_df.to_csv(file_path, index=False)

    image.close()
    return depth_df


# image_input = "vraag 5.jpg"
# depth_map = create_depth_map(image_input)
