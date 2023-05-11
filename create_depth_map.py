# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:57:32 2023

@author: Mees
"""
import torch
from PIL import Image
import os

def create_depth_map(image_input):
 
    filename, extension = os.path.splitext(image_input)
    image_output = filename + ".png"   
 
    repo = "isl-org/ZoeDepth"
    # Zoe_NK
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    
    ##### sample prediction
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)
    
    # Local file
    image = Image.open("images/" + image_input).convert("RGB")  # load 

    
    #image = get_image_from_url(URL)  # fetch
    depth = zoe.infer_pil(image)
    
    # Save raw
    from zoedepth.utils.misc import save_raw_16bit
    fpath = "Depth_map_images/" + image_output
    save_raw_16bit(depth, fpath)
    
    # Colorize output
    from zoedepth.utils.misc import colorize
    
    colored = colorize(depth)
    
    # save colored output
    Image.fromarray(colored).save("Depth_map_images/" + image_output)
    
    # display image
    image = Image.open("Depth_map_images/" + image_output)
    return(image_output)
# image_input = "vraag 13.jpg"
# create_depth_map(image_input)