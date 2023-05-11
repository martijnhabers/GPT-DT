# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:57:32 2023

@author: Mees
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
#import numpy

def create_depth_map(image_input):
 
    repo = "isl-org/ZoeDepth"
    #torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    
    # Zoe_K
    #model_zoe_k = torch.hub.load(repo, "ZoeD_K", pretrained=True)
    
    # Zoe_NK
    model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    
    ##### sample prediction
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_nk.to(DEVICE)
    
    # Local file
    image = Image.open("/images/" + image_input).convert("RGB")  # load 
    depth_numpy = zoe.infer_pil(image)  # as numpy
    plt.imshow(image)
    plt.show()
    
    depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image
    
    depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor
    
    # Tensor 
    from zoedepth.utils.misc import pil_to_batched_tensor
    X = pil_to_batched_tensor(image).to(DEVICE)
    depth_tensor = zoe.infer(X)
    
    #image = get_image_from_url(URL)  # fetch
    depth = zoe.infer_pil(image)
    
    # Save raw
    from zoedepth.utils.misc import save_raw_16bit
    fpath = "/Depth_map_images/" + image_input
    save_raw_16bit(depth, fpath)
    
    # Colorize output
    from zoedepth.utils.misc import colorize
    
    colored = colorize(depth)
    
    # save colored output
    Image.fromarray(colored).save("/Depth_map_images/" + image_input)
    
    # display image
    image = Image.open("/Depth_map_images/" + image_input)
    plt.imshow(image)
    plt.show()
    
# create_depth_map()