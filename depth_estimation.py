# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:03:09 2023

@author: Mees
"""
from PIL import Image
import pandas as pd
import os

h1 = 80#45
h2 = 170#80
# h3 = 120


# df = pd.read_csv(r"C:/Users/Mees/Desktop/vraag 5.csv")

def depth_estimation(df, image_depth):
    
    #load image
    img = Image.open("Depth_map_images/" + image_depth)
    width, height = img.size
    
    # get depth value for each object
    for r in range(0,len(df.index)):
        
          R,G,B,O = img.getpixel((df.loc[r]["x_crop_absoluut_midden"], df.loc[r]['y_crop_absoluut_midden']))
          df.at[r, "RGB"] = R
          if R <= h1:
            df.at[r, 'height_position'] = 'in the distance'
          # elif h1 < R <= h2:
          #   df.at[r, 'height_position'] = 'almost 100m away'
          elif h1 < R <= h2:#h2 < R <= h3:
            df.at[r, 'height_position'] = 'a few tens of meters away'
          else:
            df.at[r, 'height_position'] = 'a few meters away'
          
    return (df)

# image_input = "vraag 5.jpg"    
# df , image_depth = depth_estimation(df, image_input)