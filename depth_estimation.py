# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:03:09 2023

@author: Mees
"""
from PIL import Image
import pandas as pd
import numpy as np

def depth_estimation(df, image_input):
    
    # Load image
    img = Image.open("/Depth_map_images/" + image_input)
    width, height = img.size
    
    #TABLE
    classnum_name = 'class'
    xmid_name = 'x_midden'
    ymid_name = 'y_midden'
    b_name = 'breedte'
    h_name = 'hoogte'
    pred_name = 'predictions'
    class_name = 'class_naam'
    state_name = 'state'
    foto_name = 'foto_naam'
    hp_name = 'height_position'
    wp_name = 'width_position'
    pos_name = 'position'
    
            #Define distances
         #   closeness = pixel_color[0]
          #  if 151 > closeness > 100: 
        #      distance = "medium far"
        #    elif 51 < closeness < 150:
       #       distance = "far"
           # elif closeness < 50:
          #    continue
         #  else: 
         #     distance = "close"
    
    Dist = pd.DataFrame(df.loc[:,'%s'%(class_name)])
    Dist['Dist'] = np.zeros(len(Dist.index))
    for r in range(0,len(df.index)):
          x = df.loc[r,'%s'%xmid_name]
          y = df.loc[r,'%s'%ymid_name]
          R,G,B,O = img.getpixel((x, y))
          Dist.loc[r,'RGB'] = R
          Dist.loc[r,'x'] = x
          Dist.loc[r,'y'] = y
          if R <= 45:
            Dist.loc[r,'Dist'] = 'in the distance'
          elif 45 < R <= 80:
            Dist.loc[r,'Dist'] = 'almost 100m away'
          elif 80 < R <= 120:
            Dist.loc[r,'Dist'] = 'a few tens of meters away'
          else:
            Dist.loc[r,'Dist'] = 'a few meters away'
          
    print(Dist)