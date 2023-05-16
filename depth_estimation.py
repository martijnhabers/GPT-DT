# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:03:09 2023

@author: Mees
"""

def depth_estimation(df,depth_map_df):
    for r in range(0,len(df.index)):
        df.at[r,'height_position'] = int(depth_map_df.iat[int(df.at[r,'y_crop_absoluut_midden']),int(df.at[r, 'x_crop_absoluut_midden'])])
        
    return (df)