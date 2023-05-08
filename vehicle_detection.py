# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:33:27 2023

@author: Mees
"""   
 
import torch
import pandas as pd

#column = [ "x_links", "y_boven","x_rechts", "y_onder", "prediction", 'class']


def vehicle_detection(image):

    model = torch.hub.load('yolov5', 'custom', path='Yolov5_orientation', source='local')
# Load test image, could be any image
#    image = 'C:/Users/Mees/Documents/1. Studie/4. 2022-2023/Q3/BEP/vraag 4.jpg'
#model.conf = 0.5
# Inference
    results = model(image)

# Results
    results.print()
    results.save()
    x = results.xyxy[0]
#    df = pd.DataFrame(x.numpy(), columns=column) 
    return(x)
# x = vehicle_detection(0)


# classes_orientation = ["car_back",
#             'car_side',
#             'car_front',
#             'bus_back',
#             'bus_side',
#             'bus_front',
#             'truck_back',
#             'truck_side',
#             'truck_front',
#             'motorcycle_back',
#             'motorcycle_side',
#             'motorcycle_front',
#             'bicycle_back',
#             'bicycle_side',
#             'bicycle_front'
#             ]

# columns = ["xmin", "ymin","xmax","ymax","predictions", 'class']
# df1 = pd.DataFrame(x.numpy(), columns=columns)
# df1['class_naam'] = df1['class']
# df1["class_naam"].replace(range(int(len(classes_orientation))),classes_orientation, inplace=True)
# df1[['class_naam','state']] = df1['class_naam'].str.split('_', n=1, expand = True)

