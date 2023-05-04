# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:14:17 2023

@author: emmah
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:49:07 2023

@author: Mees
"""
import numpy as np
import cv2
import pandas as pd
import torch
from tensorflow import keras
from PIL import Image, ImageDraw
import clip
import os
import matplotlib.pyplot as plt

from transformers import OwlViTProcessor, OwlViTForObjectDetection

#import texts van martijn!!

######## aanpassen per foto ########
image_path = r"C:/Users/emmah/Documents/jaar 4/bep/yolo/vraag 1.jpg"


#variabelen
marge = 1.25
orientation_conf = 0.5


column1 = [ "x_links", "y_boven","x_rechts", "y_onder", "prediction", 'klas1']
fotonaam = []
state_niet_auto = [3,4,5,6,7,8,9,10,11,12,13,14]
state_niet_bus = [0,1,2,6,7,8,9,10,11,12,13,14]
state_niet_wagen = [0,1,2,3,4,5,9,10,11,12,13,14]
state_niet_motor = [0,1,2,3,4,5,6,7,8,12,13,14]
state_niet_fiets = [0,1,2,3,4,5,6,7,8,9,10,11]
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3
state = None
####################################


#data framebouwen






def dataframe_bouwen(labels, boxes, scores):


    columns = ["xmin", "ymin","xmax","ymax"]
    df = pd.DataFrame(boxes, columns = columns)
    df['class'] = labels
    df['predictions'] = scores
    df["x_midden"] = (df["xmin"] + df["xmax"])/(2)
    df["y_midden"] = (df["ymin"] + df["ymax"])/(2)
    df['width'] = df["xmax"] - df["xmin"]
    df["height"] = df['ymax'] - df ['ymin']
    df['width'] = df['width'] * marge
    df['height'] = df['height'] * marge
    
    df['xmin'] = df['x_midden'] - 0.5 * df['width']
    df['xmax'] = df['x_midden'] + 0.5 * df['width']
    df['ymin'] = df['y_midden'] - 0.5 * df['height']
    df['ymax'] = df['y_midden'] + 0.5 * df['height']
    
    
    df["class_naam"] = df["class"]
    df["class_naam"].replace(range(int(len(texts))),texts, inplace=True)
    df["state"] = df["class_naam"]
    return(df)




def crop_and_save_image(row, image_path):

    im2 = cv2.imread(image_path)
    height, width, channels = im2.shape
#    x, y, w, h = (float(lines[row][1])*width),(float(lines[row][2])*height), (float(lines[row][3])*marge*width), (float(lines[row][4])*marge*height)
    klas = texts[df["class"][row]]
    x1, y1, x2, y2 = int(df['xmin'][row]*marge_min), int(df['ymin'][row]*marge_min), int(df["xmax"][row]* marge_max), int(df["ymax"][row]*marge_max)
    if y1 < 0:
        y1 = 0
    elif y1 > height:
        y1 = height        
    if x1 < 0:
        x1 = 0
    elif x1 > width:
        x1 = width
    crop_img = im2[y1:y2, x1:x2]
    # plt.imshow(crop_img)
    map_pad = "C:/Users/emmah/Documents/jaar 4/bep/CROPS/"
    bestandsnaam = f"nieuwe_afbeelding_{klas}_{row}.jpg"
    fotonaam.append(map_pad + bestandsnaam)
    cv2.imwrite(map_pad + bestandsnaam, crop_img)

def Car_orientation(row, df):
    model = torch.hub.load('C:/Users/emmah/yolov5', 'custom', path='C:/Users/emmah/Documents/jaar 4/bep/Yolov5_orientation', source='local')
    img = df.iloc[row]["foto_naam"]
    img_data = cv2.imread(img)
    Height_crop, Width_crop, channels = img_data.shape        
    model.conf = orientation_conf
    results = model(img)
    res = results.xyxy[0]
    df1 = pd.DataFrame(res.numpy(), columns = column1)
    df1["x_midden"] = (df1["x_links"] + df1["x_rechts"])/(2*Width_crop)
    df1["y_midden"] = (df1["y_boven"] + df1["y_onder"])/(2*Height_crop)
    for i in range(len(df1['klas1'])):
        if 0.45 < df1['x_midden'][i] < 0.55 and 0.45 < df1["y_midden"][i] < 0.55:
            
            state = int(df1["klas1"][i])

    
    if state == None or state in state_niet_auto:
        print("leeg ouleh")
        df.loc[row, "state"] = " "
    elif state == 0:
        print("achterkant voertuig")
        df.loc[row, "state"] = "back"
    elif state == 1 :
        print("zijkant voertuig")
        df.loc[row, "state"] = "side"            
    elif state == 2 :
        print("voorkant voertuig")
        df.loc[row, "state"] = "front"


def Bus_orientation(row, df):
    model = torch.hub.load('C:/Users/emmah/yolov5', 'custom', path='C:/Users/emmah/Documents/jaar 4/bep/Yolov5_orientation', source='local')
    img = df.iloc[row]["foto_naam"]
    img_data = cv2.imread(img)
    Height_crop, Width_crop, channels = img_data.shape        
    model.conf = orientation_conf
    results = model(img)
    res = results.xyxy[0]
    df1 = pd.DataFrame(res.numpy(), columns = column1)
    df1["x_midden"] = (df1["x_links"] + df1["x_rechts"])/(2*Width_crop)
    df1["y_midden"] = (df1["y_boven"] + df1["y_onder"])/(2*Height_crop)
    for i in range(len(df1['klas1'])):
        if 0.45 < df1['x_midden'][i] < 0.55 and 0.45 < df1["y_midden"][i] < 0.55:
            
            state = int(df1["klas1"][i])
    
    if state == None or state in state_niet_bus:
        print("leeg ouleh")
        df.loc[row, "state"] = " "
    elif state == 3:
        print("achterkant voertuig")
        df.loc[row, "state"] = "back"
    elif state == 4 :
        print("zijkant voertuig")
        df.loc[row, "state"] = "side"            
    elif state == 5 :
        print("voorkant voertuig")
        df.loc[row, "state"] = "front"

    
    
def Truck_orientation(row, df):
        model = torch.hub.load('C:/Users/emmah/yolov5', 'custom', path='C:/Users/emmah/Documents/jaar 4/bep/Yolov5_orientation', source='local')
        img = df.iloc[row]["foto_naam"]
        img_data = cv2.imread(img)
        Height_crop, Width_crop, channels = img_data.shape        
        model.conf = orientation_conf
        results = model(img)
        res = results.xyxy[0]
        df1 = pd.DataFrame(res.numpy(), columns = column1)
        df1["x_midden"] = (df1["x_links"] + df1["x_rechts"])/(2*Width_crop)
        df1["y_midden"] = (df1["y_boven"] + df1["y_onder"])/(2*Height_crop)
        for i in range(len(df1['klas1'])):
            if 0.45 < df1['x_midden'][i] < 0.55 and 0.45 < df1["y_midden"][i] < 0.55:
                
                state = int(df1["klas1"][i])
        
        if state == None or state in state_niet_wagen:
            print("leeg ouleh")
            df.loc[row, "state"] = " "
        elif state == 6:
            print("achterkant voertuig")
            df.loc[row, "state"] = "back"
        elif state == 7 :
            print("zijkant voertuig")
            df.loc[row, "state"] = "side"            
        elif state == 8 :
            print("voorkant voertuig")
            df.loc[row, "state"] = "front"    
    


def Motor_orientation(row,df):
    model = torch.hub.load('C:/Users/emmah/yolov5', 'custom', path='C:/Users/emmah/Documents/jaar 4/bep/Yolov5_orientation', source='local')
    img = df.iloc[row]["foto_naam"]
    img_data = cv2.imread(img)
    Height_crop, Width_crop, channels = img_data.shape        
    model.conf = orientation_conf
    results = model(img)
    res = results.xyxy[0]
    df1 = pd.DataFrame(res.numpy(), columns = column1)
    df1["x_midden"] = (df1["x_links"] + df1["x_rechts"])/(2*Width_crop)
    df1["y_midden"] = (df1["y_boven"] + df1["y_onder"])/(2*Height_crop)
    for i in range(len(df1['klas1'])):
        if 0.45 < df1['x_midden'][i] < 0.55 and 0.45 < df1["y_midden"][i] < 0.55:
            
            state = int(df1["klas1"][i])

    if state == None or state in state_niet_motor:
        print("leeg ouleh")
        df.loc[row, "state"] = " "
    elif state == 9:
        print("achterkant voertuig")
        df.loc[row, "state"] = "back"
    elif state == 10 :
        print("zijkant voertuig")
        df.loc[row, "state"] = "side"            
    elif state == 11 :
        print("voorkant voertuig")
        df.loc[row, "state"] = "front"
    
def bike_orientation(row,df):
    model = torch.hub.load('C:/Users/emmah/yolov5', 'custom', path='C:/Users/emmah/Documents/jaar 4/bep/Yolov5_orientation', source='local')
    img = df.iloc[row]["foto_naam"]
    img_data = cv2.imread(img)
    Height_crop, Width_crop, channels = img_data.shape        
    model.conf = orientation_conf
    results = model(img)
    res = results.xyxy[0]
    df1 = pd.DataFrame(res.numpy(), columns = column1)
    df1["x_midden"] = (df1["x_links"] + df1["x_rechts"])/(2*Width_crop)
    df1["y_midden"] = (df1["y_boven"] + df1["y_onder"])/(2*Height_crop)
    for i in range(len(df1['klas1'])):
        if 0.45 < df1['x_midden'][i] < 0.55 and 0.45 < df1["y_midden"][i] < 0.55:
            
            state = int(df1["klas1"][i])
    
    
    if state == None or state in state_niet_fiets:
        print("leeg ouleh")
        df.loc[row, "state"] = " "
    elif state == 12:
        print("achterkant voertuig")
        df.loc[row, "state"] = "back"
    elif state == 13 :
        print("zijkant voertuig")
        df.loc[row, "state"] = "side"            
    elif state == 14 :
        print("voorkant voertuig")
        df.loc[row, "state"] = "front"
    
    
def Traffic_sign(row, df):
    bord_crop = df.iloc[row]["foto_naam"]        
    model = keras.models.load_model('C:/Users/emmah/Documents/jaar 4/bep/model.keras')  #juiste plek aangeven!
    
    data =[]
    
    image = cv2.imread(bord_crop)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    data.append(np.array(resize_image))
    
    X = np.array(data)
    X = X/255
    
    pred = np.argmax(model.predict(X), axis=1)
    print(classes[int(pred)])
    df.loc[row, "state"] = classes[int(pred)]    
    
def Traffic_light(row, df):
    device =  "cpu"
    model_lights, preprocess = clip.load("ViT-B/32", device=device)
    
    image_lights = preprocess(Image.open(df.iloc[row]["foto_naam"])).unsqueeze(0).to(device)
    opties = ["Red trafficlight", "Green trafficlight", "Yellow trafficlight"]
    opties_antwoord = ["Red", "Green", "Yellow"]
    text = clip.tokenize(opties).to(device)
    
    
    with torch.no_grad():
        image_features = model_lights.encode_image(image_lights)
        text_features = model_lights.encode_text(text)
    
        logits_per_image, logits_per_text = model_lights(image_lights, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        prediction_lights = opties_antwoord[ np.argmax(probs)]
        print(prediction_lights)
        df.loc[row, "state"] = prediction_lights    
    

dataframe_bouwen(labels, boxes, scores)

# Elke crop maken uit de tabel en foto naam aan tabel toevoegen
for row in range(df.shape[0]):
    crop_and_save_image(row, image_path)
df["foto_naam"] = fotonaam





#State van het object bepalen



classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }



for row in range(df.shape[0]):
    
#load orientation model

    if str(df.iloc[row]["class_naam"]) == "car":
      Car_onrientation(row, df)  
        

    elif str(df.iloc[row]["class_naam"]) == "bus":
        Bus_orientation(row, df)
            
    elif str(df.iloc[row]["class_naam"]) == "truck":
        Truck_orientation(row, df)

    elif str(df.iloc[row]["class_naam"]) == "motorcycle":
        Motor_orientation(row, df)
            
    elif str(df.iloc[row]["class_naam"]) == "bicycle":
        bike_orientation(row, df)
    
    elif str(df.iloc[row]["class_naam"]) == "traffic sign":
        Traffic_sign(row, df)
        
    elif str(df.iloc[row]["class_naam"]) == "traffic light":       
        Traffic_light(row, df)
         
    else:
        print('ik vind geen borden of lichten in deze oeleh')
        df.loc[row, "state"] = " "    

        
#verkeersborden en lichten identificeren




        
        






df.to_csv("C:/Users/emmah/Desktop/dataframe_voor_depth.csv")
             

""""
                            nieuwe "tabel" met naam crop en huidige class


alles P< 0.6 --> clip. 
clip --> update tabel met nieuwe zero shot voorspelling



alles door clip voor de state van het object

states:
    auto: voorkant, achterkant - remmen, niet
    stoplicht: rood, groen, oranje
    fiets: voorkant, zijkant, achterkant
    verkeersbord:

--> toevoegen aan de tabel. 

class, x, y h, w, P, class naam, crop naam. 

"""



