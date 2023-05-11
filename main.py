from OCR import *
from YoloSplit import *
from owlvit import *
from CLIPstate import *
from state_detection import *
from breaking_state_function import *
from vehicle_detection import *
from chat import *
from create_depth_map import *
from depth_estimation import *


import shutil
import os

 
    #TODO: Breaking state toevoegen?
    #TODO: matrix borden detectie/ uitlezen toevoegen
    #TODO: weg deel toevoegen --> waar de weg is/ hoe die loopt
    



# Remove leftover images from previous run of code.
if os.path.exists("tri-crop"):
    shutil.rmtree("tri-crop")


dir = os.getcwd()

for f in os.listdir(dir + "/Crops"):
    os.remove(os.path.join(dir + "/Crops", f))

# Set name of image file to analyse
image = "vraag 19.jpg"



text_weighted = [
    ["a photo of a person", 0.25],
    ["a photo of a train", 0.4],
    ["a photo of a boat", 0.4],
    ["a photo of a traffic light", 0.45],
    ["a photo of a stop sign", 0.4],
    ["a photo of a cat", 0.4],
    ["a photo of a dog", 0.4],
    ["a photo of a horse", 0.4],
    ["a photo of a sheep", 0.4],
    ["a photo of a cow", 0.4],
    ["a photo of a traffic cone", 0.4],
    ["a photo of a traffic sign", 0.35],
    ["a photo of a ball", 0.4],
    ["a photo of a tractor", 0.4],
    ["a photo of a digital traffic sign", 0.3],
    ["a photo of a digital traffic sign", 0.4],
]

weather_list = [
    "a picture of snowy",
    "a picture of sunny",
    "a picture of rainy",
    "a picture of overcast",
    "a picture of snowy weather",
    "a picture of sunny weather",
    "a picture of rainy weather",
    "a picture of overcast weather",
]

location_list = [
    "a picture of a highway",
    "a picture of a country road",
    "a picture of a motorway",
    "a picture of a city",
    "a picture of a residential area",
]

# classes is a list of all the classes shown above

classes_orientation = [
    "car_back",
    "car_side",
    "car_front",
    "bus_back",
    "bus_side",
    "bus_front",
    "truck_back",
    "truck_side",
    "truck_front",
    "motorcycle_back",
    "motorcycle_side",
    "motorcycle_front",
    "bicycle_back",
    "bicycle_side",
    "bicycle_front",
]

classes_owl = [x[0][13:] for x in text_weighted]


# splits image into 3 parts, outside-view, rear-view, and speed
# saves to tri-crop/predict/crops/outside-view
# saves to tri-crop/predict/crops/rear-view
# saves to tri-crop/predict/crops/speed
tri_crop_results = yolo_tri_crop("images/" + image)

# detects number with OCR in file, specified by its path
car_speed = easyocr_detect(os.path.join(dir, "tri-crop/predict/crops/speed/" + image))


# does a zero shot object detection on an image and returns boxes, labels, and scores
owl_boxes, owl_labels, owl_scores = owlvit_object_detect(
    text_weighted,
    os.path.join(dir, "tri-crop/predict/crops/outside-view/" + image),
)

weather, location = CLIP_state_detect(
    os.path.join(dir, "tri-crop/predict/crops/outside-view/" + image),
    weather_list,
    location_list,
)

# detecteerd de voertuigen
# detecteerd de voertuigen
image_front = "tri-crop/predict/crops/outside-view/" + image

vehicles_detected = vehicle_detection(image_front)

# maakt het dataframe
df = dataframe_bouwen(
    owl_labels,
    owl_boxes,
    owl_scores,
    classes_owl,
    vehicles_detected,
    classes_orientation,
    tri_crop_results,
    image,
)

# Elke crop maken uit de tabel en foto naam aan tabel toevoegen
for row in range(df.shape[0]):
    crop_and_save_image(row, df, image_front)
df["foto_naam"] = fotonaam

 
# bepaald de state een verkeersbord of verkeerslicht

for row in range(df.shape[0]):
    if str(df.iloc[row]["class_naam"]) == "traffic sign":
        Traffic_sign(row, df)
        
    elif str(df.iloc[row]["class_naam"]) == "traffic light":       
        Traffic_light(row, df)
        
    elif str(df.iloc[row]["state"]) == "back" and str(df.iloc[row]['class_naam']) == 'car':
        Braking(row,df)

    # change extention from jpg to png for depth estimation
filename, extension = os.path.splitext(image)
image_depth = filename + ".png"

    
if os.path.exists("/Depth_map_images/" + image_depth):
    df = depth_estimation(df, image_depth)
    
else:
    image_depth = create_depth_map(image)
    
    df = depth_estimation(df, image_depth)



df = position(df, image)

prompt, response = ChatGPT(df, car_speed, location, weather)

print(prompt)
print(response)

text_file = open("Output.txt", "w")
text_file.write(prompt)
text_file.write("")
text_file.write(response)
text_file.close()
# df.to_csv("C:/Users/Mees/Desktop/vraag 5.csv")