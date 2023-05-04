from OCR import *
from YoloSplit import *
from owlvit import *
from CLIPstate import *
import shutil

# Remove leftover images from previous run of code.
if os.path.exists("tri-crop"):
    shutil.rmtree("tri-crop")

# Set name of image file to analyse
image = "01.jpg"
dir = os.getcwd()


text_weighted = [
    ["a photo of a person", 0.25],
    ["a photo of a car", 0.3],
    ["a photo of a bicycle", 0.25],
    ["a photo of a motorbike", 0.4],
    ["a photo of a bus", 0.4],
    ["a photo of a train", 0.4],
    ["a photo of a truck", 0.4],
    ["a photo of a boat", 0.4],
    ["a photo of a traffic light", 0.4],
    ["a photo of a stop sign", 0.4],
    ["a photo of a cat", 0.4],
    ["a photo of a dog", 0.4],
    ["a photo of a horse", 0.4],
    ["a photo of a sheep", 0.4],
    ["a photo of a cow", 0.4],
    ["a photo of a traffic cone", 0.4],
    ["a photo of a traffic sign", 0.2],
    ["a photo of a ball", 0.4],
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
classes = [x[0][13:] for x in text_weighted]


# splits image into 3 parts, outside-view, rear-view, and speed
# saves to tri-crop/predict/crops/outside-view
# saves to tri-crop/predict/crops/rear-view
# saves to tri-crop/predict/crops/speed
yolo_tri_crop("images/" + image)

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
