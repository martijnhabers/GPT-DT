from OCR import *
from YoloSplit import *
from owlvit import *

# path to file
image = "images/01.jpg"


### yolo_tri_crop(image) ###
# splits image into 3 parts, outside-view, rear-view, and speed

### easyocr_detect(image) ###
# detects number in file, specified by its path

### object_detect_owlvit(text_weighted, image) ###
# does a zero shot object detection on


yolo_tri_crop(image)

print(easyocr_detect(image))


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

boxes, labels, scores = object_detect_owlvit(text_weighted, image)
