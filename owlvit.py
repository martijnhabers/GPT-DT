import os
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def owlvit_object_detect(text_weighted, image):
    img = Image.open(image)
    texts = [x[0] for x in text_weighted]
    inputs = processor(text=texts, images=img, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([img.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(
        threshold=0, outputs=outputs, target_sizes=target_sizes
    )

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = (
        results[i]["boxes"],
        results[i]["scores"],
        results[i]["labels"],
    )

    boxes = boxes.tolist()
    labels = labels.tolist()
    scores = scores.tolist()

    boxes_new = []
    scores_new = []
    labels_new = []

    for i, label in enumerate(labels):
        weight = text_weighted[label][1]
        if scores[i] > weight:
            boxes_new.append(boxes[i])
            scores_new.append(scores[i])
            labels_new.append(labels[i])

    boxes = boxes_new
    scores = scores_new
    labels = labels_new

    return boxes, labels, scores
