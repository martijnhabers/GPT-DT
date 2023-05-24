from transformers import CLIPProcessor, CLIPModel
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def CLIP_state_detect(image, weather, location, showImg=False):
    context = weather + location

    image = Image.open(image)

    if showImg == True:
        plt.imshow(image)
        plt.show()

    inputs = processor(text=context, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)
    # we can take the softmax to get the label probabilities
    probs = probs.detach().numpy()  # convert results to numpy array
    probs = np.float64(probs)
    # convert results to float64 to prevent floating point errors

    probs_weather = probs[0][0 : len(weather)]
    weather_norm = (1.0 / np.sum(probs_weather)) * probs_weather

    probs_location = probs[0][-len(location) :]
    location_norm = (1.0 / np.sum(probs_location)) * probs_location

    max_weather = np.argmax(probs_weather)
    max_location = np.argmax(probs_location) + len(probs_weather)

    image.close()

    return context[max_weather], context[max_location]
