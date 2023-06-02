import numpy as np
import cv2
import pandas as pd
import torch
from tensorflow import keras
from PIL import Image, ImageDraw
import clip
import os
import matplotlib.pyplot as plt

# variabelen
marge = 1.25
orientation_conf = 0.5


column1 = ["x_links", "y_boven", "x_rechts", "y_onder", "prediction", "klas1"]
fotonaam = []
state_niet_auto = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
state_niet_bus = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14]
state_niet_wagen = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]
state_niet_motor = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]
state_niet_fiets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
IMG_HEIGHT = 30
IMG_WIDTH = 30
IMG_HEIGHT_REM = 120
IMG_WIDTH_REM = 120
channels = 3
state = None
####################################


# data framebouwen


def dataframe_bouwen(
    labels,
    boxes,
    scores,
    texts,
    vehicles_detected,
    classes_orientation,
    tri_crop_result,
    image,
):
    columns_df1 = ["xmin", "ymin", "xmax", "ymax", "predictions", "class"]

    if torch.cuda.is_available():
        tri_crop_result[0].boxes = tri_crop_result[0].boxes.cpu()

    df1 = pd.DataFrame(vehicles_detected.numpy(), columns=columns_df1)
    if not df1.empty:
        df1["class_naam"] = df1["class"]
        df1["class_naam"].replace(
            range(int(len(classes_orientation))), classes_orientation, inplace=True
        )
        df1[["class_naam", "state"]] = df1["class_naam"].str.split(
            "_", n=1, expand=True
        )
        for row in range(df1.shape[0]):
            if df1["predictions"][row] < 0.9 and df1["class_naam"][row] not in [
                "car",
                "bicycle",
            ]:
                df1.at[row, "state"] = " "

    columns_df2 = ["xmin", "ymin", "xmax", "ymax"]
    df2 = pd.DataFrame(boxes, columns=columns_df2)
    if not df2.empty:
        df2["predictions"] = scores
        df2["class"] = labels
        df2["class_naam"] = df2["class"]
        df2["class_naam"].replace(range(int(len(texts))), texts, inplace=True)
        df2["state"] = ""

    df = pd.concat([df1, df2], ignore_index=True)
    df["x_midden"] = (df["xmin"] + df["xmax"]) / (2)
    df["y_midden"] = (df["ymin"] + df["ymax"]) / (2)
    df["width"] = df["xmax"] - df["xmin"]
    df["height"] = df["ymax"] - df["ymin"]
    df["width"] = df["width"] * marge
    df["height"] = df["height"] * marge

    df["xmin"] = df["x_midden"] - 0.5 * df["width"]
    df["xmax"] = df["x_midden"] + 0.5 * df["width"]
    df["ymin"] = df["y_midden"] - 0.5 * df["height"]
    df["ymax"] = df["y_midden"] + 0.5 * df["height"]
    # add empty view column for rear and front view
    df["view"] = ""

    # find the index needed to find classes 0 (front view), and 1 (rear view)
    index_front = tri_crop_result[0].boxes.cls.tolist().index(0)
    x_min_front, y_min_front, x_max_front, y_max_front = (
        tri_crop_result[0].boxes.xyxy[index_front].numpy()
    )
    df["x_crop_absoluut_midden"] = df["x_midden"] + x_min_front
    df["y_crop_absoluut_midden"] = df["y_midden"] + y_min_front

    if os.path.exists("tri-crop/predict/crops/rear-view/" + image):
        index_rear = tri_crop_result[0].boxes.cls.tolist().index(1)

        x_min_rear, y_min_rear, x_max_rear, y_max_rear = (
            tri_crop_result[0].boxes.xyxy[index_rear].numpy()
        )

        # adjusted (adj) x min and y min rear view mirror in reference frame of the cropped front view
        x_min_rear_adj = x_min_rear - x_min_front
        y_min_rear_adj = y_min_rear - y_min_front
        x_max_rear_adj = x_max_rear - x_min_front
        y_max_rear_adj = y_max_rear - y_min_front

        for row in range(df.shape[0]):
            if (
                df["y_midden"][row] > y_min_rear_adj
                and df["y_midden"][row] < y_max_rear_adj
                and df["x_midden"][row] > x_min_rear_adj
                and df["x_midden"][row] < x_max_rear_adj
            ):
                df.loc[row, "view"] = "rear"
            else:
                df.loc[row, "view"] = "front"
    else:
        df["view"] = "front"

    return df


def crop_and_save_image(row, df, image_front, fotonaam):
    im2 = cv2.imread(image_front)
    height, width, channels = im2.shape
    #    x, y, w, h = (float(lines[row][1])*width),(float(lines[row][2])*height), (float(lines[row][3])*marge*width), (float(lines[row][4])*marge*height)
    klas = str([df["class_naam"][row]]).strip("[]")
    klas = df.loc[row]["class_naam"]
    x1, y1, x2, y2 = (
        int(df["xmin"][row]),
        int(df["ymin"][row]),
        int(df["xmax"][row]),
        int(df["ymax"][row]),
    )
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
    map_pad = "Crops/"
    bestandsnaam = f"Crop_{klas}_{row}.jpg"
    fotonaam.append(map_pad + bestandsnaam)
    cv2.imwrite(map_pad + bestandsnaam, crop_img)
    return fotonaam


model_traffic_sign = keras.models.load_model(
    "models/model.keras"
)  # juiste plek aangeven!


def Traffic_sign(row, df):
    bord_crop = df.iloc[row]["foto_naam"]

    data = []

    image = cv2.imread(bord_crop)
    image_fromarray = Image.fromarray(image, "RGB")
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    data.append(np.array(resize_image))

    X = np.array(data)
    X = X / 255

    pred = np.argmax(model_traffic_sign.predict(X), axis=1)
    print(classes[int(pred)])
    df.loc[row, "state"] = classes[int(pred)]
    return df


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


def Traffic_light(row, df):
    img = df.iloc[row]["foto_naam"]
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    binary_1 = round_up_to_odd(width / 2)
    binary_2 = -20

    bounding_boxes = [
        (
            width * 0.25,
            height * 0.15,
            width * 0.75,
            height * 0.45,
        ),  # Bounding box coordinates for location 1
        (
            width * 0.25,
            height * 0.45,
            width * 0.75,
            height * 0.65,
        ),  # Bounding box coordinates for location 2
        (
            width * 0.25,
            height * 0.65,
            width * 0.75,
            height * 0.85,
        ),  # Bounding box coordinates for location 3
    ]
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        binary_1,
        binary_2,
    )
    img_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rood = img_show[
        int(bounding_boxes[0][1]) : int(bounding_boxes[0][3]),
        int(bounding_boxes[0][0]) : int(bounding_boxes[0][2]),
    ]
    geel = img_show[
        int(bounding_boxes[1][1]) : int(bounding_boxes[1][3]),
        int(bounding_boxes[1][0]) : int(bounding_boxes[1][2]),
    ]
    groen = img_show[
        int(bounding_boxes[2][1]) : int(bounding_boxes[2][3]),
        int(bounding_boxes[2][0]) : int(bounding_boxes[2][2]),
    ]

    # plt.imshow(img_show)
    # plt.title(rood)
    # plt.show()
    # plt.imshow(rood)
    # plt.title("rood")
    # plt.show()
    # plt.imshow(geel)
    # plt.title("geel")
    # plt.show()
    # plt.imshow(groen)
    # plt.title("groen")
    # plt.show()
    # Count the number of black pixels for red
    _, binary_rood = cv2.threshold(rood, 1, 255, cv2.THRESH_BINARY)
    black_pixels_rood = np.count_nonzero(binary_rood == 0)

    # Count the number of black pixels for yellow
    _, binary_geel = cv2.threshold(geel, 1, 255, cv2.THRESH_BINARY)
    black_pixels_geel = np.count_nonzero(binary_geel == 0)

    # Count the number of black pixels for green
    _, binary_groen = cv2.threshold(groen, 1, 255, cv2.THRESH_BINARY)
    black_pixels_groen = np.count_nonzero(binary_groen == 0)

    if black_pixels_rood > black_pixels_geel and black_pixels_rood > black_pixels_groen:
        # print("licht is rood")
        df.loc[row, "state"] = "Red"
    elif (
        black_pixels_geel > black_pixels_rood and black_pixels_geel > black_pixels_groen
    ):
        # print("licht is geel")
        df.loc[row, "state"] = "Yellow"
    elif (
        black_pixels_groen > black_pixels_geel
        and black_pixels_groen > black_pixels_rood
    ):
        # print("licht is groen")
        df.loc[row, "state"] = "Green"


model = keras.models.load_model("models/model_remv1.keras")

device = "cpu"
model_kind, preprocess = clip.load("ViT-B/32", device=device)


def kind_of_niet(row, df):
    img = df.loc[row, "foto_naam"]
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    # opties = ["a picture of a red trafficlight", "a picture of a yellow traffic light", "a picture of a green trafficlight"]
    opties = ["a child", "an adult"]
    text = clip.tokenize(opties).to(device)

    with torch.no_grad():
        # image_features = model_kind.encode_image(image)
        # text_features = model_kind.encode_text(text)

        logits_per_image, logits_per_text = model_kind(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        prediction_kind = opties[np.argmax(probs)]
    df.loc[row, "class_naam"] = prediction_kind
    return df


def Braking(row, df):
    brake_crop = df.iloc[row]["foto_naam"]

    data = []

    image = cv2.imread(brake_crop)
    image_fromarray = Image.fromarray(image, "RGB")
    resize_image = image_fromarray.resize((IMG_HEIGHT_REM, IMG_WIDTH_REM))
    data.append(np.array(resize_image))

    X = np.array(data)
    X = X / 255

    pred = np.argmax(model.predict(X), axis=1)
    print(classes_rem[int(pred)])
    df.loc[row, "state"] = df.loc[row, "state"] + " " + classes_rem[int(pred)]
    return df


classes_rem = {
    0: "not braking",
    1: "braking",
}


classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
    43: "Back of traffic sign",
    44: "Bicycle lane",
    45: "Pedestrian crossing",
}


# verkeersborden en lichten identificeren

# df.to_csv("C:/Users/emmah/Desktop/dataframe_voor_depth.csv")


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
