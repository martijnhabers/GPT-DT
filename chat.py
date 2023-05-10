import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import openai

# -----------------------------------KEYS AND LOCATIONS-------------------------------------------------

# img = "C:\\Users\Gebruiker\Documents\BEP\\vraag x.jpg"
openAI_key = "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"

# ---------        --------      ------VARIABLES------        ---------         ------------

# COLUMN NAMES THAT MAY VARY
country = "The Netherlands"

# TABLE WITH COLUMN NAMES
classnum_name = "class"
xmid_name = "x_midden"
ymid_name = "y_midden"
b_name = "breedte"
h_name = "hoogte"
pred_name = "predictions"
class_name = "class_naam"
state_name = "state"
foto_name = "foto_naam"
hp_name = "height_position"
wp_name = "width_position"
pos_name = "position"


# PROBABILITY OF BOUNDING BOXES
P = 0.4

# ------------------------------------SEGMENTING TABLES------------------------------------------


def position(df, image_path):
    image = Image.open(os.path.join(os.getcwd(), "images/" + image_path))

    for k in range(0, len(df.index)):
        if df.loc[k, "%s" % (pred_name)] < P:
            df = df.drop(k)

    df = df.reset_index(drop=True)

    # ---------        --------      ---------POSITION----------        ---------         ------------
    # TODO: Make this call a function so that different position methods can be used

    w, h = image.size
    # print(w)

    # VERTICAL SECTIONS
    v1 = 0.45
    v2 = 0.675

    # HORIZONTAL SECTIONS
    h1 = 0.2
    h2 = 0.41

    # PLOTTING VERTICAL SECTIONS
    plt.axvline(x=v1 * w, color="r", linestyle="--")
    plt.axvline(x=v2 * w, color="r", linestyle="--")

    # PLOTTING HORIZONTAL SECTIONS
    plt.axhline(y=h1 * h, color="b", linestyle=":")
    plt.axhline(y=h2 * h, color="b", linestyle=":")

    # GETTING THE IMAGE UPRIGHT
    plt.axis([0, w, 0, h])
    image = np.flipud(image)
    plt.imshow(image)
    plt.show()

    # ---------        --------      --------POSITIONING---------        ---------         ------------

    df["%s" % hp_name] = np.zeros(len(df.index))
    df["%s" % wp_name] = np.zeros(len(df.index))
    for b in range(0, len(df.index)):
        PositionPercW = df.loc[b, "%s" % (xmid_name)] / w

        PositionPercH = df.loc[b, "%s" % (ymid_name)] / h

        # #---------        --------      --------LEFT&RIGHT---------        ---------         ------------

        if PositionPercW < v1:
            Position = "Left"
        elif PositionPercW > v2:
            Position = "Right"
        else:
            Position = "Middle"
        df.loc[b, "%s" % wp_name] = Position

        # ---------        --------      ------------DEPTH-----------        ---------         ------------

        if PositionPercH <= h1:
            Position = "Very close"
        elif PositionPercH <= h2:
            Position = "Close"
        else:
            Position = "Far"
        df.loc[b, "%s" % hp_name] = Position

        for i in range(0, len(df.index)):
            if df.loc[i, "%s" % wp_name] == "Left":
                if df.loc[i, "%s" % hp_name] == "Very close":
                    df.loc[i, "%s" % pos_name] = "adjacent to the left"  ####
                elif df.loc[i, "%s" % hp_name] == "Close":
                    df.loc[i, "%s" % pos_name] = "close left"  ####
                elif df.loc[i, "%s" % hp_name] == "Far":
                    df.loc[i, "%s" % pos_name] = "distanced left"  #####

            if df.loc[i, "%s" % wp_name] == "Middle":
                if df.loc[i, "%s" % hp_name] == "Very close":
                    df.loc[i, "%s" % pos_name] = "too close straightly infront"  #####
                elif df.loc[i, "%s" % hp_name] == "Close":
                    df.loc[i, "%s" % pos_name] = "adjacently straight infront"  #####
                elif df.loc[i, "%s" % hp_name] == "Far":
                    df.loc[i, "%s" % pos_name] = "straight infront at a distant"  #####

            if df.loc[i, "%s" % wp_name] == "Right":
                if df.loc[i, "%s" % hp_name] == "Very close":
                    df.loc[i, "%s" % pos_name] = "adjacent to the right"  ####
                elif df.loc[i, "%s" % hp_name] == "Close":
                    df.loc[i, "%s" % pos_name] = "close right"  ####
                elif df.loc[i, "%s" % hp_name] == "Far":
                    df.loc[i, "%s" % pos_name] = "distanced right"  ####
    return df


# ---------        --------      ---------DESCRIPTION---------        ---------         ------------


def ChatGPT(df, speed, location):
    CARS = []
    TL = []
    TS = []
    PERSON = []
    BICYCLES = []
    OTHERS = []

    for a in range(0, len(df.index)):
        if df.loc[a, "%s" % (class_name)] == "car":
            if df.loc[a, "%s" % (state_name)] == "front":
                CARS.append(
                    "A car approaching from %s" % (df.loc[a, "%s" % (pos_name)])
                )

            elif df.loc[a, "%s" % (state_name)] == "rear":
                CARS.append("A car %s" % (df.loc[a, "%s" % (pos_name)]))

            else:
                CARS.append(
                    "A car %s" % (df.loc[a, "%s" % (pos_name)])
                )  # SIDE OF THE CAR

        elif df.loc[a, "%s" % (class_name)] == "traffic light":
            # CARS.append(df.loc[a,'0'])
            TL.append(
                "A %s %s"
                % (df.loc[a, "%s" % (state_name)], df.loc[a, "%s" % (class_name)])
            )

        elif df.loc[a, "%s" % (class_name)] == "traffic sign":
            # CARS.append(df.loc[a,'0'])
            TS.append('A "%s" traffic sign' % (df.loc[a, "%s" % (state_name)]))

        elif df.loc[a, "%s" % (class_name)] == "person":
            # CARS.append(df.loc[a,'0'])
            PERSON.append(df.loc[a, "%s" % (class_name)])
            PERSON.append(df.loc[a, "%s" % (state_name)])

        elif df.loc[a, "%s" % (class_name)] == "bicycle":
            if df.loc[a, "%s" % (state_name)] == "front":
                BICYCLES.append(
                    "A bicycle approaching from %s" % (df.loc[a, "%s" % (pos_name)])
                )

            elif df.loc[a, "%s" % (state_name)] == "rear":
                BICYCLES.append("A bicycle %s" % (df.loc[a, "%s" % (pos_name)]))

            else:
                BICYCLES.append(
                    "A bicycle %s" % (df.loc[a, "%s" % (pos_name)])
                )  # SIDE OF THE CAR

        else:
            # CARS.append(df.loc[a,'0'])
            OTHERS.append(df.loc[a, "%s" % (class_name)])
            OTHERS.append(df.loc[a, "%s" % (state_name)])

    # IF empty
    if bool(CARS) == False:
        CARS.append("There are no cars in sight")

    if bool(TL) == False:
        TL.append("There are no traffic lights in sight")

    if bool(TS) == False:
        TS.append("There are no traffic signs in sight")

    if bool(PERSON) == False:
        PERSON.append("There are no pedestrians in sight")

    if bool(BICYCLES) == False:
        BICYCLES.append("There are no bicycles in sight")

    if bool(OTHERS) == False:
        OTHERS.append("there are no more objects than the ones mentioned above")

    # --------------------------------------ChatGPT-------------------------------------------------

    # Set up the OpenAI API client
    openai.api_key = openAI_key

    # Set up the model and prompt
    model_engine = "text-davinci-003"
    prompt = (
        "Assume you are driving in %s. You are driving in a %s area at %d km/h. You see the following cars: %s. You see the following traffic signs: %s. You see the following traffic lights: %s. You see the following pedestrians: %s. You see the following bicyclist: %s. Additionally, you see: %s. Generate a multiple choice question with the following answer choices: 'Let go of the gas pedal', 'Brake' or 'Do nothing'. After showing the question and answers, pick your answer. Give your thorough reason behind it."
        % (country, location, speed, CARS, TS, TL, PERSON, BICYCLES, OTHERS)
    )

    # Generate a response ChatGPT
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text

    return (prompt, response)
