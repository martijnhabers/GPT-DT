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

# PROBABILITY OF BOUNDING BOXES
P = 0.0

# ------------------------------------SEGMENTING TABLES------------------------------------------

# print(df)

def position(df, image_path):
    img = Image.open(os.path.join(os.getcwd(), "images/" + image_path))
    for k in range(0, len(df.index)):
        if df.loc[k, "predictions"] < P:
            df = df.drop(k)

    df = df.reset_index(drop=True)

    # ---------        --------      ---------POSITION----------        ---------         ------------
    # TODO: Make this call a function so that different position methods can be used
#    Image.open(img)
    w, h = img.size
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
    img1 = np.flipud(img)
    plt.imshow(img1)
    plt.show()

    # ---------        --------      --------POSITIONING---------        ---------         ------------

    # df["%s" % hp_name] = np.zeros(len(df.index))
    # df["%s" % wp_name] = np.zeros(len(df.index))
    for b in range(0, len(df.index)):
        
        #if df.loc[b,'%s'%rear_name] == 'front':
        #    continue
        
        PositionPercW = df.loc[b, "x_midden"] / w

        PositionPercH = (h - df.loc[b, "y_midden"]) / h

        # #---------        --------      --------LEFT&RIGHT---------        ---------         ------------

        if PositionPercW < v1:
            Position = "Left"
        elif PositionPercW > v2:
            Position = "Right"
        else:
            Position = "Middle"
        df.loc[b, "width_position"] = Position

        # ---------        --------      ------------DEPTH-----------        ---------         ------------

        if PositionPercH <= h1:
            Position = "Very close"
        elif PositionPercH <= h2:
            Position = "Close"
        else:
            Position = "Far"
        df.loc[b, "height_position"] = Position

        for i in range(0, len(df.index)):
            if df.loc[i, "width_position"] == "Left":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "adjacent to the left"  ####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "close left"  ####
                elif df.loc[i, 'height_position'] == "in the distance":
                    df.loc[i, "position"] = "distanced left"  #####

            if df.loc[i, "width_position"] == "Middle":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "too close straightly infront"  #####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "adjacently straight infront"  #####
                elif df.loc[i, 'height_position'] == "Far":
                    df.loc[i, "position"] = "straight infront at a distant"  #####

            if df.loc[i, "width_position"] == "Right":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "adjacent to the right"  ####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "close right"  ####
                elif df.loc[i, 'height_position'] == "in the distance":
                    df.loc[i, "position"] = "distanced right"  ####
    return (df)

# ---------        --------      ---------DESCRIPTION---------        ---------         ------------


def ChatGPT(df, speed, location, weather):
    CARS = []
    TL = []
    TS = []
    PERSON = []
    BICYCLES = []
    REAR = []
    OTHERS = []
    
    for k in range(0, len(df.index)):
        
        if isinstance(df.loc[k,'state'], str) is False:
            df.loc[k,'state'] = ''
            
            
        if df.loc[k, "view"] == 'rear':
            if df.loc[k, "state"][:5] == 'front':
                REAR.append('A %s %s'%(df.loc[k, 'class_naam'], df.loc[k,'position']))
            
            df = df.drop(k)
    
    df = df.reset_index(drop=True)
    
    
    for a in range(0, len(df.index)):
        if df.loc[a, "class_naam"] == "car":
            if df.loc[a, "state"][:5] == "front":
                CARS.append(
                    "A car approaching from %s" % (df.loc[a, "position"])
                )

            elif df.loc[a, "state"][:4] == "back":
                CARS.append("A car %s" % (df.loc[a, "position"]))

            else:
                CARS.append(
                    "A car %s" % (df.loc[a, "position"])
                )  # SIDE OF THE CAR

        elif df.loc[a, "class_naam"] == "traffic light":
            TL.append(
                "A %s %s"
                % (df.loc[a, "state"], df.loc[a, "class_naam"])
            )

        elif df.loc[a, "class_naam"] == "traffic sign":
            TS.append('A "%s" traffic sign' % (df.loc[a, "state"]))

        elif df.loc[a, "class_naam"] == "person":
            PERSON.append(df.loc[a, "class_naam"])
            PERSON.append(df.loc[a, "state"])

        elif df.loc[a, "class_naam"] == "bicycle":
            if df.loc[a, "state"] == "front":
                BICYCLES.append(
                    "A bicycle approaching from %s" % (df.loc[a, "position"])
                )

            elif df.loc[a, "state"] == "back":
                BICYCLES.append("A bicycle %s" % (df.loc[a, "position"]))

            else:
                BICYCLES.append(
                    "A bicycle %s" % (df.loc[a, "position"])
                )  # SIDE OF THE BICYCLE

        else:
            OTHERS.append(df.loc[a, "class_naam"])
            OTHERS.append(df.loc[a, "state"])

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
        
    if bool(REAR) == False:
        REAR.append("There are no significant objects behind you")


    # --------------------------------------ChatGPT-------------------------------------------------

    # Set up the OpenAI API client
    openai.api_key = openAI_key

    # Set up the model and prompt
    model_engine = "text-davinci-003"
    
    #WEATHER String split for chat gpt
    weather = weather[13:]
    
    #LOCATION String split for chat gpt
    location = location[13:]
    
    prompt1 =  "Assume you are driving in %s. You are driving in %s at %s km/h. The weather condition is %s. " % (country, location, speed, weather)
    prompt2 = f"This is your front view; You see the following cars: {', '.join(CARS)}. You see the following traffic signs: {', '.join(TS)}. You see the following traffic lights: {', '.join(TL)}. You see the following pedestrians: {', '.join(PERSON)}. You see the following bicyclist: {', '.join(BICYCLES)}. Additionally, you see: {', '.join(OTHERS)}. "
    prompt3 = f"This is your rear view: You see the following  cars: {', '.join(REAR)}."
    prompt4 = f"Given the described situation above, what would you do: 'Let go of the gas pedal', 'Brake' or 'Do nothing'. "
    prompt5 = ''#f"Describe the situation in your own words."
    prompt6 = f"Show the three options I gave you and pick your answer. Give your thorough reason behind it."
    prompt = prompt1 +''+ prompt2 +''+ prompt3 +''+ prompt4 + prompt5 + prompt6
    
    # Generate a response ChatGPT
    completion = openai.Completion.create(
        engine=model_engine,
        prompt= prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text

    return (prompt,response)
