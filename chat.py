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

def position(df, image_path, v1, v2):
    img = Image.open(os.path.join(os.getcwd(), "images/" + image_path))
    for k in range(0, len(df.index)):
        if df.loc[k, "predictions"] < P:
            df = df.drop(k)

    df = df.reset_index(drop=True)

    # ---------        --------      ---------POSITION----------        ---------         ------------
    # TODO: Make this call a function so that different position methods can be used
    w, h = img.size

    # # VERTICAL SECTIONS
    # v1 = 0.375
    # v2 = 0.625

    # # HORIZONTAL SECTIONS
    # h1 = 0.2
    # h2 = 0.41
    
    #BIKELINE SEGMENTS:
    xb1 = 0.09 * w
    xb2 = 0.91 * w

    # PLOTTING VERTICAL SECTIONS
    plt.axvline(x=v1 * w, color="r", linestyle="--")
    plt.axvline(x=v2 * w, color="r", linestyle="--")
    plt.axvline(x=xb1 * w, color="b", linestyle="--")
    plt.axvline(x=xb2 * w, color="b", linestyle="--")

    # # PLOTTING HORIZONTAL SECTIONS
    # plt.axhline(y=h1 * h, color="b", linestyle=":")
    # plt.axhline(y=h2 * h, color="b", linestyle=":")

    # GETTING THE IMAGE UPRIGHT
    plt.axis([0, w, 0, h])
    img1 = np.flipud(img)
    plt.imshow(img1)
    plt.show()

    # ---------        --------      --------POSITIONING---------        ---------         ------------

    for b in range(0, len(df.index)):
        
        PositionPercW = df.loc[b, "x_midden"] / w
        # PositionPercH = (h - df.loc[b, "y_midden"]) / h

        # #---------        --------      --------LEFT&RIGHT---------        ---------         ------------

        if PositionPercW < v1:
            Position = "Left"
        elif PositionPercW > v2:
            Position = "Right"
        else:
            Position = "Middle"
        df.loc[b, "width_position"] = Position

        # ---------        --------      ------------DEPTH-----------        ---------         ------------

        # if PositionPercH <= h1:
        #     Position = "Very close"
        # elif PositionPercH <= h2:
        #     Position = "Close"
        # else:
        #     Position = "Far"
            
        # df.loc[b, "height_position"] = Position
        
#TODO explicit description for rear items
        
    for i in range(0, len(df.index)):
        if df.loc[i, "view"] == 'front':
            if df.loc[i, "width_position"] == "Left":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "adjacent to the left"  ####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "close left"  ####
                elif df.loc[i, 'height_position'] == "in the distance":
                    df.loc[i, "position"] = "distanced left"  #####

            if df.loc[i, "width_position"] == "Middle":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "straightly infront and very close"  #####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "straight infront"  #####
                elif df.loc[i, 'height_position'] == "in the distance":
                    df.loc[i, "position"] = "straight infront at a distance"  #####

            if df.loc[i, "width_position"] == "Right":
                if df.loc[i, 'height_position'] == "a few meters away":
                    df.loc[i, "position"] = "adjacent to the right"  ####
                elif df.loc[i, 'height_position'] == "a few tens of meters away":
                    df.loc[i, "position"] = "close right"  ####
                elif df.loc[i, 'height_position'] == "in the distance":
                    df.loc[i, "position"] = "distanced right"  ####
        else:
            if df.loc[i,'height_position'] == ' a few meters away':
                df.loc[i, "position"] = 'closely behind you'
            else:
                df.loc[i, "position"] = 'farfar'
                
            
        if df.loc[i, "position"] == 'farfar':
            df = df.drop(i)
                    
    df = df.reset_index(drop=True)
                
           
                    
    #BICYCLE INTO BICYCLIST WITHOUT PERSON RECOGNITION
    for a in range(0,len(df.index)):
        if df.loc[a,'class_naam'] == 'bicycle':
            if xb1 < df.loc[a,'x_midden'] < xb2:
                df.loc[a,'class_naam'] = 'bicyclist'
            else:
                df.loc[a,'class_naam'] = 'parked bicycle'
    return (df)

# ---------        --------      ---------DESCRIPTION---------        ---------         ------------


def ChatGPT(df, speed, location, weather, compare = False):
    CARS = []
    TL = []
    TS = []
    PERSON = []
    BICYCLES = []
    REAR = []
    OTHERS = []
    
    B = []
    P = []
    XB = []
    XP = []
    
    xrange = 15
    
    for k in range(0, len(df.index)):
        
        if isinstance(df.loc[k,'state'], str) is False:
            df.loc[k,'state'] = ''
            
            
        if df.loc[k, "view"] == 'rear':
            if df.loc[k, "state"][:5] == 'front':
                REAR.append('A %s %s'%(df.loc[k, 'class_naam'], df.loc[k,'position']))
            
            df = df.drop(k)
    
    df = df.reset_index(drop=True)
    
    for c in range(0, len(df.index)):
        if df.loc[c,'class_naam'] == 'bicyclist':
            B.append(df.loc[c,'x_midden'])
            
        elif df.loc[c,'class_naam'] == 'person':
            P.append(df.loc[c,'x_midden'])
            
        
    bb = len(B)
    pp = len(P)
    
    for b in range(0, bb):
        for p in range(0, pp):
            dx = abs(B[b] - P[p])
            if dx <= xrange:
                XB.append(B[b])
                XP.append(P[p])
               
    #BICYCLE INTO BICYCLIST
    for a in range(0,len(df.index)):
        # if df.loc[a,'class_naam'] == 'bicyclist':
        #     if df.loc[a,'state'] != 'back' or df.loc[a,'state'] != 'front'
                
                
                    
    #REMOVE PERSON ON TOP OF BICYCLIST        
        if df.loc[a,'class_naam'] == 'person':
            for p in range(0,len(XP)):
                if int(df.loc[a,'x_midden']) == int(XP[p]):
                    df.loc[a,'class_naam'] = 'drop'
        
                    
    for a in range(0, len(df.index)):
        if df.loc[a,'class_naam'] == 'drop':
            df = df.drop(a)
    df = df.reset_index(drop=True)
        
    #CHOPPING DATAFRAME IN ITEMS
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
            PERSON.append("A person %s"%df.loc[a, "position"])

        elif df.loc[a, 'class_naam'] == "bicyclist":
            if df.loc[a, "state"] == "front":
                BICYCLES.append(
                    "A bicyclist approaching from %s" % (df.loc[a,'position'])
                )

            elif df.loc[a, 'state'] == "rear":
                BICYCLES.append("A bicyclist %s" % (df.loc[a, 'position']))

            else:
                BICYCLES.append(
                    "A bicyclist %s" % (df.loc[a, 'position'])
                )  # SIDE OF THE BICYCLE

        else:
            if df.loc[a, 'position'] == 'back':
                OTHERS.append("A %s "%df.loc[a, "class_naam"])

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
    
    if compare == True:
        prompt1 =  "Assume you are driving in %s. You are driving in %s at %s km/h. The weather condition is %s. " % (country, location, speed, weather)
        prompt2 = f"This is your front view; You see the following cars: {', '.join(CARS)}. You see the following traffic signs: {', '.join(TS)}. You see the following traffic lights: {', '.join(TL)}. You see the following pedestrians: {', '.join(PERSON)}. You see the following bicyclist: {', '.join(BICYCLES)}. Additionally, you see: {', '.join(OTHERS)}. "
        prompt3 = f"This is your rear view: You see the following: {', '.join(REAR)}. "
        prompt4 = f"Given the described situation above, what would you do: 'A) Let go of the gas pedal', 'B) Brake' or 'C) Do nothing'. "
        prompt5 = f"Choose one of the 3 options I gave you. Show me just your answer."
        prompt = prompt1 +''+ prompt2 +''+ prompt3 +''+ prompt4 + prompt5
        
    else:
        prompt1 =  "Assume you are driving in %s. You are driving in %s at %s km/h. The weather condition is %s. " % (country, location, speed, weather)
        prompt2 = f"This is your front view; You see the following cars: {', '.join(CARS)}. You see the following traffic signs: {', '.join(TS)}. You see the following traffic lights: {', '.join(TL)}. You see the following pedestrians: {', '.join(PERSON)}. You see the following bicyclist: {', '.join(BICYCLES)}. Additionally, you see: {', '.join(OTHERS)}. "
        prompt3 = f"This is your rear view: You see the following: {', '.join(REAR)}. "
        prompt4 = f"These are your possible answers: 'A) Let go of the gas pedal', 'B) Brake' or 'C) Do nothing'. "
        prompt5 = f"Show me all possible answers as a list. Then, choose one of them. Show me your choice and give a thorough reasoning on why you chose this "
        prompt = prompt1 +''+ prompt2 +''+ prompt3 +''+ prompt4 + prompt5
        
       
    
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
    if compare == True:
        prompt = " "

    return (prompt, response)
