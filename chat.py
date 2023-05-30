import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

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
    # for k in range(0, len(df.index)):
    #     if df.loc[k, "predictions"] < P:
    #         df = df.drop(k)

    # df = df.reset_index(drop=True)

    # ---------        --------      ---------POSITION----------        ---------         ------------
    # TODO: Make this call a function so that different position methods can be used
    w, h = img.size

    # # VERTICAL SECTIONS
    # v1 = 0.375
    # v2 = 0.625

    # # HORIZONTAL SECTIONS
    # h1 = 0.2
    # h2 = 0.41

    # BIKELINE SEGMENTS:
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

    # TODO explicit description for rear items

    for i in range(0, len(df.index)):
        if df.loc[i, "view"] == "front":
            if df.loc[i, "width_position"] == "Left":
                df.loc[i, "position"] = (
                    str(df.loc[i, "height_position"])
                    + " meters infront of you and to your left"
                )
            if df.loc[i, "width_position"] == "Middle":
                df.loc[i, "position"] = (
                    str(df.loc[i, "height_position"])
                    + " meters directly infront of you"
                )
            if df.loc[i, "width_position"] == "Right":
                df.loc[i, "position"] = (
                    str(df.loc[i, "height_position"])
                    + " meters infront of you and to your right"
                )
            
            if df.loc[i, "class_naam"] == "car" or df.loc[i, "class_naam"] == "truck":
                if df.loc[i, "state"] == "front":
                    df.loc[i, "position"] = (
                        str(df.loc[i, "height_position"])
                        + " meters"
                    )
                    
        else:
            if df.loc[i, "height_position"] < 10:
                df.loc[i, "position"] = "closely behind you"
            elif df.loc[i, "view"] == "rear":
                df = df.drop(i)

    df = df.reset_index(drop=True)

    # BICYCLE INTO BICYCLIST WITHOUT PERSON RECOGNITION
    for a in range(0, len(df.index)):
        if df.loc[a, "class_naam"] == "bicycle":
            if xb1 < df.loc[a, "x_midden"] < xb2:
                df.loc[a, "class_naam"] = "bicyclist"


    img.close()
    return df


# ---------        --------      ---------DESCRIPTION---------        ---------         ------------


def ChatGPT(df, speed, location, weather, compare=False):
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
        if isinstance(df.loc[k, "state"], str) is False:
            df.loc[k, "state"] = ""

        if df.loc[k, "view"] == "rear":
            if df.loc[k, "state"][:5] == "front":
                REAR.append(
                    "A %s %s" % (df.loc[k, "class_naam"], df.loc[k, "position"])
                )

            df = df.drop(k)

    df = df.reset_index(drop=True)

    for c in range(0, len(df.index)):
        if df.loc[c, "class_naam"] == "bicyclist":
            B.append(df.loc[c, "x_midden"])

        elif df.loc[c, "class_naam"] == "person":
            P.append(df.loc[c, "x_midden"])

    bb = len(B)
    pp = len(P)

    for b in range(0, bb):
        for p in range(0, pp):
            dx = abs(B[b] - P[p])
            if dx <= xrange:
                XB.append(B[b])
                XP.append(P[p])

    # BICYCLE INTO BICYCLIST
    for a in range(0, len(df.index)):
        # if df.loc[a,'class_naam'] == 'bicyclist':
        #     if df.loc[a,'state'] != 'back' or df.loc[a,'state'] != 'front'

        # REMOVE PERSON ON TOP OF BICYCLIST
        if df.loc[a, "class_naam"] == "person":
            for p in range(0, len(XP)):
                if int(df.loc[a, "x_midden"]) == int(XP[p]):
                    df.loc[a, "class_naam"] = "drop"

    for a in range(0, len(df.index)):
        if df.loc[a, "class_naam"] == "drop":
            df = df.drop(a)
    df = df.reset_index(drop=True)

    # CHOPPING DATAFRAME IN ITEMS
    for a in range(0, len(df.index)):
        if df.loc[a, "class_naam"] == "car" or df.loc[a, "class_naam"] == "truck" or df.loc[a, "class_naam"] == "bus" :
            if df.loc[a, "state"][:5] == "front":
                CARS.append("A %s on the adjacent lane approaching from %s" % (df.loc[a, "class_naam"], df.loc[a, "position"]))

            elif df.loc[a, "state"][:4] == "back":
                CARS.append("A %s %s" % (df.loc[a, "class_naam"], df.loc[a, "position"]))

            else:
                CARS.append("A %s %s" % (df.loc[a, "class_naam"],df.loc[a, "position"]))  # SIDE OF THE CAR

        elif df.loc[a, "class_naam"] == "traffic light":
            TL.append("A %s %s" % (df.loc[a, "state"], df.loc[a, "class_naam"]))

            
        elif df.loc[a, "class_naam"] == "traffic sign":
            if df.loc[a,"state"] == 'Back of traffic sign':
                break
            TS.append('A "%s" traffic sign' % (df.loc[a, "state"]))
            
        elif df.loc[a, "class_naam"] == "stop sign":
            TS.append('A "Stop" traffic sign')

        elif df.loc[a, "class_naam"] == "person":
            PERSON.append("A person %s" % df.loc[a, "position"])

        elif df.loc[a, "class_naam"] == "bicyclist":
            if df.loc[a, "state"] == "front":
                BICYCLES.append(
                    "A bicyclist approaching from %s" % (df.loc[a, "position"])
                )

            elif df.loc[a, "state"] == "rear":
                BICYCLES.append("A bicyclist %s" % (df.loc[a, "position"]))

            else:
                BICYCLES.append(
                    "A bicyclist %s" % (df.loc[a, "position"])
                )  # SIDE OF THE BICYCLE

        else:
            OTHERS.append("A %s %s " % (df.loc[a, "class_naam"], df.loc[a, "position"]))

    # --------------------------------------ChatGPT-------------------------------------------------

    # Set up the OpenAI API client
    openai.api_key = openAI_key

    # Set up the model and prompt
    model_engine = "text-davinci-003"

    # WEATHER String split for chat gpt
    weather = weather[13:]

    # LOCATION String split for chat gpt
    location = location[13:]

    if compare == True:
        
#------------------------------------------- NO IF EMPTY ---------------------------------   
        prompt1 =f'''
        Choose to either A) Brake B) Let go of the accelerator or C) Do Nothing based on the given context.
        Context:
            
        """
        Assume you are driving in {country}.
        You are driving in {location} at {speed} km/h. 
        The weather condition is {weather}.
        This is your front view; '''

        prompt2 = ''
        prompt3 = ''
        prompt4 = ''
        prompt5 = ''
        prompt6 = ''
        prompt7 = ''
        prompt8 = ''
        prompt9 = f''' """
        Give your answer in one letter, after which you should provide thorough reasoning.
        Letter: '''

        if bool(CARS) == True:
            prompt2 = f"You see the following cars: {', '.join(CARS)}." 
        if bool(TS) == True:
            prompt3 = f"You see the following traffic signs: {', '.join(TS)}. "
        if bool(TL) == True:
            prompt4 = f"You see the following traffic lights: {', '.join(TL)}. "
        if bool(PERSON) == True:
            prompt5 = f"You see the following pedestrians: {', '.join(PERSON)}. "
        if bool(BICYCLES) == True:
            prompt6 = f"You see the following bicyclist: {', '.join(BICYCLES)}. "
        if bool(OTHERS) == True:
            prompt7 = f"Additionally, you see: {', '.join(OTHERS)}. "
        if bool(REAR) == True:
            prompt8 = f"This is your rear view: You see the following: {', '.join(REAR)}. "

    
            
            
        prompt = prompt1+prompt2+prompt3+prompt4+prompt5+prompt6+prompt7+prompt8+prompt9
    
#---------------------------------------IF EMPTY ----------------------------------------        

    # if bool(CARS) == False:
    #     CARS.append("There are no cars in sight")

    # if bool(TL) == False:
    #     TL.append("There are no traffic lights in sight")

    # if bool(TS) == False:
    #     TS.append("There are no traffic signs in sight")

    # if bool(PERSON) == False:
    #     PERSON.append("There are no pedestrians in sight")

    # if bool(BICYCLES) == False:
    #     BICYCLES.append("There are no bicycles in sight")

    # if bool(OTHERS) == False:
    #     OTHERS.append("there are no more objects than the ones mentioned above")

    # if bool(REAR) == False:
    #     REAR.append("There are no significant objects behind you")
        
    #     prompt = f'''Choose to A) Brake B) Let go of the accelerator or C) Do Nothing based on the given context.
    #     Context:
            
    #     """
    #     Assume you are driving in {country}. You are driving in {location} at {speed} km/h. The weather condition is {weather}.
    #     This is your front view; You see the following cars: {', '.join(CARS)}. You see the following traffic signs: {', '.join(TS)}. You see the following traffic lights: {', '.join(TL)}. You see the following pedestrians: {', '.join(PERSON)}. You see the following bicyclist: {', '.join(BICYCLES)}. Additionally, you see: {', '.join(OTHERS)}.
    #     This is your rear view: You see the following: {', '.join(REAR)}.
    #     """
    #     Give your answer in one letter, after which you should provide thorough reasoning.
        
    #     Letter:'''

    # else:
    #     prompt = f'''
    #     Assume you are driving in {country}. You are driving in {location} at {speed} km/h. The weather condition is {weather}.
    #     """This is your front view; You see the following cars: {', '.join(CARS)}. You see the following traffic signs: {', '.join(TS)}. You see the following traffic lights: {', '.join(TL)}. You see the following pedestrians: {', '.join(PERSON)}. You see the following bicyclist: {', '.join(BICYCLES)}. Additionally, you see: {', '.join(OTHERS)}.
    #     This is your rear view: You see the following: {', '.join(REAR)}.
    #     Given the described situation above, what would you do: "A) Brake", "B) Let go of the gas pedal" or "C) Do nothing". 
    #     Consider the following:
        
    #     A) Brake = drastically reducing speed for urgent danger.
    #         -When you’re driving the maximum allowed speed, you usually should brake if you encounter:
    #             -Weaker road users, like children or pedestrians.
    #             -There is oncoming traffic on narrow roads.
    #             -You’re driving past road work or other obstacles.
    #             -You’re on a chaotic or dangerous intersection.
    #             -You’re in a busy residential area, or near a school.
    #             -You’re nearing a sharp or dangerous turn.
    #             -Large speed differences between you and other road users.
    #             -For yellow and red traffic lights.
        
    #     B) Let go of the gas pedal = reducing some speed.
    #         -When you don’t have a full overview of the situation.
    #         -When there is no danger.
    #         -If the speed limit changes.
        
    #     C) Do Nothing = continue driving your current speed.
    #         -If there is no direct danger.
    #         -If there is a proper amount of distance between you and other road users.
    #     """
        
    #     Show me all possible answers in the following format:
    #         A)...
    #         B)...
    #         C)...
    #     Then, choose one of them. Show me your choice and give a thorough reasoning on why you chose this. Use the following format:
    #         Answer: ...
    #         Reasoning: ...'''

    # Generate a response ChatGPT
    # completion = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=1024,
    #     n=1,
    #     stop=None,
    #     temperature=0.0,
    # )

    # response = completion.choices[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs)

    completion = completion_with_backoff(
        model="gpt-3.5-turbo",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. You are taking the dutch driving exam and wil be presented with what you see around you. Answer as concisely as possible and only take the dutch traffic rules in to consideration.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    response = completion["choices"][0]["message"]["content"].strip()

    return (prompt, response)
