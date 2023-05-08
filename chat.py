
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import openai


#-----------------------------------KEYS AND LOCATIONS-------------------------------------------------


loc_sheet = "C:\\Users\Gebruiker\Documents\BEP\dataframe_voor_depth_x.csv"#"C:\\Users\Gebruiker\Documents\BEP\dataframe_voor_ivan.csv"
img = "C:\\Users\Gebruiker\Documents\BEP\\vraag x.jpg"
openAI_key = 0


#---------------------------------------PRIMING---------------------------------------------


# country = "Netherlands"
# type = "traffic signs"

df = pd.read_csv(r'%s' %(loc_sheet), index_col = 0) #df = pd.read_excel(r'...') 
image = Image.open(img)
plt.imshow(image)

#---------        --------      ------VARIABLES------        ---------         ------------

#COLUMN NAMES THAT MAY VARY
country = 'The Netherlands'
location = 'country road'
speed = 60

#TABLE
classnum_name = 'class'
xmid_name = 'x_midden'
ymid_name = 'y_midden'
b_name = 'breedte'
h_name = 'hoogte'
pred_name = 'prediction'
class_name = 'class_naam'
state_name = 'state'
foto_name = 'foto_naam'
hp_name = 'height_position'
wp_name = 'width_position'
pos_name = 'position'


#PROBABILITY OF BOUNDING BOXES
P = 0.4



#------------------------------------SEGMENTING TABLES------------------------------------------

def position(df,image):
        
    for k in range(0, len(df.index)):
        if df.loc[k,'%s'%(pred_name)] < P:
            df = df.drop(k)
    
    df = df.reset_index(drop=True)       
    
#---------        --------      ---------POSITION----------        ---------         ------------
    
    with open("Position.py") as f:
        exec(f.read())

    return df

position(df,image)
#---------        --------      ---------DESCRIPTION---------        ---------         ------------

print(df)

CARS = []
TL = []
TS = []
PERSON = []
BICYCLES = []
OTHERS = []

for a in range(0,len(df.index)):

    if df.loc[a,'%s'%(class_name)] == 'car':
        
        if df.loc[a,'%s'%(state_name)] == 'front':
            CARS.append('A car approaching from %s'%(df.loc[a,'%s'%(pos_name)]))
            
        elif df.loc[a,'%s'%(state_name)] == 'rear':
            CARS.append('A car %s'%(df.loc[a,'%s'%(pos_name)]))
            
        else:
            CARS.append('A car %s'%(df.loc[a,'%s'%(pos_name)])) #SIDE OF THE CAR
            
    
    elif df.loc[a,'%s'%(class_name)] == 'traffic light':
        # CARS.append(df.loc[a,'0'])
        TL.append('A %s %s' %(df.loc[a,'%s'%(state_name)], df.loc[a,'%s'%(class_name)]))
   
    
    elif df.loc[a,'%s'%(class_name)] == 'traffic sign':
        # CARS.append(df.loc[a,'0'])
        TS.append('A "%s" traffic sign'%(df.loc[a,'%s'%(state_name)]))
        
    elif df.loc[a,'%s'%(class_name)] == 'person':
        # CARS.append(df.loc[a,'0'])
        PERSON.append(df.loc[a,'%s'%(class_name)])
        PERSON.append(df.loc[a,'%s'%(state_name)])
        
    elif df.loc[a,'%s'%(class_name)] == 'bicycle':
        
        if df.loc[a,'%s'%(state_name)] == 'front':
            BICYCLES.append('A bicycle approaching from %s'%(df.loc[a,'%s'%(pos_name)]))
            
        elif df.loc[a,'%s'%(state_name)] == 'rear':
            BICYCLES.append('A bicycle %s'%(df.loc[a,'%s'%(pos_name)]))
            
        else:
            BICYCLES.append('A bicycle %s'%(df.loc[a,'%s'%(pos_name)])) #SIDE OF THE CAR
             
    else:
        # CARS.append(df.loc[a,'0'])
        OTHERS.append(df.loc[a,'%s'%(class_name)])
        OTHERS.append(df.loc[a,'%s'%(state_name)])
        
        
#IF empty
if bool(CARS) == False:
    CARS.append('There are no cars in sight')
    
if bool(TL) == False:
    TL.append('There are no traffic lights in sight')
    
if bool(TS) == False:
    TS.append('There are no traffic signs in sight')
    
if bool(PERSON) == False:
    PERSON.append('There are no pedestrians in sight')
    
if bool(BICYCLES) == False:
    BICYCLES.append('There are no bicycles in sight')

if bool(OTHERS) == False:
    OTHERS.append('there are no more objects than the ones mentioned above')



#--------------------------------------ChatGPT-------------------------------------------------


# Set up the OpenAI API client
openai.api_key = openAI_key

# Set up the model and prompt
model_engine = "text-davinci-003"

prompt = "Assume you are driving in %s. You are driving in a %s area at %d km/h. You see the following cars: %s. You see the following traffic signs: %s. You see the following traffic lights: %s. You see the following pedestrians: %s. You see the following bicyclist: %s. Additionally, you see: %s. Generate a multiple choice question with the following answer choices: 'Let go of the gas pedal', 'Brake' or 'Do nothing'. After showing the question and answers, pick your answer. Give your thorough reason behind it."%(country, location, speed, CARS, TS, TL, PERSON, BICYCLES, OTHERS)
print(prompt)


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
print(response)
