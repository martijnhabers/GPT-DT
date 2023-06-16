# GPT-DT

Using GPT for Driving Test

This project aims to solve a driving test by using several different machine learning models that extract contextual information from a Dutch driving test.



## How to use

To use the code, there are some prerequisites

    pip install requirements-colab.txt
    
Next, there are [some models](https://drive.google.com/drive/folders/1N7cgvYQfyR6eU-bgWNLKyDyYf6MSPRiu?usp=sharing) that need to be put into the /models folder.


Next, place the images you want to analyze inside the /images folder.


Insert your API key for OpenAI inside the chat.py file.


Create a folder with the name of "Crops" in the working directory


Running main.py will run through all the images in the /images folder. The results are then saved in the /results folder. The results are given in the following format:

* crops
* - folder including crops detected from all the found objects in the image
* df
* - folder including the completed dataframe including all information from running the model
* texts
* - Folder containing all the prompts and responses from each question
* tri-crop
* - The split
* confusion.png
* - A confusion matrix based on all the results of the model
* results.csv
* - Csv file containing the results from all the images
