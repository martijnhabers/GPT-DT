from OCR import *
from YoloSplit import *
from owlvit import *
from CLIPstate import *
from state_detection import *
from breaking_state_function import *
from vehicle_detection import *
from chat import *
from create_depth_map import *
from depth_estimation import *

from sklearn import metrics
from datetime import datetime
import shutil
import os
import time
import seaborn as sns


# TODO: Breaking state toevoegen?
# TODO: matrix borden detectie/ uitlezen toevoegen
# TODO: weg deel toevoegen --> waar de weg is/ hoe die loopt


dir = os.getcwd()

# Run once to initialize run folder to save data to
results_dir = datetime.now().strftime("%Y-%m-%d[%H.%M.%S]")
results_dir = os.path.join("results", results_dir)

subdirs = ["crops", "tri-crop", "df", "texts"]

for name in subdirs:
    subdirectory_path = os.path.join(results_dir, name)
    os.makedirs(subdirectory_path)


def run_program(image):
    # Remove leftover images from previous run of code.
    if os.path.exists("tri-crop"):
        shutil.rmtree("tri-crop")

    for f in os.listdir(dir + "/Crops"):
        os.remove(os.path.join(dir + "/Crops", f))

    text_weighted = [
        ["a photo of a person", 0.25],
        ["a photo of a train", 0.4],
        ["a photo of a railroad crossing", 0.4],
        ["a photo of a boat", 0.4],
        ["a photo of a traffic light", 0.45],
        ["a photo of a stop sign", 0.4],
        ["a photo of a animal", 0.4],
        ["a photo of a traffic cone", 0.4],
        ["a photo of a traffic sign", 0.35],
        ["a photo of a ball", 0.4],
        ["a photo of a farm vehicle", 0.5],
        # ["a photo of a variable speed sign", 0.15],
        ["a photo of a digital traffic sign", 0.4],
    ]

    weather_list = [
        "a picture of snowy weather",
        "a picture of foggy weather",
        "a picture of sunny weather",
        "a picture of rainy weather",
        "a picture of overcast weather",
    ]

    location_list = [
        "a picture of a highway",
        "a picture of a provincial road",
        "a picture of a country road",
        "a picture of a county road",
        "a picture of a urban road",
        "a picture of a residential road",
    ]

    # classes is a list of all the classes shown above

    classes_orientation = [
        "car_back",
        "car_side",
        "car_front",
        "bus_back",
        "bus_side",
        "bus_front",
        "truck_back",
        "truck_side",
        "truck_front",
        "motorcycle_back",
        "motorcycle_side",
        "motorcycle_front",
        "bicycle_back",
        "bicycle_side",
        "bicycle_front",
    ]

    classes_owl = [x[0][13:] for x in text_weighted]

    # splits image into 3 parts, outside-view, rear-view, and speed
    # saves to tri-crop/predict/crops/outside-view
    # saves to tri-crop/predict/crops/rear-view
    # saves to tri-crop/predict/crops/speed
    tri_crop_results = yolo_tri_crop("images/" + image)

    filename, extension = os.path.splitext(image)
    image = filename + ".jpg"
    # detects number with OCR in file, specified by its path
    car_speed = easyocr_detect(
        os.path.join(dir, "tri-crop/predict/crops/speed/" + image)
    )

    # does a zero shot object detection on an image and returns boxes, labels, and scores
    owl_boxes, owl_labels, owl_scores = owlvit_object_detect(
        text_weighted,
        os.path.join(dir, "tri-crop/predict/crops/outside-view/" + image),
    )

    # owl_boxes, owl_labels, owl_scores = [[], [], []]

    weather, location = CLIP_state_detect(
        os.path.join(dir, "tri-crop/predict/crops/outside-view/" + image),
        weather_list,
        location_list,
    )

    # detecteerd de voertuigen
    image_front = "tri-crop/predict/crops/outside-view/" + image
    vehicles_detected = vehicle_detection(image_front)
    # vehicles_detected = torch.empty(0, 6)

    # maakt het dataframe
    df = dataframe_bouwen(
        owl_labels,
        owl_boxes,
        owl_scores,
        classes_owl,
        vehicles_detected,
        classes_orientation,
        tri_crop_results,
        image,
    )

    # Elke crop maken uit de tabel en foto naam aan tabel toevoegen
    fotonaam = []
    for row in range(df.shape[0]):
        fotonaam = crop_and_save_image(row, df, image_front, fotonaam)
    df["foto_naam"] = fotonaam

    # bepaald de state een verkeersbord of verkeerslicht

    for row in range(df.shape[0]):
        if str(df.iloc[row]["class_naam"]) == "traffic sign":
            Traffic_sign(row, df)

        elif str(df.iloc[row]["class_naam"]) == "traffic light":
            Traffic_light(row, df)

        elif str(df.loc[row, "class_naam"]) == "person":
            kind_of_niet(row, df)

        elif (
            str(df.iloc[row]["state"]) == "back"
            and str(df.iloc[row]["class_naam"]) == "car"
        ):
            Braking(row, df)

        # change extention from jpg to png for depth estimation
    depth_df_file = filename + ".csv"
    image_input = filename + ".jpeg"

    if os.path.exists("Depth_map_csv/" + depth_df_file):
        depth_df = pd.read_csv("Depth_map_csv/" + depth_df_file)
        df = depth_estimation(df, depth_df)

    else:
        depth_df = create_depth_map(image_input)
        df = depth_estimation(df, depth_df)

    df = position(df, image_input, 0.375, 0.625)
    prompt = generate_prompt(df, car_speed, location, weather, compare=True)
    response = make_api_call(prompt)

    # End of processing code: below writes the results to the results folder

    df.to_csv(os.path.join(results_dir, f"df/{filename}.csv"))

    shutil.copytree("tri-crop/predict/crops", f"{results_dir}/tri-crop/" + filename)
    shutil.copytree("Crops", f"{results_dir}/crops/" + filename)

    with open(os.path.join(results_dir, f"texts/{filename}.txt"), "w") as file:
        file.write(prompt)
        file.write("\n")
        file.write(response)

    return prompt, response, car_speed, df, location, weather


# TODO: Add loop of program


# load in ground truth data
truth = pd.read_csv("ground-truth/ground-truth-validation.csv")
results = truth.copy(deep=True)
results["Answer(word)"] = None
results["Answer(letter)"] = None
results["Speed"] = None
results.insert(4, "Location", "", True)
results.insert(5, "Weather", "", True)


start_time = time.time()

# loop over all images, capture output to not clutter notebook
for row in range(len(truth.index)):
    if len(truth.index) != len(os.listdir("images")):
        raise Exception(
            "Length of ground truth is not the same as the number of images"
        )
    tru_row = truth.loc[row]
    res_row = results.loc[row]

    image = tru_row["Filename"]
    prompt, response, car_speed, df, location, weather = run_program(image)
    resp_char = response.strip(" \n\t")[0]

    if resp_char == "A":
        resp_word = "Brake"
    elif resp_char == "B":
        resp_word = "Release Accelerator"
    elif resp_char == "C":
        resp_word = "Nothing"
    else:
        resp_word = "unknown"

    res_row["Answer(letter)"] = resp_char
    res_row["Speed"] = car_speed
    res_row["Location"] = location
    res_row["Weather"] = weather
    res_row["Answer(word)"] = resp_word

end_time = time.time()
elapsed_time = end_time - start_time

# After script finishes :

confu = metrics.confusion_matrix(truth[["Answer(letter)"]], results[["Answer(letter)"]])
score = metrics.accuracy_score(truth[["Answer(letter)"]], results[["Answer(letter)"]])


with open(os.path.join(results_dir, "info.txt"), "w") as file:
    file.write(f"Accuracy: {score}")
    file.write("\n")
    file.write(f"Runtime: {elapsed_time}s")
    file.write("\n")

np.save(os.path.join(results_dir, "confusion"), confu)
sns.heatmap(confu, annot=True, cmap="magma")
plt.savefig(os.path.join(results_dir, "confusion.png"))

results.to_csv(os.path.join(results_dir, "results.csv"))
