import cv2
from matplotlib import pyplot as plt
import os
import easyocr

reader = easyocr.Reader(["en"])


def easyocr_detect(image, showImg=False):
    if not os.path.isfile(image):
        raise Exception("Input image path is not a file!")

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    # img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2
    )
    convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if showImg == True:
        plt.imshow(convert)
        plt.show()

    result = reader.readtext(img, allowlist="0123456789")
    if len(result) > 0:
        return result[0][1]
    else:
        return "unknown"
