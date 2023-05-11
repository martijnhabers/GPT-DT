import cv2
from matplotlib import pyplot as plt
import os
import easyocr

reader = easyocr.Reader(["en"])

# adaptive threshold parameters
binary_1 = 17
binary_2 = 0


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y) : int(bottom_y), int(left_x) : int(right_x)]
    return img_cropped


def easyocr_detect(image, showImg=False):
    if not os.path.isfile(image):
        return ('unknown')

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = crop_img(img, 0.85)
    img = cv2.GaussianBlur(img, (5, 5), 1)

    # img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, binary_1, binary_2
    )
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, None, value=255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if showImg == True:
        plt.imshow(img)
        plt.show()

    result = reader.readtext(img, allowlist="0123456789")
    if len(result) > 0:
        return result[0][1]
    else:
        return "unknown"
