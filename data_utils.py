from PIL import Image, ImageEnhance
import numpy as np
from constants import *

def read_img(img):
    img = Image.open(img)
    img = img.convert('RGB') #ensure image is RGB
    return img

def adjust_brightness(img, value):
    filter = ImageEnhance.Brightness(img)
    img = filter.enhance(value)

    return img, value

def adjust_contrast(img, value):
    filter = ImageEnhance.Contrast(img)
    img = filter.enhance(value)

    return img, value

def adjust_saturation(img, value):
    filter = ImageEnhance.Color(img)
    img = filter.enhance(value)

    return img, value

def adjust_sharpness(img, value):
    filter = ImageEnhance.Sharpness(img)
    img = filter.enhance(value)

    return img, value


def random_manipulation(img):

    # we want the model to correct images that are in the specified ranges in `CONSTANTS.py`
    # the correction will be added to the original values of the image


    brightness_value = np.random.uniform(*BRIGHTNESS)
    saturation_value = np.random.uniform(*SATURATION)
    contrast_value = np.random.uniform(*CONTRAST)
    sharpness_value = np.random.uniform(*SHARPNESS)

    brightness_adjustment = brightness_value - 1
    saturation_adjustment = saturation_value - 1
    contrast_adjustment = contrast_value - 1
    sharpness_adjustment = sharpness_value - 1
    

    img, _ = adjust_brightness(img, brightness_value)
    img, _ = adjust_saturation(img, saturation_value)
    img, _ = adjust_contrast(img, contrast_value)
    img, _ = adjust_sharpness(img, sharpness_value)

    return img, [brightness_adjustment, saturation_adjustment, contrast_adjustment, sharpness_adjustment]


def defined_manipulation(img, brightness, saturation, contrast, sharpness):
    img, _ = adjust_brightness(img, brightness)
    img, _ = adjust_saturation(img, saturation)
    img, _ = adjust_contrast(img, contrast)
    img, _ = adjust_sharpness(img, sharpness)

    return img