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

    if np.random.rand() > .5: # sometimes let it predict normal images

        brightness_value = 1
        saturation_value = 1
        contrast_value = 1
        sharpness_value = 1

        brightness_adjustment = brightness_value - NORMAL_BRIGHTNESS
        saturation_adjustment =  saturation_value - NORMAL_SATURATION
        contrast_adjustment = contrast_value - NORMAL_CONTRAST
        sharpness_adjustment = sharpness_value - NORMAL_SHARPNESS
        

        return img, [brightness_adjustment, saturation_adjustment, contrast_adjustment, sharpness_adjustment]


    else:

        brightness_value = np.random.uniform(*BRIGHTNESS)
        saturation_value = np.random.uniform(*SATURATION)
        contrast_value = np.random.uniform(*CONTRAST)
        sharpness_value = np.random.uniform(*SHARPNESS)

        brightness_adjustment = brightness_value - NORMAL_BRIGHTNESS
        saturation_adjustment = saturation_value - NORMAL_SATURATION
        contrast_adjustment = contrast_value - NORMAL_CONTRAST
        sharpness_adjustment = sharpness_value - NORMAL_SHARPNESS


        #normalize between 0 and 1

        brightness_adjustment = (brightness_adjustment-MIN_BRIGHTNESS_LABEL)/(MAX_BRIGHTNESS_LABEL-MIN_BRIGHTNESS_LABEL)
        saturation_adjustment = (saturation_adjustment-MIN_SATURATION_LABEL)/(MAX_SATURATION_LABEL-MIN_SATURATION_LABEL)
        contrast_adjustment = (contrast_adjustment-MIN_CONTRAST_LABEL)/(MAX_CONTRAST_LABEL-MIN_CONTRAST_LABEL)
        sharpness_adjustment = (sharpness_adjustment-MIN_SHARPNESS_LABEL)/(MAX_SHARPNESS_LABEL-MIN_SHARPNESS_LABEL)

        

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


def unnormalize(brightness, saturation, contrast, sharpness):
    brightness = brightness*(MAX_BRIGHTNESS_LABEL-MIN_BRIGHTNESS_LABEL) + MIN_BRIGHTNESS_LABEL
    saturation = saturation*(MAX_SATURATION_LABEL-MIN_SATURATION_LABEL) + MIN_SATURATION_LABEL
    contrast = contrast*(MAX_CONTRAST_LABEL-MIN_CONTRAST_LABEL) + MIN_CONTRAST_LABEL
    sharpness = sharpness*(MAX_SHARPNESS_LABEL-MIN_SHARPNESS_LABEL) + MIN_SHARPNESS_LABEL

    return brightness, saturation, contrast, sharpness
