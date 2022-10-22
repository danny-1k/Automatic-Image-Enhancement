from PIL import Image, ImageEnhance
import numpy as np


def read_img(img):
    img = Image.open(img)

    return 

def adjust_brightness(img, value):
    filter = ImageEnhance.Brightness(img)
    img = filter(value)

    return img, value

def adjust_contrast(img, value):
    filter = ImageEnhance.Contrast(img)
    img = filter(value)

    return img, value

def adjust_saturation(img, value):
    filter = ImageEnhance.Color(img)
    img = filter(value)

    return img, value

def adjust_sharpness(img, value):
    filter = ImageEnhance.Sharpness(img)
    img = filter(value)

    return img, value


def random_manipulation(img, min=-2, max=2):

    brightness_valaue = np.random.uniform(min, max)
    saturation_value = np.random.uniform(min, max)
    contrast_value = np.random.uniform(min, max)
    sharpness_value = np.random.uniform(min, max)

    img, _ = adjust_brightness(img, brightness_valaue)
    img, _ = adjust_saturation(img, saturation_value)
    img, _ = adjust_contrast(img, contrast_value)
    img, _ = adjust_sharpness(img, sharpness_value)

    return img, [brightness_valaue, saturation_value, contrast_value, sharpness_value]