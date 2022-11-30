# AutoCorrect

Experimentation tings, yeah

So couple of ideas

The dataset is made by randomly pertubing the image's brightness, saturation, contrast and sharpness.

A vgg16 network is run over the augumented image and if the confidence >= `rho = .2`, then the image is *"realistic"* *"Plausible"*.

This Helps in removing images with conditions that are likely never going to occur.

## Experiments

- Convert RGB image to HSV and predict Saturation and Value (Brightness) channels for the corrected Image.
  - Model architecture -> Segnet with Vgg backbone

