import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

IMG_SIZE = 224
_base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
_base.trainable = False

def mobilenet_vec(path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = preprocess_input(image.img_to_array(img))
    return _base.predict(arr[None, ...], verbose=0).flatten()
