import tensorflow as tf
import numpy as np
from PIL import Image as im
from model1 import VanillaNN as PredictionModel
import os


img_idx = 23
img_file_name = f"image{img_idx}.bin"
if not os.path.isfile(img_file_name):
    images, labels = tf.keras.datasets.fashion_mnist.load_data()
    img = images[0][img_idx]
    with open(img_file_name, "wb") as file:
        np.save(file, img)
else:
    with open(img_file_name, "rb") as file:
        img = np.load(file)

model = PredictionModel.create_prediction_model("173338")
prediction = model.predict_image_label(img)
print (prediction)