import tensorflow as tf
import numpy as np
from PIL import Image as im
import os
import wandb_helpers as wbh

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def predict_image_label(model, image):
    '''
    Predict a label based on a given image.
    The image must be an array of uint8, containing 28 * 28 = 784 items. 
    The shape does not matter. It will be reshaped to the necessary form.
    '''
    image = image / 255.0
    image = image.reshape([1, 28, 28])
    predictions = model.predict(image)
    label_idx = np.argmax(predictions[0])
    prediction = {
        "label id": label_idx,
        "label": class_names[label_idx],
        "certainty": predictions[0][label_idx]
    }
    return prediction

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

with wbh.start_wandb_run("FCNN", None) as run:
    model = wbh.read_model(run, "FCNN", "latest")
    prediction = predict_image_label(model, img)
    print (prediction)