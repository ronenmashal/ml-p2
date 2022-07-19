from flask import Flask 
from flask import request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)

local_dir = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(f"{local_dir}\\work-model.h5")

@app.route("/", methods = [ "GET" ])
def home():
    return "<h1>ML Project 2 Is up and running</h1>"

@app.route("/api/v1/check", methods = [ "GET" ])
def predict():
    if not 'image[]' in request.args:
        return validation_error("Image was not provided in request.")

    image = np.array(request.args.getlist('image[]')).astype(np.uint8)
    if (len(image) != 28 * 28):
        return validation_error("Image is not of correct size. It must be 28 x 28 pixels.")

    response = model.predict_image_label(image)
    return jsonify(str(response))
    

def validation_error(message):
    return jsonify({ "error": message })

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
