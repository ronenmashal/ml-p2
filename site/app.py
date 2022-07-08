from flask import Flask 
from flask import request, jsonify
import numpy as np

# If you want to change the model used by the web-service to predict the label, replace this
# import with the module of the new model. Also, make sure it has the same static and private
# functions as VanillaNN.
#
import tensorflow
from model1 import VanillaNN as PredictionModel

app = Flask(__name__)
model = PredictionModel.create_prediction_model("164453")

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

