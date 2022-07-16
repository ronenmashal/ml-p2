from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
from PIL import Image as im
import os
import wandb_helpers as wbh

with wbh.start_wandb_run("FCNN", None) as run:
    model = wbh.read_model(run, "FCNN", "latest")
    prediction = predict_image_label(model, img)
    print (prediction)