import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider, Button
import tensorflow as tf
import numpy as np
from PIL import Image as im
import requests

images, labels = tf.keras.datasets.fashion_mnist.load_data()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
img = plt.imshow(im.fromarray(images[0][0]))

slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(
    ax = slider_ax,
    label = "Image",
    valmin = 0,
    valmax = 60000 - 1,
    valinit = 0,
    valstep = 1.0,
    orientation = "horizontal"
)

def update(val):
    global img
    idx = int(val)
    img.set_data(im.fromarray(images[0][idx]))

slider.on_changed(update)

check_area = plt.axes([0.8, 0.025, 0.1, 0.04])
check_button = Button(check_area, 'check')

def check(x):
    request_params = { 'image[]': images[0][int(slider.val)] }
    result = requests.get("http://localhost:5000/api/v1/check", request_params)
    answer = result.json()
    print (answer)

check_button.on_clicked(check)

plt.show()