from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
from PIL import Image as im
import os
import wandb_helpers as wbh

import seaborn as sns
import matplotlib.pyplot as plt     


#with wbh.start_wandb_run("FCNN-metrics", None) as run:
    #train_set, validation_set, test_set = wbh.read_datasets(run)
    #model = wbh.read_model(run, "FCNN", "latest")
    
test_set = wbh.read_dataset('.\\artifacts\\fashion-mnist-v2', 'test')
model = tf.keras.models.load_model('.\\artifacts\\FCNN-v0')
#y_test = np.argmax(test_set.labels)
predictions = model.predict(test_set.images)
y_test = np.argmax(predictions, axis = 1)
print (classification_report(test_set.labels, y_test))
cm = confusion_matrix(test_set.labels, y_test)

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

ax = plt.subplot()
h = sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)

plt.show()