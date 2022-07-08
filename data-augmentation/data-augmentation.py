from numpy import expand_dims
from keras.utils import load_img
from keras.utils import img_to_array
#from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing_my_image import ImageDataGenerator
import matplotlib.pyplot as plt

img = load_img('golden_retriever.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)
'''
datagen = ImageDataGenerator(width_shift_range=[-0.1,0.1])
datagen = ImageDataGenerator(height_shift_range=[-0.1,0.1])
datagen = ImageDataGenerator(horizontal_flip=True)
datagen = ImageDataGenerator(vertical_flip=True)
datagen = ImageDataGenerator(rotation_range=10)
datagen = ImageDataGenerator(brightness_range=[0.7,1.3])
datagen = ImageDataGenerator(zoom_range=[0.7,1.3])
datagen = ImageDataGenerator(shear_range=10)
'''
# create generator instance
datagen = ImageDataGenerator(
    width_shift_range=[-0.1, 0.1],
    height_shift_range=[-0.1, 0.1],
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=5, # degrees
    brightness_range=[0.7, 1.3],
    zoom_range=0.3,
    shear_range=5, # degrees
    fill_mode='constant')
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    plt.subplot(330 + 1 + i)
    # generate batch of images using iterator
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

'''
def data_augmentation(images, labels):
    datagen = ImageDataGenerator(
        width_shift_range=[-0.1, 0.1],
        height_shift_range=[-0.1, 0.1],
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=5,  # degrees
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        shear_range=5,  # degrees
        fill_mode='constant')
    it = datagen.flow(images, labels, batch_size=1)
'''

