from tensorflow.python import keras
from tensorflow.python.keras import layers,models,backend
import numpy as ny
img_shape = (28,28,1)
batch_size = 16
latent_dim = 2

input_img = keras.Input(shape=img_shape)
x = layers.Conv2D(32,3,padding='same',activation='relu')(input_img)
x = layers.Conv2D(64,3,padding='same',activation='relu',)