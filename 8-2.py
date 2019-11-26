from tensorflow.python import keras
import numpy as ny
path = keras.utils.get_file(
     'nietzsche.txt',
     origin= 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text = open(path).read().lower()
print(len(text))