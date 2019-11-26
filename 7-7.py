from tensorflow.python import keras as keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
import os
import tensorboard
max_features = 2000
max_len = 500
data_path = r"F:\5-model data\imdb.npz"
(x_train,y_train),(x_test,y_test) = imdb.load_data(
    path=data_path,
    num_words=max_features
)
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)
model = keras.models.Sequential()
model.add(layers.Embedding(
    max_features,
    128,
    input_length=max_len,
    name='embed'
))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(
    optimizer='rmsprop',
    loss=  'binary_crossentropy',
    metrics=['acc']
)
log_dir = r"E:\1- data\log\1.log"
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        embeddings_freq=1,
    )
]
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    validation_split=0.2

)
from tensorflow.python.keras.utils import plot_model
import pydot,graphviz
plot_model(model,to_file='model.png')