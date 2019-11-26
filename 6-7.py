from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Embedding,Flatten
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import preprocessing
path  = r"F:\5-model data\imdb.npz"
max_feature = 10000
maxlen = 20
(x_trian,y_train),(x_test,y_test) = imdb.load_data(
    path=path,
    num_words=max_feature
)
x_trian = preprocessing.sequence.pad_sequences(x_trian,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
model = Sequential()
model.add(Embedding(10000,8,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)
model.summary()
model.fit(
    x_trian,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)