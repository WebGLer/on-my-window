from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding,Flatten,Dense
model = Sequential()
model.add(Embedding(10000,50,input_length=100))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()