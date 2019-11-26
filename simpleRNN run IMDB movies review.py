from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
max_features = 10000
maxlen = 500
batch_size = 32
print('Lodaing data...')
imdb_path = r"F:\5-model data\imdb.npz"
(input_train,y_train),(input_test,y_test) = imdb.load_data(
    path=imdb_path,
    num_words=max_features
)
print(len(input_train),'train sequences')
print(len(input_test),'test sequences')

print('pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)

#用Embedding层和SimpleRNN层来训练
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding,SimpleRNN,Dense
model = Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(
    optimizer='rmsprop',
    loss= 'binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    input_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

import matplotlib.pyplot as plt
acc =history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epoches = range(1,len(acc)+1)

plt.plot(epoches,acc,'bo',label = 'Training acc')
plt.plot(epoches,val_acc,'b',label = 'validation acc')
plt.title('Training and validation accuracy')
plt.legend()


plt.figure()

plt.plot(epoches,loss,'bo',label = 'Training loss')
plt.plot(epoches,val_loss,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()