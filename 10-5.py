import numpy as ny
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#定义文档
docs = [
    'Well done',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent',
    'Weak',
    'Pool effort',
    'not good',
    'poor work',
    'could have done better'
]
labels = ny.array([1,1,1,1,1,0,0,0,0,0])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
vocab_size = len(tokenizer.word_index)+1
encoded_docs = tokenizer.texts_to_sequences(docs)
print(encoded_docs)
max_length = 4
paded_docs = pad_sequences(encoded_docs,maxlen=max_length,padding='post')
print(paded_docs)
embedding_index = dict()
glove_path = r"F:\5-model and data\Glove\glove.6B.100d.txt"
with open(glove_path,'r',encoding='utf-8')as f:
    line = f.readline()
    while line:
        values = line.split()
        word = values[0]
        coefs = ny.asarray(values[1:])
        embedding_index[word] =coefs
        line = f.readline()
print('Load %s word vectors.'%len(embedding_index))
embedding_matrix = ny.zeros((vocab_size,100))
for word ,i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#定义模型
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding,Dense,Flatten
model = Sequential()
e = Embedding(vocab_size,100,weights=[embedding_matrix],input_length=4,trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
#compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
model.summary()
#fit model
history = model.fit(
    paded_docs,
    labels,
    epochs=50,
    verbose=0
)
import matplotlib.pyplot as plt
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,label = 'Training acc')
plt.title('Training accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,label = 'Training loss')
plt.title('Training loss')
plt.legend()
plt.show()