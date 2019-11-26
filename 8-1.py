from tensorflow.python import keras
import numpy as np
file_path = r"F:\5-model data\nietzsche.txt"
with open(file_path,'r',encoding='utf-8')as f:
    text = f.read().lower()

print(len(text))

#将字符序列向量化
maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0,len(text)-maxlen,step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print('Number of sequences:',len(sentences))
#语料中将唯一字符组成的列表
chars = sorted(list(set(text)))
# print(chars)
print('Unique characters:',len(chars))
char_indices = dict((char,chars.index(char)) for char in chars)
print('Vectorization...')
x = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
for i ,sentence in enumerate(sentences):
    for t,char in enumerate(sentence):
        x[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1

# print(y.shape)
print('构建网路')
from tensorflow.python.keras import models,layers
model = models.Sequential()
model.add(layers.LSTM(128,input_shape=(maxlen,len(chars))))
model.add(layers.Dense(len(chars),activation='softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer
)