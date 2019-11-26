import os
imdb_dir = r'F:\5-model data\aclImdb\aclImdb'
train_dir = os.path.join(imdb_dir,'train')
labels = []
texts = []
for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    # print(dir_name)
    for name in os.listdir(dir_name):
        # print(name)
        # print(os.path.join(dir_name,name))
        if name[-4:] == '.txt':
            # print('1')
            with open(os.path.join(dir_name,name),'r',encoding='utf-8') as f:
                # print(f.read())
                texts.append(f.readline())
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
# print(texts)
#6-9
'''
对IMDB原始数据的文本进行分词
'''
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as ny
maxlen = 100
train_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer= Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences =tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.'%len(word_index))
data = pad_sequences(sequences,maxlen=maxlen)
labels = ny.asarray(labels)
print('shape of data tensor:',data.shape)
print('shape of labels tensor:',labels.shape)
indices = ny.arange(data.shape[0])
ny.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:train_samples]
y_train = labels[:train_samples]
x_val = data[train_samples:train_samples+validation_samples]
y_val = labels[train_samples:train_samples+validation_samples]



#6-10
'''
解析glove文件，该文件是由一个单词对应一个100维的向量
将其配置成一个embdeddings_index 字典：embdeddings_index[word] = coefs
'''
glove_dir = r"F:\5-model data\Glove"
embdeddings_index = {}
with open(os.path.join(glove_dir,'glove.6B.50d.txt'),'r',encoding='utf-8')as f:
    for line in f:
        values = line.split()
        word = values[0]

        coefs = ny.asarray(values[1:],dtype='float32')
        embdeddings_index[word] = coefs



#  6-11
'''

'''
embedding_dim = 50
embedding_matrix = ny.zeros((max_words,embedding_dim))
for word ,i in word_index.items():
    if i < max_words:
        embedding_vector = embdeddings_index.get(word)
        # print(embedding_vector)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
print("embedding_matrix.shape:",embedding_matrix.shape)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding,Flatten,Dense
model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()