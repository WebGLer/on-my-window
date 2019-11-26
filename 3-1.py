from tensorflow.python.keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# print(train_data)
# word_index = imdb.get_word_index()
# reverse_word_index = dict(
#     [(value,key) for (key,value) in word_index.items()]
# )
# decoded_revied = ' '.join([reverse_word_index.get(i-3,"?") for i in train_data[0]])
# # print(reverse_word_index)
# # print(decoded_revied)
import numpy as ny
def vectorize_sequences(sequences,dimension = 10000):

    results = ny.zeros((len(sequences),dimension))
    for i ,sequence in enumerate(sequences):    #enumerate枚举，遍历对象
        results[i,sequence] =1.
    return results
# #把训练数据和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train,x_test)
#将标签向量化
y_train = ny.asarray(train_labels).astype('float32')
y_test = ny.asarray(test_labels).astype('float32')

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val =y_train[:10000]
partial_y_train = y_train[10000:]
from tensorflow.python.keras import layers,models
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    optimizer = 'rmsprop',
    loss= 'binary_crossentropy',
    metrics=['acc']#accuracy
)
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_val,y_val)
)

history_dict = history.history
print(history_dict.keys())


import matplotlib.pyplot as plt
#绘制训练损失和验证损失
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')         #bo表示蓝色圆点
plt.plot(epochs,val_loss_values,'b',label='Validation loss')    #b表示蓝色实线
plt.title('Training loss and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()       #清空图像
#绘制训练精度和验证精度
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,acc,'bo',label='Training acc')         #bo表示蓝色圆点
plt.plot(epochs,val_acc,'b',label='Validation acc')    #b表示蓝色实线
plt.title('Training acc and Validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()
r = model.evaluate(x_test,y_test)
print(r)