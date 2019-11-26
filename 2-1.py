from tensorflow.python.keras.datasets import mnist
(train_images,train_lables),(test_images,test_lables) = mnist.load_data()
# print(train_images.shape)
print(train_lables[2])
#构建网络
from tensorflow.python.keras import layers,models
net_work = models.Sequential()
net_work.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
net_work.add(layers.Dense(10,activation='softmax'))
# net_work.summary()
#编译网络
net_work.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#训练数据预处理
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

#对标签编码
from tensorflow.python.keras.utils import to_categorical
train_lables = to_categorical(train_lables)
test_lables = to_categorical(test_lables)


#拟合模型
net_work.fit(
    train_images,
    train_lables,
    epochs=5,
    batch_size=128
)
# import time
# now = time.strftime('%Y-%m-%d %H_%M_%S')
# file_path = "E:\\1- data\\models"+now+"cats_and_dogs_small.h5"
# net_work.save(file_path)
test_loss,test_acc = net_work.evaluate(test_images,test_lables)
print('test_loss:',test_loss,"test_acc:",test_loss)