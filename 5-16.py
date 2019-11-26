
from tensorflow.python.keras.applications import VGG16
data_path = "F:\\5-data\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
conv_base = VGG16(
    weights = data_path,
    include_top =False,
    input_shape = (150,150,3)
)
conv_base.summary()
#提取预训练的卷积基提取特征
import os
import numpy as ny
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
base_dir = "D:\\参赛文件\\cats_and_dogs_small"
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
def extract_features(directory,sample_count):
    features = ny.zeros(shape=(sample_count,4,4,512))
    labels = ny.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150,150),
        batch_size = batch_size,
        class_mode='binary'
    )

    i = 0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size :(i + 1) * batch_size] = features_batch
        labels[i * batch_size :(i + 1) * batch_size] = labels_batch
        i +=1
        if i *batch_size >= sample_count:
            break
    return features,labels

train_features ,train_labels = extract_features(train_dir,2000)
validation_features ,validation_labels = extract_features(validation_dir,1000)
test_features , test_labels = extract_features(test_dir,1000)
# print(train_labels)
# print(test_labels)
# print(validation_labels)

#目前提取的特征形状为（samples，4,4,，512）我们要将其输入到密集连接分类器中，
#所以目前必须将其形状展平为（samples，8192）
train_features = ny.reshape(train_features,(2000,4*4*512))
validation_features = ny.reshape(validation_features,(1000,4*4*512))
test_features = ny.reshape(test_features,(1000,4*4*512))

#定义并训练密集连接分类器

from tensorflow.python.keras import models,layers,optimizers
model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim= 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc']
)
history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=20,
    validation_data=(
        validation_features,
        validation_labels
    )
)

#保存模型
import time
now = time.strftime('%Y-%m-%d %H-%M-%S')
file_path = "E:\\1- data\\models\\"+now+" cats_and_dogs_small VGG16.h5"
model.save(file_path)


#再次绘制loss曲线和acc曲线图
#绘制训练过程中的损失曲线和精度曲线
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.legend()

# import time
# now = time.strftime('%Y-%m-%d %H %M %S')
# model_path = "E:\\1- data\\models\\"
# model_name = "-cats_and_dogs loss:{0} acc:{1} val_loss:{2} val_acc:{3}.h5"\
#     .format(round(loss[-1],3),round(acc[-1],3),round(val_loss[-1],3),round(val_acc[-1],3))
# model_file_path = model_path+now+model_name
# print(model_file_path)
plt.show()