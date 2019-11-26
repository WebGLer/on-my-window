#给出文件夹路径
import os
base_dir = "D:\\参赛文件\\cats_and_dogs_small"
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')
#使用数据增强的特征提取
from tensorflow.python.keras.applications import VGG16
data_path = "F:\\5-model data\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
conv_base = VGG16(
    weights = data_path,
    include_top =False,
    input_shape = (150,150,3)
)
from tensorflow.python.keras import layers,models
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
print("model.trainable_weights=",len(model.trainable_weights))
# conv_base.trainable = False


#冻结conv_base层网络并将最后一个卷积层解冻
set_trainable = False
for layer in conv_base.layers:
    if layer.name =='block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print("After freeaed model.trainable_weights=",len(model.trainable_weights))
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc']
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)
#保存模型
import time
now = time.strftime('%Y-%m-%d %H-%M-%S')
file_path = "E:\\1- data\\models\\"+now+" cats_and_dogs VGG16-数据增强-模型最后卷积层网络微调.h5"
model.save(file_path)
#下面是绘制图像
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label ='Validation acc' )
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label ='Validation loss' )
plt.title('Training and Validation Loss')
plt.legend()
plt.show()