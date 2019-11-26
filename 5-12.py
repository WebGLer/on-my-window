from tensorflow.python.keras.preprocessing import image
import os
original_dataset_dir = "D:\\参赛文件\\kaggle1902\\train"
base_dir = "D:\\参赛文件\\cats_and_dogs_small"  #保存较小数据集的目录
# os.mkdir(base_dir)      #创建文件夹

#分别创建训练、测试、验证目录
train_dir = os.path.join(base_dir,'train')
# os.makedirs(train_dir)
validation_dir = os.path.join(base_dir,'validation')
# os.makedirs(validation_dir)
test_dir = os.path.join(base_dir,'test')
# os.mkdir(test_dir)

#创建猫和狗的训练目录
train_cats_dir = os.path.join(train_dir,'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs')
# os.mkdir(train_dogs_dir)

#创建猫和狗的验证目录
validation_cats_dir = os.path.join(validation_dir,'cats')
# os.mkdir(validation_cats_dir)
validation_dogs_dir =os.path.join(validation_dir,'dogs')
# os.mkdir(validation_dogs_dir)

#创建猫和狗的测试目录
test_cats_dir = os.path.join(test_dir,'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs')
# os.mkdir(test_dogs_dir)

#定义一个包含dropout的网络
from tensorflow.python.keras import layers,models
from tensorflow.python.keras import optimizers
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

#利用数据增强生成器训练卷积神经网络
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
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
now = time.strftime('%Y-%m-%d %H %M %S')
file_path = "E:\\1- data\\models\\"+now+"cats_and_dogs_small_2.h5"
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
plt.show()






#使用数据增强并观察随机增强后的图片

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# datagen =ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# fnames = [os.path.join(train_cats_dir,fname)for fname in os.listdir(train_cats_dir)]
# img_path = fnames[5]        #选择一张图像进行增强
# img = image.load_img(img_path,target_size=(150,150))        #读取图像并调整大小
# x = image.img_to_array(img)     #将其转换为形状（150,150,3）的数组
# x = x.reshape((1,)+x.shape)
# i = 0
#
# import matplotlib.pyplot as plt
# for batch in datagen.flow(x,batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i+=1
#     if i%4 ==4:
#         break
#
#     plt.show()