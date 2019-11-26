import os,shutil
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

#将前1000张猫的图片复制到train_cats_dir的文件夹下
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(train_cats_dir,fname)
#     shutil.copyfile(scr,dst)
#
#
# #将500张猫的图片复制到validation_cats_dir的文件夹下
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(validation_cats_dir,fname)
#     shutil.copyfile(scr,dst)
#
#
# #将500张猫的图片复制到test_cats_dir的文件夹下
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir,fname)
#     dst = os.path.join(test_cats_dir,fname)
#     shutil.copyfile(scr,dst)
#
# # 将前1000张狗的图片复制到train_dogs_dir的文件夹下
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(scr, dst)
#
# # 将500张狗的图片复制到validation_dogs_dir的文件夹下
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(scr, dst)
#
# # 将500张狗的图片复制到test_dogs_dir的文件夹下
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     scr = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(scr, dst)

print('total training cats images :',len(os.listdir(train_cats_dir)))
print('*'*100)
print('total training dogs images :',len(os.listdir(train_dogs_dir)))
print('*'*100)
print('total validation cats images :',len(os.listdir(validation_cats_dir)))
print('*'*100)
print('total validation dogs images :',len(os.listdir(validation_dogs_dir)))
print('*'*100)
print('total test cats images :',len(os.listdir(test_cats_dir)))
print('*'*100)
print('total test dogs images :',len(os.listdir(test_dogs_dir)))
print('*'*100)


#接下来就是构建网络
from tensorflow.python.keras import layers,models
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
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

#配置模型用于训练
from tensorflow.python.keras import optimizers
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=0.0001),
    metrics= ['acc']
)

#数据预处理
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

# for datas_batch ,labels_batch in train_generator:
#     print('data batchs shape:',datas_batch.shape)
#     print('data labels shape:',labels_batch.shape)
#     break
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

import time
now = time.strftime('%Y-%m-%d %H-%M-%S')
file_path = "E:\\1- data\\models\\"+now+" cats_and_dogs_small_1.h5"
model.save(file_path)
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