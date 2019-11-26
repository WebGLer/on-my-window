from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.python.keras.preprocessing import image
def rotations():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen
train_cats_dir="D:\\参赛文件\\cats_and_dogs_small\\train\\cats"
fnames = [os.path.join(train_cats_dir,fname)for fname in os.listdir(train_cats_dir)]
img_path = fnames[1]        #选择一张图像进行增强
img = image.load_img(img_path,target_size=(150,150))        #读取图像并调整大小
x = image.img_to_array(img)     #将其转换为形状（150,150,3）的数组
x = x.reshape((1,)+x.shape)
i = 0

import matplotlib.pyplot as plt
for batch in rotations().flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4 ==4:
        break

    plt.show()