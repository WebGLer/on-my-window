# import numpy as ny
# a = ny.array([1,2,3,4,5])
# b = a.T
# # c = ny.dot(a,b)
# # print(c,c.ndim)
# #
# # print(a.ndim)
# # print(a)
# # print(b.ndim)
# print(b)
# z= ny.zeros(5)
# print(z)
# from tensorflow.python.keras.datasets import imdb
# (train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
# print(len(train_data))
# # for i ,j in enumerate(train_data):
# #     print(i,j)
#
# import numpy as ny
# def vectorize_sequences(sequences,dimension = 10000):
#     print(len(sequences))
#     b = (len(sequences),dimension)
#     a =ny.zeros(b)
#     print(a.shape)
#     # results = ny.zeros(len(sequences),dimension)
#     # for i ,sequence in enumerate(sequences):    #enumerate枚举，遍历对象
#     #     results[i,sequence] =1.
#     # return results
#
# x = vectorize_sequences(train_data)
# print(x)
# import numpy as ny
# a = ny.random.randint(1,20,10)
# for i in a:
#     print(i)
#
# print('*'*20)
# a_b = a[3:6]
# b = a[:3]
# c = a[6:]
# a_c = ny.hstack((b,c))
# for i in a_c:
#     print(i)
# #2019年9月24日20:23:18
# from tensorflow.python.keras.utils import to_categorical
# import numpy as ny
# a = ny.array(
#     [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#     ]
# )
# b = to_categorical(a-1)
#
# print(a.shape)
# print(b,b.shape)
# fnames = ['cats.{}.png'.format(i) for i in range(10)]
# for i in fnames:
#     print(i)



# import time
# now = time.strftime('%Y-%m-%d %H %M %S')
# model_path = "E:\\1- data\\models\\"
# model_name = "-cats_and_dogs loss:{0} acc:{1} val_loss:{2} val_acc:{3}.h5"\
#     .format(round(loss[-1],3),round(acc[-1],3),round(val_loss[-1],3),round(val_acc[-1],3))
# model_file_path = model_path+now+model_name
# print(model_file_path)


# import time
# now = time.strftime('%Y-%m-%d %H_%M_%S')
# file_path = "E:\\1- data\\models\\"+now+" cats_and_dogs_small_1.h5"
# model.save(file_path)
import numpy as ny
# a = []
# for i in range(1,10):
#     a.append(i)
# print(a)
# print(a[:,-1,1])
a =(1,2,3,4,5,6,7,8,9)
print(a[:,1])