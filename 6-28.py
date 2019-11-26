import os
data_dir = r'D:\参赛文件'
file_name = os.path.join(data_dir,'jena_climate_2009_2016 (1).csv')

# print(data)
# i = 1

with open(file_name,'r',encoding= 'utf-8')as f:
    data=f.read()


lines = data.split('\n')

header = lines[0].split(',')
lines = lines[1:]
print('header:',header)

# for i in lines:
#     print(i)
#
# #解析数据
import numpy as ny
float_data = ny.zeros((len(lines),len(header)-1))
for i ,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

# print(float_data)
#绘制温度时间序列

# import matplotlib.pyplot as plt
# temp = float_data[:,1]
# plt.plot(range(len(temp)),temp)
# # plt.legend()
# plt.figure()
# plt.plot(range(1440),temp[:1440])
#
# plt.show()
mean = float_data[:200000].mean(axis = 0)
float_data -=mean

std = float_data[:200000].std(axis = 0)
float_data -=std

#生成时间序列样本及其目标的生成器
def generator(data,lookback,delay,min_index,max_index,
              shuffle = False,batch_size = 128,step = 6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = ny.random.randint(
                min_index + lookback,
                max_index,
                size=batch_size
            )
        else:
            if i + batch_size >=max_index:
                i = min_index + lookback
            rows = ny.arange(i,min(i +batch_size,max_index))
            i +=len(rows)
        samples = ny.zeros((
            len(rows),
            lookback //step,
            data.shape[-1]
        ))
        targets = ny.zeros(
            (
                len(rows,)
            )
        )
        for j ,row in enumerate(rows):
            indices = range(rows[j]-lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples,targets


#准备训练生成器，验证生成器和测试生成器
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index= 0,
    max_index=200000,
    shuffle=True
)

val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index= 200001,
    max_index=300000
)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index= 300001,
    max_index=None
)
val_steps = (300000-200001-lookback)//batch_size
test_steps = (len(float_data) - 300001-lookback) //batch_size
print(test_steps)
def evaluate_naive_mathod():
    batch_maes = []
    for step in range(val_gen):
        samples,targets = next(val_gen)
        preds = samples[:,-1,1]
