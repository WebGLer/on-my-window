import numpy as np
import matplotlib.pyplot as plt


arr = []
for i in range(20):
    arr.append(np.random.randint(1,20))


# #求均值
arr_mean = np.mean(arr)
#求方差
arr_var = np.var(arr)
#求标准差
arr_std = np.std(arr,ddof=1)
print('arr里面的数据：',arr)
print("平均值为：%f" % arr_mean)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)
temp_1 = 1/(np.sqrt(2*np.pi))
temp_2 = -np.square(arr)/2
plt.plot(range(len(arr)),temp_1*np.exp(temp_2),'red')
plt.show()