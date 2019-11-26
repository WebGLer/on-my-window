from tensorflow.python.keras.datasets import boston_housing
data_path = "D:\\data\\boston_housing.npz"
(train_datas,train_targets),(test_datas,test_targets) = boston_housing.load_data(path=data_path)
print(train_datas.shape)

#数据标准化
#每列的平均值
mean= train_datas.mean(axis=0)
#减去平均值
train_datas -= mean
#每列的
std = train_datas.std(axis=0)
train_datas /= std
test_datas -=mean
test_datas /=std


#构建网络
from tensorflow.python.keras import layers,models
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_datas.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer='rmsprop',
        loss= 'mse',
        metrics=['mae']
    )
    return model


#K折验证
import numpy as ny
k = 4
num_val_samples = len(train_datas) //k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #',i)
    val_data = train_datas[i*num_val_samples:(i+1)*num_val_samples]
    val_targets =train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partail_train_datas = ny.concatenate(
        [train_datas[:i * num_val_samples],
         train_datas[(i+1)* num_val_samples :]],
        axis=0
    )
    partail_train_targets = ny.concatenate(
        [
            train_targets[:i * num_val_samples],
            train_targets[(i+1) * num_val_samples :]
        ],
        axis=0
    )

    model = build_model()
    history = model.fit(
        partail_train_datas,
        partail_train_targets,
        validation_data=(
            val_data,
            val_targets
        ),
        epochs = num_epochs,
        batch_size= 1 ,
        verbose= 0
    )
    val_mse ,val_mae = model.evaluate(
        val_data,
        val_targets,
        verbose=0
    )
    all_scores.append(val_mae)

    print(all_scores,ny.mean(all_scores))