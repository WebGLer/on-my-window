from tensorflow.python.keras.applications import vgg16
weights_path = r"F:\5-model data\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
model = vgg16.VGG16(
    weights =weights_path,
    include_top =False
)
model.summary()

#设置DeepDream配置
layer_contributions = {
    'mixed2':0.2,
    'mixed3':3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}