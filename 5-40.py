from tensorflow.python.keras.applications.vgg16 import VGG16
weights_path = r'F:\5-model data\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = VGG16(
    weights =weights_path,
    include_top = False
)
model.summary()