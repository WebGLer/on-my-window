#
from tensorflow.python.keras.applications import VGG16
data_path = "F:\\5-data\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
conv_base = VGG16(
    weights = data_path,
    include_top =False,
    input_shape = (150,150,3)
)
conv_base.summary()
print(conv_base.trainable)

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
