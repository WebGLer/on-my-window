from tensorflow.python.keras.applications import VGG19
data_path = "F:\\5-model data\\vgg19_weights_tf_dim_ordering_th_kernels.h5"
VGG19(



weights =data_path
).summary()