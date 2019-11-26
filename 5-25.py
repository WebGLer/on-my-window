img_path = r"D:\参赛文件\cats_and_dogs_small\test\cats\cat.1700.jpg"
from tensorflow.python.keras.preprocessing import image
import numpy as ny
img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = ny.expand_dims(img_tensor,axis=0)
img_tensor /= 255.
print(img_tensor.shape)
# import matplotlib.pyplot as plt
# plt.imshow(img_tensor[0])
# plt.show()


from tensorflow.python.keras import models
origenal_mode_path = "E:\\1- data\\2.h5"
model = models.load_model(origenal_mode_path)
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs = model.input,outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,7],cmap = 'viridis')
plt.show()