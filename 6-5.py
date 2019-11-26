from tensorflow.python.keras.layers import Embedding
# from tensorflow.python.keras import layers,models
embedding_layer = Embedding(1000,32)
# model = models.Sequential()
# model.add(layers.Embedding(1000,32))
# model.summary()
print(embedding_layer.activity_regularizer)