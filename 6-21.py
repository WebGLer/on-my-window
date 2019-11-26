import numpy as ny
timesteps = 100
input_features = 32
output_features = 64
inputs = ny.random.random((100,32))
print('inputs:',inputs.shape)
state_t = ny.zeros((64,))
w = ny.random.random((64,32))
u = ny.random.random((64,64))
b = ny.random.random((64,))

# successive_outputs = []
# for input_t in inputs:
#     output_t = ny.tanh(ny.dot(w,input_t)+ny.dot(u,state_t)+b)
#     successive_outputs.append(output_t)
#     state_t = output_t

