import numpy as ny
samples = ['The cat sat on the mat.','The dog ate my homework']
dimensionlity = 1000
max_length = 10
results = ny.zeros((len(samples),max_length,dimensionlity))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word))%dimensionlity
        results[i,j,index] = 1.
print(results.shape)