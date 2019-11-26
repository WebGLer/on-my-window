from tensorflow.python.keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.','The dog ate my homework']
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)
print(tokenizer.fit_on_texts(samples))
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
one_hot_results =tokenizer.texts_to_matrix(samples,mode='binary')
print(one_hot_results)
print(one_hot_results.shape)
word_index = tokenizer.word_index
print(word_index)