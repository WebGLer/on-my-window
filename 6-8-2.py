import os
imdb_dir = r'F:\5-model data\aclImdb\aclImdb'
train_dir = os.path.join(imdb_dir,'train')
labels = []
texts = []
for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    # print(dir_name)
    for name in os.listdir(dir_name):
        print(name[-4:])