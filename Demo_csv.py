import pandas as pd
import os,csv
data_dir = r'D:\参赛文件'
file_name = os.path.join(data_dir,'jena_climate_2009_2016 (1).csv')
data =pd.read_csv(file_name)
print(data[0:1])