# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:15:45 2020

@author: User
"""
import pandas as pd
import os
label=pd.read_csv('dev.csv')
data_dir=os.listdir('D:\stuff\class\DL\cnn_mango\C1-P1_Dev\data')
data=pd.DataFrame(data_dir)
data['n']="\\"
data['p']='D:\stuff\class\DL\cnn_mango\C1-P1_Dev\data'
data['path']=data['p']+data['n']+data[0]
data.rename(columns={data.columns[0]:'image_id'},inplace=True)
data=pd.merge(data,label,on='image_id')
data=data[['path','label']]
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
data['label']=l.fit_transform(data['label'])
data.to_csv('matlab_test_mango.csv')