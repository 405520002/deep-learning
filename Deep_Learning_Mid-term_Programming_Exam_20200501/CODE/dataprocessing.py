# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:40:19 2020

@author: User
"""

import os
import pandas as pd
data={}
data_dir=os.listdir('D:\stuff\class\DL\Deep_Learning_Mid-term_Programming_Exam_20200501\DATA\packet')
for i in range(len(data_dir)):
        data[i]=pd.read_csv(data_dir[i],error_bad_lines=False,engine='python')
ackflooding=data[0]
arpspoofing=data[1]
arpspoofing2=data[2]
arpspoofing=pd.concat([arpspoofing,arpspoofing],axis=0)
synflooding=pd.concat([data[3],data[4]],axis=0)
httpflooding=pd.concat([data[5],data[6]],axis=0)
normal=data[7]
#add label
def add_label(data,label):
    data['label']=label
    return data

###final data###        
ackflooding=add_label(ackflooding,'ackflooding').reset_index(drop=True)
arpspoofing=add_label(arpspoofing,'arpspoofing').reset_index(drop=True)
synflooding=add_label(synflooding,'synflooding').reset_index(drop=True)
httpflooding=add_label(httpflooding,'httpflooding').reset_index(drop=True)
normal=add_label(normal,'normal').reset_index(drop=True)

def info_processing(d):
    new= d["Info"].str.split(">", n =-1, expand = True) 
    new2=new[1].str.split('[ACK]',n=1,expand=True)
    new2[0]=new2[0].str.split('[',n=1,expand=True)
    new2[['3','4']]=new2[1].str.split(']',n=1,expand=True)
    new3=new2['4'].str.split(" ",n=5,expand=True)
    new4=new[0].str.split('[',n=1,expand=True)
    new5=new4[1].str.split(']',n=1,expand=True)
    new5.rename(columns={new5.columns[0]:'a0',new5.columns[1]:'a1'},inplace=True)
    new4.rename(columns={new4.columns[0]:'b0',new4.columns[1]:'b1'},inplace=True)
    new4=pd.concat([new4,new5],axis=1)
    for i in range(len(new4['b0'])):
        if new4['b1'][i]!= None:
            new4['b0'][i]=new4['a1'][i]
            print(i)
    new4=new4.drop(columns=['b1','a1'])
    new3.rename(columns={new3.columns[5]:'five'},inplace=True)
    new3=new3.drop(columns=['five'])
    array=['Seq', 'Ack', 'Win', 'Len']
    for i in range(1,5):
        print(i)
        new3_1=new3[i].str.split("=",n=-1,expand=True)
        new3_1.rename(columns={new3_1.columns[1]:array[i-1],new3_1.columns[0]:'five'},inplace=True)
        new3_1=new3_1.drop(columns=['five'])
        new3=pd.concat([new3,new3_1],axis=1)
    new3=new3[['Seq', 'Ack', 'Win', 'Len']]
    new3=pd.concat([new3,new4],axis=1)
    new2=pd.concat([new3,new2[0]],axis=1)
    new2.rename(columns={new2.columns[4]:'first_num',new2.columns[5]:'task',new2.columns[6]:'second_num'},inplace=True)
    d=pd.concat([d,new2],axis=1)
    return d

ack=info_processing(ackflooding)
arp=info_processing(arpspoofing)
http=info_processing(httpflooding)
syn=info_processing(synflooding)
norm=info_processing(normal)
final_data=pd.concat([ack,arp,http,syn,norm],axis=0)
final_data=final_data.fillna(-1)
final_data=final_data.drop(columns=['Info'])



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
features=['Source', 'Destination', 'Protocol','first_num', 'task', 'second_num','Ack', 'Win', 'Len']
final_data['task']=final_data['task'].astype(str)
final_data['second_num']=final_data['second_num'].astype(str)
final_data['Ack']=final_data['Ack'].astype(str)
final_data['Win']=final_data['Win'].astype(str)
final_data['Len']=final_data['Len'].astype(str)
for i in features: 
    print()
    final_data[i]=l.fit_transform(final_data[i])

def labelcode(x):
    if x=='ackflooding':
        return 1
    elif x== 'arpspoofing':
        return 2
    elif  x== 'httpflooding':
        return 3
    elif x== 'synflooding':
        return 4
    else:return 0
    
final_data['label']=final_data['label'].apply(labelcode)
final_data=final_data.sort_values(by=['Time'])
sc = MinMaxScaler(feature_range = (0, 1))
array=['Ack', 'Win', 'Len']

s_features=['Time', 'Source', 'Destination', 'Protocol', 'Length', 'label',
       'Seq', 'Ack', 'Win', 'Len', 'first_num', 'task', 'second_num']

for i in s_features:
    final_data[[i]]=sc.fit_transform(final_data[[i]])
final_data=final_data.drop(columns=['No.'])
final_data.to_csv('final_data.csv')
        


