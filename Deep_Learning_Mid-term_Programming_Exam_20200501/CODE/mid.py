# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:02:32 2020

@author: User
"""

import pandas as pd
import numpy as np 
import os
import datetime
#read file into dictionary
data=os.listdir('D:\stuff\class\DL\Deep_Learning_Mid-term_Programming_Exam_20200501\DATA')
dict={}
for i in range(len(data)-1):
        dict[i]=pd.read_csv(data[i],error_bad_lines=False,engine='python')

#check null value
for i in range(len(dict)-1):
    print(dict[i].isnull().sum())

#date to season    
def season(x):
    if x<=5:
        if x<2:
            return 4
        else: return 1
    else:
        if x<9:
            return 2
        else: 
            if x<=11:
                return 3
            else: return 4
#processing dict[0]#air_reserv
d0=dict[0]
d0['visit_datetime'] = pd.to_datetime(d0['visit_datetime'])
d0['visit_datetime'] = d0['visit_datetime'].dt.date
d0['reserve_datetime'] = pd.to_datetime(d0['reserve_datetime'])
d0['reserve_datetime'] = d0['reserve_datetime'].dt.date    
d0['r_diff']= d0.apply(lambda r: (r['visit_datetime']- r['reserve_datetime']).days, axis=1)
d0 = d0.groupby(['air_store_id','visit_datetime'], as_index=False)[['r_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
d0['m']= pd.to_datetime(d0['visit_date']).dt.month
d0['y']=pd.to_datetime(d0['visit_date']).dt.year
d0['d']=pd.to_datetime(d0['visit_date']).dt.day
d0['season']=d0['m'].apply(season)
d0['i']=d0['air_store_id']+d0['visit_date'].astype(str)
#processing dict[1]air_store
d1=dict[1]

#processing dict[2] airvisit_data
#seperate m,y,d
d2=dict[2]
d2['visit_date'] = pd.to_datetime(d2['visit_date'])
d2['visit_date'] = d2['visit_date'].dt.date
d2['m']= pd.to_datetime(d2['visit_date']).dt.month
d2['y']=pd.to_datetime(d2['visit_date']).dt.year
d2['d']=pd.to_datetime(d2['visit_date']).dt.day
#merge d3 d2
d3=dict[3]
d3=d3.rename(columns={'calendar_date':'visit_date'})
d2['visit_date'] = d2['visit_date'].astype(str)
d2=pd.merge(d2,d3,on="visit_date")
#use group by to make sure the date of each airstore is unique 
#add columns the min mean max sum visitors in each day of week
unique_stores=d2['air_store_id'].unique()
d2['i']=d2['air_store_id']+d2['day_of_week']
temp = d2.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
temp['i']=temp['air_store_id']+temp['day_of_week']
temp=temp.drop(columns=['air_store_id','day_of_week'])
d2=pd.merge(d2,temp,on=['i'])
temp = d2.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
temp['i']=temp['air_store_id']+temp['day_of_week']
temp=temp.drop(columns=['air_store_id','day_of_week'])
d2=pd.merge(d2,temp,on=['i'])

temp = d2.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
temp['i']=temp['air_store_id']+temp['day_of_week']
temp=temp.drop(columns=['air_store_id','day_of_week'])
d2=pd.merge(d2,temp,on=['i'])

temp = d2.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
temp['i']=temp['air_store_id']+temp['day_of_week']
temp=temp.drop(columns=['air_store_id','day_of_week'])
d2=pd.merge(d2,temp,on=['i'])

temp = d2.groupby(['air_store_id','day_of_week'], as_index=False)['visitors'].sum().rename(columns={'visitors':'sum_visitors'})
temp['i']=temp['air_store_id']+temp['day_of_week']
temp=temp.drop(columns=['air_store_id','day_of_week'])
d2=pd.merge(d2,temp,on=['i'])
d2=pd.merge(d2,d1,on=['air_store_id'])


#processing dict[4] hpg_reserve
#using groupby to make sure that every order in visitdate is unique
#add 2 columns r_diff visit visitors-reserve visitors and seperate y,m,d,season 
d4=dict[4]
d4['visit_datetime'] = pd.to_datetime(d4['visit_datetime'])
d4['visit_datetime'] = d4['visit_datetime'].dt.date
d4['reserve_datetime'] = pd.to_datetime(d4['reserve_datetime'])
d4['reserve_datetime'] = d4['reserve_datetime'].dt.date    
d4['r_diff']= d4.apply(lambda r: (r['visit_datetime']- r['reserve_datetime']).days, axis=1)
#merge store relation_info to have merging index
d6=pd.read_csv('store_id_relation.csv')
d4=pd.merge(d4,d6,on=['hpg_store_id'],how='inner')
d4 = d4.groupby(['air_store_id','visit_datetime'], as_index=False)[['r_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})
d4['m']= pd.to_datetime(d4['visit_date']).dt.month
d4['y']=pd.to_datetime(d4['visit_date']).dt.year
d4['d']=pd.to_datetime(d4['visit_date']).dt.day
d4['season']=d4['m'].apply(season)
d4['i']=d4['air_store_id']+d4['visit_date'].astype(str)
#processing dict hpg_store_info
d5=dict[5]
#merge some people didnt go to the restaurant by reserve fillna -1
d2['i']=d2['air_store_id']+d2['visit_date'].astype(str)
data_all= pd.merge(d2,d4,how='left', on=['i']) 
d0['i']=d0['air_store_id']+d0['visit_date'].astype(str)
data_all= pd.merge(data_all,d0,how='left', on=['i']) 
data_all=data_all.fillna(-1)
data_all=data_all.drop(columns=['air_store_id_y','visit_date_y','m_y', 'y_y', 'd_y','m', 'y', 'd', 'season_y','season_x'])
data_all=data_all.drop(columns=['air_store_id','air_store_id', 'visit_date'])

####label encoding and scaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
sc2 = MinMaxScaler(feature_range = (-1, 1))
features=['min_visitors', 'mean_visitors',
       'max_visitors', 'median_visitors', 'sum_visitors','latitude', 'longitude']

for i in features:
    data_all[[i]]=sc.fit_transform(data_all[[i]])

features=['r_diff_x','reserve_visitors_x', 'r_diff_y', 'reserve_visitors_y']

for i in features: 
    data_all[[i]]=sc2.fit_transform(data_all[[i]])
    
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
features=['air_store_id_x','air_genre_name','air_area_name']

for i in features:    
    data_all[i]=l.fit_transform(data_all[i])

def get_dum(feature,data_all):
    data=pd.get_dummies(data_all[feature])
    data_all=pd.concat([data_all,data],axis=1)
    data_all=data_all.drop(columns=[feature])
    return data_all
    
data_all=get_dum('day_of_week',data_all)
data_all=get_dum('m_x',data_all)   
data_all=get_dum('y_x',data_all) 
data_all=get_dum('air_genre_name',data_all) 
data_all=get_dum('air_area_name',data_all) 
data=data_all.drop(columns=['i'])

features=['air_store_id_x','d_x']
for i in features:
    data[[i]]=sc.fit_transform(data[[i]])
    
#split train_test

train=data[data[2016]==1]
test=data[data[2017]==1]
train = train.sort_values('visit_date_x')
train=train.drop(columns=['visit_date_x'])
test = test.sort_values('visit_date_x')
test=test.drop(columns=['visit_date_x'])
train_size = int(len(train) * 0.7)
val_size = len(train) - train_size
train, val = train.iloc[0:train_size,:], train.iloc[train_size:len(train),:]
trainX=train.drop(columns=['visitors']).to_numpy()
trainY=train[['visitors']].to_numpy() 
valX=val.drop(columns=['visitors']).to_numpy()
valY = val[['visitors']].to_numpy()
#add time series
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))
#because didn't drop so change two columns so that is same with train data (year doesn't matter)
test[2016]=1
test[2017]=0
testX=test.drop(columns=['visitors']).to_numpy()
testy=test[['visitors']].to_numpy()
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#model_training
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import regularizers
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM,TimeDistributed
#model 1
def lstm_model():
    model = Sequential()
    model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2]),activation='sigmoid',return_sequences=True))
    model.add(LSTM(5,activation='sigmoid'))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='Adam',metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=250, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
        
    return model,history
#model2 with batch normalization and drop out
def lstm_model_improve():
    model = Sequential()
    model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2]),activation='sigmoid',return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(5,activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=250, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='loss', patience=5, verbose=0)])
        
    return model,history

#model3 with sgd
def lstm_model_sgd():
    model = Sequential()
    model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2]),activation='sigmoid',return_sequences=True))
    model.add(LSTM(5,activation='sigmoid'))
    model.add(Dense(1,activation='relu'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd,metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=250, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0)])
        
    return model,history
#model4 using leaky relu
from keras.layers import LeakyReLU
def lstm_model_leaky_relu():
    model = Sequential()
    model.add(LSTM(10, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
    model.add(LeakyReLU(alpha=0.1))
    model.add(LSTM(10,activation='sigmoid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1))
    model.add(LeakyReLU(alpha=0.1))
    model.compile(loss='mse', optimizer='Adam',metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=250, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=15, verbose=0)])
        
    return model,history

#using initializers
from keras.layers import initializers
def lstm_model_leaky_relu_init():
    model = Sequential()
    model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True,kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(LSTM(5,activation='sigmoid'))
    model.add(Dense(1))
    model.add(LeakyReLU(alpha=0.1))
    model.compile(loss='mse', optimizer='Adam',metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=250, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='val_mean_absolute_error', patience=15, verbose=0)])
        
    return model,history


def lstm_model_init():
    model = Sequential()
    model.add(LSTM(5, input_shape=(trainX.shape[1], trainX.shape[2]),activation='sigmoid',return_sequences=True,kernel_initializer=initializers.RandomNormal(stddev=0.01),
    bias_initializer=initializers.Zeros()))
    model.add(LSTM(5,activation='sigmoid'))
    model.add(Dense(1,activation='relu'))
    model.compile(loss='mse', optimizer='Adam',metrics=['mae'])
    history = model.fit(trainX, trainY, epochs=350, batch_size=500,
                            validation_data=(valX, valY), verbose=1, shuffle=False,callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
        
    return model,history
#history = model.fit(X_train.to_numpy(), y_train.to_numpy(), epochs=50, batch_size=1500, validation_data=(X_val, y_val), verbose=2, shuffle=False, callbacks = [EarlyStopping(monitor='val_loss', patience=6, verbose=0)])


model1,history1=lstm_model()
model2,history2=lstm_model_improve()#using dropout and batch normalization
model3,history3=lstm_model_sgd()
model4,history4=lstm_model_leaky_relu()
model5, history5=lstm_model_leaky_relu_init()#leaky relu won't cause vanishing gradient problem
model6,history6=lstm_model_init()#best_model but sometimes cause vanishing gradient problem

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
%matplotlib qt
def evaluate_draw_model(model):     
    y_hat=model.predict(testX)
    score=[mean_squared_error(testy, y_hat),mean_absolute_error(testy, y_hat)]
    y_hat=pd.DataFrame(y_hat,columns=['visitors'])
    y=pd.DataFrame(testy,columns=['visitors'])
    y=y.reset_index()
    y_hat=y_hat.reset_index()
    plt.plot(y['index'],y['visitors'],color='green')
    plt.plot(y_hat['index'],y_hat['visitors'],color='blue')
    return score,y_hat

error={}
#add error data into dict
def error_data(history1,model,modelstr):  
    score,y_hat=evaluate_draw_model(model)
    mse=history1.history['loss'][len(history1.history['loss'])-1]
    mae=history1.history['mean_absolute_error'][len(history1.history['loss'])-1]
    error.update({modelstr+' train mse':mse})
    error.update({modelstr+' train mae':mae})
    error.update({modelstr+' test mse':score[0]})
    error.update({modelstr+' test mae':score[1]})    
    return y_hat

yhat6=error_data(history6,model6,'model6')
yhat5=error_data(history5,model5,'model5')
yhat4=error_data(history4,model4,'model4')
yhat3=error_data(history3,model3,'model3')
yhat2=error_data(history2,model2,'model2')
yhat1=error_data(history1,model1,'model1')

e=pd.DataFrame(error,index=['Key']).T
e.to_csv('D:\\stuff\\class\\DL\\Deep_Learning_Mid-term_Programming_Exam_20200501\\new data\\model_error.csv')
                      
#draw history
def draw(history1):
    plt.plot(history1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#export model
from sklearn.externals import joblib
path='D:\\stuff\\class\\DL\\Deep_Learning_Mid-term_Programming_Exam_20200501\\new data\\'
joblib.dump(model1, path+'model1.pkl')
joblib.dump(model2, path+'model2.pkl')
joblib.dump(model3, path+'model3.pkl')
joblib.dump(model4, path+'model4.pkl')
joblib.dump(model5, path+'model5.pkl')
joblib.dump(model6, path+'model6.pkl')



    









 




            
  

    
