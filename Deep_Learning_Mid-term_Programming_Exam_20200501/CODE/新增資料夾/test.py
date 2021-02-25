# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:41:44 2020

@author: User
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
data=pd.read_csv('air_reserve.csv')
data['visit_datetime'] = pd.to_datetime(data['visit_datetime'])
data['month']=data['visit_datetime'].dt.month
people_count=dict()
m=data['month'].unique()
for month in m:
    dg=data.groupby('month').get_group(month).sum().reserve_visitors 
    count={month:dg}
    people_count.update(count)
p=pd.DataFrame([people_count])
    
plt.plot(np.array(p.columns).reshape(12,),np.array(p.head(1).values).reshape(12,))      
    
print(people_count.keys) 