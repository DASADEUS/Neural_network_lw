# -*- coding: utf-8 -*-


import keras as k
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

uploaded = files.upload()

DATA=pd.read_csv('bestsellers with categories.csv', delimiter=',')

DATA

DATA['Year'].unique()

DATA['Genre'].unique()

len(DATA['Author'].unique())

DATA = DATA.drop('Name', 1)
DATA = DATA.drop('Price', 1)
Authors=DATA['Author'].unique()
maxr=max(DATA['Reviews'])
for i in range(len(DATA['Genre'])):
  if DATA['Genre'][i]=='Non Fiction': DATA['Genre'][i]=1
  else: DATA['Genre'][i]=0
for i in range(len(DATA['Author'])):
  DATA['Author'][i]=int(list(np.array(Authors)).index(DATA['Author'][i]))
best=[]
for i in range(len(DATA['Reviews'])):
  if (DATA['Reviews'][i]>12000)and(DATA['User Rating'][i]>4.5):
    best.append(1)
  else: best.append(0)
DATA.insert(loc=0, column='Blokbaster', value=best)

DATA.head()

def authblok(df):
  ba=[]
  for i in df['Author']:
    for j in range(len(df['Author'])):
      if df['Blokbaster'][j]==1:
        ba.append(df['Author'][j])
  return list(set(ba))

def blosnext(dfr,dfn):
  z=authblok(dfn)
  dfr['Blokbaster']=0
  for i in range(len(dfr['Author'])):
    if dfr['Author'][i] in z:
      dfr['Blokbaster'][i]=1
  return dfr

authblok(df12)

df09=DATA.groupby(["Year"]).get_group(2009)
df09=df09.reset_index(drop=True)
df10=DATA.groupby(["Year"]).get_group(2010)
df10=df10.reset_index(drop=True)
df11=DATA.groupby(["Year"]).get_group(2011)
df11=df11.reset_index(drop=True)
df12=DATA.groupby(["Year"]).get_group(2012)
df12=df12.reset_index(drop=True)
df13=DATA.groupby(["Year"]).get_group(2013)
df13=df13.reset_index(drop=True)
df14=DATA.groupby(["Year"]).get_group(2014)
df14=df14.reset_index(drop=True)
df15=DATA.groupby(["Year"]).get_group(2015)
df15=df15.reset_index(drop=True)
df16=DATA.groupby(["Year"]).get_group(2016)
df16=df16.reset_index(drop=True)
df17=DATA.groupby(["Year"]).get_group(2017)
df17=df17.reset_index(drop=True)
df18=DATA.groupby(["Year"]).get_group(2018)
df18=df18.reset_index(drop=True)
df19=DATA.groupby(["Year"]).get_group(2019)
df19=df19.reset_index(drop=True)

df09=blosnext(df09,df10)
df10=blosnext(df10,df11)
df11=blosnext(df11,df12)
df12=blosnext(df12,df13)
df13=blosnext(df13,df14)
df14=blosnext(df14,df15)
df15=blosnext(df15,df16)
df16=blosnext(df16,df17)
df17=blosnext(df17,df18)
df18=blosnext(df18,df19)

learning_data=df09.copy(deep=True)
learning_data=learning_data.append(df10, ignore_index=True)
learning_data=learning_data.append(df11, ignore_index=True)
learning_data=learning_data.append(df12, ignore_index=True)
learning_data=learning_data.append(df13, ignore_index=True)
learning_data=learning_data.append(df14, ignore_index=True)
learning_data=learning_data.append(df15, ignore_index=True)
learning_data=learning_data.append(df16, ignore_index=True)
learning_data=learning_data.append(df17, ignore_index=True)
learning_data=learning_data.append(df18, ignore_index=True)
learning_data= learning_data.drop('Year', 1)
learning_data['Reviews']/=100000

result_data=df19.copy(deep=True)
result_data=result_data.drop('Year', 1)
result_data=result_data.drop('Blokbaster', 1)
result_data['Reviews']/=100000
df19=df19.drop('Blokbaster', 1)

result_data

train_mass_input=[]
train_mass_output=[]
for i in range(len(learning_data['User Rating'])):
  train_mass_input.append([learning_data['User Rating'][i],learning_data['Reviews'][i],learning_data['Genre'][i]])
  train_mass_output.append(learning_data['Blokbaster'][i])
train_mass_input=np.array(train_mass_input,dtype=float)
train_mass_output=np.array(train_mass_output,dtype=float)

result_mass_input=[]
for i in range(len(result_data['User Rating'])):
  result_mass_input.append([result_data['User Rating'][i],result_data['Reviews'][i],result_data['Genre'][i]])
result_mass_input=np.array(result_mass_input,dtype=float)

train_mass_input

model=k.Sequential()
model.add(k.layers.Dense(units=5,activation="relu"))
model.add(k.layers.Dense(units=1,activation="relu"))
model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])
fit_results=model.fit(x=train_mass_input,y=train_mass_output, epochs=100, validation_split=0.2)

predicted_test=model.predict(result_mass_input)
result=df19
result['pred']=predicted_test
max(predicted_test)
print(result)