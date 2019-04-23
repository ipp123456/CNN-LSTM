# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:07:41 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.layers.core import Dense, Dropout,Activation
from keras.layers import Conv1D, MaxPooling1D,BatchNormalization,Flatten,Reshape,TimeDistributed,Lambda,Permute,Bidirectional
from keras.models import Model
from keras.layers.recurrent import LSTM 
from keras import callbacks
from keras.layers import Input
from sklearn import preprocessing
from Layer_one import MyLayer_one
from keras import backend as K


def get_sample2(path):
    data = pd.read_excel(path,index=None)
    p1 = np.array(data.iloc[:,2:18])
    p2 = np.array(data.iloc[:,19:])
    p3 = np.array(data.iloc[:,18]).reshape(len(p2),1)
    all_parament = np.concatenate((p1,p2,p3),axis=-1)
    return all_parament
def min_max(Parament):
    min_max_scaler = preprocessing.MinMaxScaler()
    all_parament = min_max_scaler.fit_transform(Parament)
    parament = all_parament[:,:-1]
    labels = all_parament[:,-1]
    length = parament.shape[0]-21
    data_set = np.zeros((length,21,parament.shape[-1]))
    data_lab = np.zeros((length,1))
    for i in range(length):
        data_set[i]=parament[i:i+21][:]
        data_lab[i] = labels[i+21]
    return data_set,data_lab  
  
parament1 = get_sample2('data/0401-2063A.xlsx')
parament2 = get_sample2('data/0401-2063B.xlsx')
parament3 = get_sample2('data/0401-2063C.xlsx')

Parament = np.concatenate((parament1,parament2,parament3))
Dataset,Lab = min_max(Parament)
train_data = Dataset[:-1000][:][:]
train_lab = Lab[:-1000][:]

x_train,x_val,y_train,y_val = train_test_split(train_data,train_lab,test_size=0.2)
#
x_test = Dataset[-1000:][:][:]
y_test = Lab[-1000:][:]

print('Defining a Simple Keras Model...')
Mydot = Lambda(lambda x: K.batch_dot(x[0],x[1]))

input_shape = x_train.shape[1:]

inputs = Input(shape=input_shape)

conv1 = Conv1D(32, 3)(inputs)
print('conv1:',np.shape(conv1))
conv1_a =Activation('relu')(conv1)
conv1_pool =MaxPooling1D(2)(conv1_a)
print('conv1_pool:',np.shape(conv1_pool))

conv2 = Conv1D(64, 3)(conv1_pool)
print('conv2:',np.shape(conv2))
conv2_a =Activation('relu')(conv2)
conv2_pool =MaxPooling1D(3)(conv2_a)
print('conv2_pool:',np.shape(conv2_pool))

cnn_out = Flatten()(conv2_pool)
cnn_out = Dense(128,activation='relu')(cnn_out)
cnn_out = Dropout(0.3)(cnn_out)
cnn_out = Dense(21,activation='relu')(cnn_out)
cnn_out = Reshape((21,1))(cnn_out)


lstm = LSTM(units=50,return_sequences=True)(inputs)
lstm_hid = TimeDistributed(Dense(50, activation='tanh'))(lstm)
print('lstm_hid:',np.shape(lstm_hid))
con_att = MyLayer_one()([lstm_hid,cnn_out])
con_att = Activation('softmax')(con_att)
con_att = Permute([2, 1])(con_att)
print('con_att:',np.shape(con_att))
att_con_mul = Mydot([con_att, lstm_hid])
print('att_con_mul:',np.shape(att_con_mul))
att_done = Flatten()(att_con_mul)

output = Dense(128)(att_done)
output = Dropout(0.3)(output)
output = Dense(1)(output)
output = Activation('sigmoid')(output)
model = Model(inputs=inputs, outputs=output)
print('Compiling the Model...')
model.compile(loss='binary_crossentropy',
              optimizer='adam',metrics=['accuracy'])
print("Train...")
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
saveBestModel = callbacks.ModelCheckpoint('model_1/lscn_model_1_2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=16, epochs=100,verbose=1, 
          validation_data=(x_val, y_val),callbacks=[earlyStopping, saveBestModel])

score = model.predict(x_test)
save_data = pd.DataFrame(y_test, columns=['True'])
save_data['Predict'] = score
save_data.to_excel('output_1/lscn_1_predict_out_y_2.xlsx')