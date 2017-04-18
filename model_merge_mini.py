# example python script for loading spikefinder data
#
# for more info see https://github.com/codeneuro/spikefinder
#
# requires numpy, pandas, matplotlib
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

calcium_train = []
spikes_train = []
ids = []

for dataset in range(10):
    calcium_train.append(np.array(pd.read_csv('spikefinder.train/'+str(dataset+1) + '.train.calcium.csv')))
    spikes_train.append(np.array(pd.read_csv('spikefinder.train/'+str(dataset+1) + '.train.spikes.csv')))
    ids.append(np.array([dataset]*calcium_train[-1].shape[1]))

maxlen = max([c.shape[0] for c in calcium_train])
calcium_train_padded = np.hstack([np.pad(c,((0,maxlen-c.shape[0]),(0,0)),'constant',constant_values=np.nan) for c in calcium_train])
spikes_train_padded = np.hstack([np.pad(c,((0,maxlen-c.shape[0]),(0,0)),'constant',constant_values=np.nan) for c in spikes_train])
ids_stacked = np.hstack(ids)
sample_weight = 1.+3*(ids_stacked<5)
sample_weight /= sample_weight.mean()
calcium_train_padded[spikes_train_padded<-1] = np.nan
spikes_train_padded[spikes_train_padded<-1] = np.nan
weights = 1.-(np.isnan(spikes_train_padded)).T.astype(np.float)

calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
spikes_train_padded[np.isnan(spikes_train_padded)] = -1.

calcium_train_padded = calcium_train_padded.T[:,:,np.newaxis]
spikes_train_padded = spikes_train_padded.T[:,:,np.newaxis]

ids_onehot = np.zeros((calcium_train_padded.shape[0],calcium_train_padded.shape[1],10))
for n,i in enumerate(ids_stacked):
    ids_onehot[n,:,i] = 1.
data_train = np.concatenate((calcium_train_padded,ids_onehot),2)

from keras.models import Sequential, Model
from keras.layers.core import Masking
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization

from keras import backend as K
import tensorflow as tf


main_input = Input(shape=(None,1), name='main_input')
dataset_input = Input(shape=(None,10), name='dataset_input')
x = Conv1D(10,500,padding='same',input_shape=(None,1))(main_input)
x = Activation('tanh')(x)
x = Dropout(0.1)(x)
x = Conv1D(10,5,padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.1)(x)
x = Conv1D(10,5,padding='same')(x)
x = Activation('relu')(x)
x = Dropout(0.1)(x)
x = Conv1D(1,5,padding='same')(x)
output = Activation('sigmoid')(x)

model = Model(inputs=[main_input,dataset_input],outputs=output)

#model = Sequential()
#model.add(Conv1D(10,300,padding='same',input_shape=(None,1)))
#model.add(Activation('tanh'))
#model.add(Dropout(0.1))
#model.add(Conv1D(10,30,padding='same'))
#model.add(Activation('relu'))
#model.add(Dropout(0.1))
#model.add(Conv1D(10,30,padding='same'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.1))
#model.add(Conv1D(10,30,padding='same'))
#model.add(Activation('tanh'))
#model.add(Dropout(0.1))
#model.add(Conv1D(1,30,padding='same'))
#model.add(Activation('sigmoid'))

def pearson_corr(y_true, y_pred,
        pool=True):
    """Calculates Pearson correlation as a metric.
    This calculates Pearson correlation the way that the competition calculates
    it (as integer values).
    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    if pool:
        y_true = pool1d(y_true, length=4)
        y_pred = pool1d(y_pred, length=4)

    mask = tf.to_float(y_true>=0.)
    samples = K.sum(mask,axis=1,keepdims=True)
    x_mean = y_true - K.sum(mask*y_true, axis=1, keepdims=True)/samples
    y_mean = y_pred - K.sum(mask*y_pred, axis=1, keepdims=True)/samples

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean * mask, axis=1)
    d = (K.sum(K.square(x_mean)* mask, axis=1) *
         K.sum(K.square(y_mean)* mask, axis=1))

    return 1.-K.mean(n / (K.sqrt(d) + 1e-12))

def pool1d(x, length=4):
    """Adds groups of `length` over the time dimension in x.
    Args:
        x: 3D Tensor with shape (batch_size, time_dim, feature_dim).
        length: the pool length.
    Returns:
        3D Tensor with shape (batch_size, time_dim // length, feature_dim).
    """

    x = tf.expand_dims(x, -1)  # Add "channel" dimension.
    avg_pool = tf.nn.avg_pool(x,
        ksize=(1, length, 1, 1),
        strides=(1, length, 1, 1),
        padding='SAME')
    x = tf.squeeze(avg_pool, axis=-1)

    return x * length

model.compile(loss=pearson_corr,optimizer='adam')
#model.fit(data_train,spikes_train_padded, epochs=150, batch_size=5, validation_split=0.2)
model.fit([calcium_train_padded,ids_onehot],spikes_train_padded, epochs=500, batch_size=5, validation_split=0.2,sample_weight=sample_weight)
model.save_weights('convnet_mini')
