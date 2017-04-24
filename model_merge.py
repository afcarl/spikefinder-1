# example python script for loading spikefinder data
#
# for more info see https://github.com/codeneuro/spikefinder
#
# requires numpy, pandas, matplotlib
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import AveragePooling1D
from keras.layers.core import Masking
from keras.layers.merge import Concatenate
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

from keras import backend as K
import tensorflow as tf

import simple_spearmint

def load_data():
    calcium_train = []
    spikes_train = []
    ids = []
    for dataset in range(10):
        calcium_train.append(np.array(pd.read_csv('spikefinder.train/'+str(dataset+1) + '.train.calcium.csv')))
        spikes_train.append(np.array(pd.read_csv('spikefinder.train/'+str(dataset+1) + '.train.spikes.csv')))
        ids.append(np.array([dataset]*calcium_train[-1].shape[1]))

    maxlen = max([c.shape[0] for c in calcium_train])
    calcium_train_padded = np.hstack(
        [np.pad(c,((0,maxlen-c.shape[0]),(0,0)),'constant',constant_values=np.nan) for c in calcium_train])
    spikes_train_padded = np.hstack(
        [np.pad(c,((0,maxlen-c.shape[0]),(0,0)),'constant',constant_values=np.nan) for c in spikes_train])

    ids_stacked = np.hstack(ids)

    sample_weight = 1.+1.5*(ids_stacked<5)
    sample_weight /= sample_weight.mean()

    calcium_train_padded[spikes_train_padded<-1] = np.nan
    spikes_train_padded[spikes_train_padded<-1] = np.nan
    calcium_train_padded[np.isnan(calcium_train_padded)] = 0.
    spikes_train_padded[np.isnan(spikes_train_padded)] = -1.

    calcium_train_padded = calcium_train_padded.T[:,:,np.newaxis]
    spikes_train_padded = spikes_train_padded.T[:,:,np.newaxis]

    ids_onehot = np.zeros((calcium_train_padded.shape[0],calcium_train_padded.shape[1],10))
    for n,i in enumerate(ids_stacked):
        ids_onehot[n,:,i] = 1.

    return calcium_train_padded,spikes_train_padded,ids_onehot, sample_weight

def build_model(params):
    main_input = Input(shape=(None,1), name='main_input')
    dataset_input = Input(shape=(None,10), name='dataset_input')
    x = Conv1D(10,params['input_width'],padding='same',input_shape=(None,1))(main_input)
    x = Activation(params['hidden_function'])(x)
    #x = Dropout(params['p_dropout'])(x)
    x = Conv1D(10,params['hidden_width'],padding='same')(x)
    x = Activation(params['hidden_function'])(x)
    x = Bidirectional(LSTM(10, return_sequences=True), merge_mode='concat', weights=None)(x)

    #x = Concatenate()([x,dataset_input])
    #x = Dropout(params['p_dropout'])(x)
    #x = Conv1D(10,1,padding='same')(x)
    #x = Activation(params['hidden_function'])(x)
    x = Dropout(params['p_dropout'])(x)
    x = Conv1D(10,params['hidden_width'],padding='same')(x)
    x = Activation(params['hidden_function'])(x)
    x = Dropout(params['p_dropout'])(x)
    x = Conv1D(10,params['hidden_width'],padding='same')(x)
    x = Activation(params['hidden_function'])(x)
    x = Dropout(params['p_dropout'])(x)
    x = Conv1D(1,params['hidden_width'],padding='same')(x)
    output = Activation('sigmoid')(x)

    return Model(inputs=[main_input,dataset_input],outputs=output)

def pearson_corr(y_true, y_pred,
        pool=True):
    """Calculates Pearson correlation as a metric.
    This calculates Pearson correlation the way that the competition calculates
    it (as integer values).
    y_true and y_pred have shape (batch_size, num_timesteps, 1).
    """

    if pool:
        y_true = 4.*AveragePooling1D(pool_size=4)(y_true)
        y_pred = 4.*AveragePooling1D(pool_size=4)(y_pred)

    mask = tf.to_float(y_true>=0.)
    samples = K.sum(mask,axis=1,keepdims=True)
    x_mean = y_true - K.sum(mask*y_true, axis=1, keepdims=True)/samples
    y_mean = y_pred - K.sum(mask*y_pred, axis=1, keepdims=True)/samples

    # Numerator and denominator.
    n = K.sum(x_mean * y_mean * mask, axis=1)
    d = (K.sum(K.square(x_mean)* mask, axis=1) *
         K.sum(K.square(y_mean)* mask, axis=1))

    return 1.-K.mean(n / (K.sqrt(d) + 1e-12))


calcium_train_padded,spikes_train_padded,ids_onehot, sample_weight = load_data()


def objective(params):
    #test_chunks = [0,11,32,45,51,60]
    #test_ids = [np.random.randint(test_chunks[i],test_chunks[i+1]) for i in range(5)]
    test_ids = np.random.choice(range(60),12,replace=False)
    train_ids = range(ids_onehot.shape[0])
    for i in test_ids:
        train_ids.remove(i)
    model = build_model(params)
    model.compile(loss=pearson_corr,optimizer='adam')
    ret = model.fit([calcium_train_padded[train_ids],ids_onehot[train_ids]],
              spikes_train_padded[train_ids],
              epochs=300,
              callbacks=[EarlyStopping(patience=10)],
              batch_size=5,
              sample_weight=sample_weight[train_ids],
             validation_data = (
                 [calcium_train_padded[test_ids],ids_onehot[test_ids]],spikes_train_padded[test_ids]))
    #model.save_weights('convnet')
    return ret.history['val_loss'][-1] 

# Define a parameter space
# Supported parameter types are 'int', 'float', and 'enum'
parameter_space = {'p_dropout': {'type': 'float', 'min': 0., 'max': 0.5},
                   'hidden_width': {'type': 'int', 'min': 2, 'max': 50},
                   'input_width': {'type': 'int', 'min': 100, 'max': 500},
                   'hidden_function': {'type': 'enum', 'options': ['relu', 'tanh']}}
# Create an optimizer
ss = simple_spearmint.SimpleSpearmint(parameter_space)

# Seed with 5 randomly chosen parameter settings
# (this step is optional, but can be beneficial)
for n in xrange(5):
    # Get random parameter settings
    suggestion = ss.suggest_random()
    # Retrieve an objective value for these parameters
    value = objective(suggestion)
    print "Random trial {}: {} -> {}".format(n + 1, suggestion, value)
    # Update the optimizer on the result
    ss.update(suggestion, value)

# Run for 100 hyperparameter optimization trials
for n in xrange(100):
    # Get a suggestion from the optimizer
    suggestion = ss.suggest()
    # Get an objective value; the ** syntax is equivalent to
    # the call to objective above
    value = objective(suggestion)
    print "GP trial {}: {} -> {}".format(n + 1, suggestion, value)
    # Update the optimizer on the result
    ss.update(suggestion, value)
    #best_parameters, best_objective = ss.get_best_parameters()
    #print "Best parameters {} for objective {}".format(
    #best_parameters, best_objective)
    ss.chooser.best()


