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
sample_weight = 1.+(ids_stacked<5)
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


