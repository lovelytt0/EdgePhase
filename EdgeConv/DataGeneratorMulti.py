#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 03/31/2022
"""
from __future__ import division, print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import os

from obspy.signal.trigger import trigger_onset
import torch
from torch import nn
from obspy.geodetics.base import locations2degrees
from itertools import chain
import pickle
import random




class DataGeneratorMulti(torch.utils.data.Dataset):
    
    """ 
    
    Keras generator with preprocessing 
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Name of hdf5 file containing waveforms data.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32
        Batch size.
           
            
    shuffle: bool, default=True
        Shuffeling the list.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'.
            
    label_type: str, default=gaussian 
        Labeling type: 'gaussian', 'triangle', or 'box'.
             
    augmentation: bool, default=True
        If True, half of each batch will be augmented version of the other half.
            


    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection, P, and S respectively.
    
    """   
    
    def __init__(self, 
                 list_IDs, 
                 file_name='Dataset_builder', 
                 dim = 6000, 
                 batch_size=1, 
                 shuffle=True, 
                 norm_mode = 'std',
                 label_type = 'triangle',                 
                 augmentation = False,                  
                 ):
       
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_name = file_name        
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.label_type = label_type       
        self.augmentation = augmentation   

    def __len__(self):
        'Denotes the number of batches per epoch'
#         if self.augmentation:
#             return 2*int(np.floor(len(self.list_IDs) ))
#         else:
#         return int(np.floor(len(self.list_IDs)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]   
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y, edge_index = self._data_generation(self.list_IDs[index])


        return torch.tensor(X, dtype=torch.float),torch.tensor(y, dtype=torch.float), torch.tensor(edge_index), self.list_IDs[index]
    
    def testitem(self, index):

        X, y, edge_index,select_keys = self._testdata_generation(self.list_IDs[index])


        return torch.tensor(X, dtype=torch.float),torch.tensor(y, dtype=torch.float), torch.tensor(edge_index), self.list_IDs[index],select_keys
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)  
    
    def _normalize(self, data, mode = 'max'):  
        'Normalize waveforms in each batch'
        
        data -= np.mean(data)
        if mode == 'max':
            max_data = np.max(data)
#             assert(max_data.shape[-1] == data.shape[-1])
#             max_data[max_data == 0] = 1
            if max_data == 0:
                max_data = 1
            data /= max_data              

        elif mode == 'std':               
            std_data = np.std(data)
#             assert(std_data.shape[-1] == data.shape[-1])
            if std_data == 0:
                std_data = 1
#             std_data[std_data == 0] = 1
            data /= std_data
        return data
    
    def _label(self, a=0, b=20, c=40):  
        'Used for triangolar labeling'
        
        z = np.linspace(a, c, num = 2*(b-a)+1)
        y = np.zeros(z.shape)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half]-a) / (b-a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c-z[second_half]) / (c-b)
        return y

    def _data_generation(self, readfile):

        try:
            with open(os.path.join(self.file_name,readfile), 'rb') as handle:
                dataset= pickle.load(handle)
        except:
            print('error')
            print(readfile)
             
        if self.augmentation:
            key_nb = 32 if len(dataset) > 32 else len(dataset)
            select_keys = random.sample(list(dataset.keys()),key_nb)
        else:
            key_nb = len(dataset)
            select_keys=list(dataset.keys())
        X = np.zeros(( key_nb, 3, self.dim))
        y = np.zeros(( key_nb, 2, self.dim))
        loc = np.zeros((key_nb,2))

        # event labels
        if readfile.split('/')[1] == 'noises':
            for idx, sta in enumerate(select_keys):

                loc[idx,0] = dataset[sta]['lat']
                loc[idx,1] = dataset[sta]['lon']
                waves = dataset[sta]['waveforms']
                for wave in waves:

                    chan = wave.stats.channel[-1]
                    length = min(self.dim, len(wave))
                    data = np.array(wave)
                    data = self._normalize(data, self.norm_mode)  
                    if chan == 'E' or chan == '1':
                        X[idx, 0, :length] = data[:length]
                    if  chan == 'N' or chan == '2':
                        X[idx, 1, :length] = data[:length]
                    if  chan == 'Z':
                        X[idx, 2, :length] = data[:length]
        else:
            for idx, sta in enumerate(select_keys):

                loc[idx,0] = dataset[sta]['lat']
                loc[idx,1] = dataset[sta]['lon']
                waves = dataset[sta]['waveforms']
                for wave in waves:
                    chan = wave.stats.channel[-1]
                    length = min(self.dim, len(wave))
                    data = np.array(wave)
                    data = self._normalize(data, self.norm_mode)  
                    if chan == 'E' or chan == '1':
                        X[idx, 0, :length] = data[:length]
                    if  chan == 'N' or chan == '2':
                        X[idx, 1, :length] = data[:length]
                    if  chan == 'Z':
                        X[idx, 2, :length] = data[:length]

                ps = dataset[sta]['phase_p']   
                ss = dataset[sta]['phase_s']
                # create Y
                if len(ps) > 0:
                    try:
                        spt = int((ps[0] - wave.stats.starttime)*100)
    #                     print(spt)
                        if spt and (spt-20 >= 0) and (spt+21 < self.dim):
                            y[idx,0, spt-20:spt+21] = self._label()
                        elif spt and (spt+21 < self.dim):
                                y[idx,0, 0:spt+spt+1] = self._label(a=0, b=spt, c=2*spt)
                        elif spt and (spt-20 >= 0):
                            pdif = self.dim - spt
                            y[idx,0, spt-pdif-1:self.dim] = self._label(a=spt-pdif, b=spt, c=2*pdif)
                    except:
                        pass
                if len(ss) > 0:
                    try:
                        sst = int((ss[0] - wave.stats.starttime)*100)
    #                     print(sst)
                        if sst and (sst-20 >= 0) and (sst+21 < self.dim):
                            y[idx,1, sst-20:sst+21] = self._label()
                        elif sst and (sst+21 < self.dim):
                            y[idx, 1,0:sst+sst+1] = self._label(a=0, b=sst, c=2*sst)
                        elif sst and (sst-20 >= 0):
                            sdif = self.dim - sst
                            y[idx,1, sst-sdif-1:self.dim] = self._label(a=sst-sdif, b=sst, c=2*sdif)   
                    except:
                        pass


        row_a=[]
        row_b=[]  
        for i in range(key_nb):
            for j in range(key_nb):
                dis = locations2degrees(loc[i,0],loc[i,1],loc[j,0],loc[j,1])
                if dis < 1:
                    row_a.append(i)
                    row_b.append(j)
        edge_index=[row_a,row_b]

        return X, y, edge_index


    def _testdata_generation(self, readfile):

        try:
            with open(os.path.join(self.file_name,readfile), 'rb') as handle:
                dataset= pickle.load(handle)
        except:
            print('error')
            print(readfile)

        key_nb = len(dataset)
        select_keys=list(dataset.keys())
        X = np.zeros(( key_nb, 3, self.dim))
        y = np.full(( key_nb, 3),-99.99)
        loc = np.zeros((key_nb,2))
        # event labels
        if readfile.split('/')[1] == 'noises':
            for idx, sta in enumerate(select_keys):

                loc[idx,0] = dataset[sta]['lat']
                loc[idx,1] = dataset[sta]['lon']
                waves = dataset[sta]['waveforms']
                for wave in waves:

                    chan = wave.stats.channel[-1]
                    length = min(self.dim, len(wave))
                    data = np.array(wave)
                    data = self._normalize(data, self.norm_mode)  
                    if chan == 'E' or chan == '1':
                        X[idx, 0, :length] = data[:length]
                    if  chan == 'N' or chan == '2':
                        X[idx, 1, :length] = data[:length]
                    if  chan == 'Z':
                        X[idx, 2, :length] = data[:length]
        else:
            for idx, sta in enumerate(select_keys):

                loc[idx,0] = dataset[sta]['lat']
                loc[idx,1] = dataset[sta]['lon']
                waves = dataset[sta]['waveforms']
                for wave in waves:
                    chan = wave.stats.channel[-1]
                    length = min(self.dim, len(wave))
                    data = np.array(wave)
                    data = self._normalize(data, self.norm_mode)  
                    if chan == 'E' or chan == '1':
                        X[idx, 0, :length] = data[:length]
                    if  chan == 'N' or chan == '2':
                        X[idx, 1, :length] = data[:length]
                    if  chan == 'Z':
                        X[idx, 2, :length] = data[:length]

                ps = dataset[sta]['phase_p']   
                ss = dataset[sta]['phase_s']
                if len(ps)>0:
                    y[idx,0] = ps[0]-wave.stats.starttime
                else:
                    y[idx,0] = 0.0
                if len(ss)>0:
                    y[idx,1]= ss[0]-wave.stats.starttime
                else:
                    y[idx,1] = 0.0


                waveform=np.array(waves).T
                if waveform.ndim!=2:
                      continue
                spt =int(y[idx,0]*100)
                sst= int(y[idx,1]*100)
                if spt>0 and sst>0 and sst<waveform.shape[0]:
                    s1=np.percentile(abs(waveform[sst:min(sst+300,waveform.shape[0])]), 95, axis=0)
                    p1=np.percentile(abs(waveform[max(0,spt-500):spt-100]), 95, axis=0)
                    y[idx,2]=np.average(10*np.log10(s1**2/p1**2))
                    
                    


        row_a=[]
        row_b=[]  
        for i in range(key_nb):
            for j in range(key_nb):
                dis = locations2degrees(loc[i,0],loc[i,1],loc[j,0],loc[j,1])
                if dis < 1:
                    row_a.append(i)
                    row_b.append(j)
        edge_index=[row_a,row_b]

        return X, y, edge_index,select_keys

