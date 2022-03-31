#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
last update: 05/27/2021

"""




from __future__ import print_function
import os
# os.environ['KERAS_BACKEND']='tensorflow'
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf

import sys
sys.path.append('/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/')
import matplotlib
matplotlib.use('agg')
import numpy as np
import csv
import h5py
import time
import shutil
# from .EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from gMLPhase.EqT_utils import generate_arrays_from_file, picker
np.warnings.filterwarnings('ignore')
import datetime
from tqdm import tqdm
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
from gMLPhase.gMLP_torch import gMLPmodel
from EdgeConv.trainerEQT import Graphmodel

from torch import nn
import torch
import pytorch_lightning as pl
from torch.utils import mkldnn as mkldnn_utils
from torch.utils.data import DataLoader, Dataset
from EdgeConv.DataGeneratorMulti import DataGeneratorMulti

from tqdm import tqdm
import seisbench.models as sbm

import obspy
from glob import glob
import pickle

from matplotlib import pyplot as plt 
import random
import pandas as pd
import gc 

def _trim_nan(x):
        """
        Removes all starting and trailing nan values from a 1D array and returns the new array and the number of NaNs
        removed per side.
        """
        mask_forward = np.cumprod(np.isnan(x)).astype(
            bool
        )  # cumprod will be one until the first non-Nan value
        x = x[~mask_forward]
        mask_backward = np.cumprod(np.isnan(x)[::-1])[::-1].astype(
            bool
        )  # Double reverse for a backwards cumprod
        x = x[~mask_backward]

        return x, np.sum(mask_forward.astype(int)), np.sum(mask_backward.astype(int))
    
output_name ='/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/Try_EQT_2'

# gnn_weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/Try_EQT_2/checkpoints/epoch=38-step=27845.ckpt'
gnn_weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/Try_EQT_2/checkpoints/epoch=72-step=52121.ckpt'

# save_dir = os.path.join(os.getcwd(), output_name)
# save_figs = os.path.join(save_dir, 'figures')

# if os.path.isdir(save_figs):
#     shutil.rmtree(save_figs)  
# os.makedirs(save_figs) 

# print('Loading the model ...', flush=True)        
df=pd.read_csv('station.csv')
edge_index= torch.load('edge_index.pt')
print(edge_index)

# load model
sig = nn.Sigmoid()
eqt_model = sbm.EQTransformer.from_pretrained("original")
eqt_model.eval()
gnn_model = Graphmodel(eqt_model)
state = torch.load(gnn_weight_path)['state_dict']
gnn_model.load_state_dict( state_dict = state)
gnn_model.eval()

# date_folder = ['20201030','20201031']+ [str(i) for i in range(20201101,20201106)]
date_folder =  [str(i) for i in range(20201116,20201131)]


for days in date_folder[:]:
    print(days)
    for hour in range(1,25):

        date= days +'/'+str(hour)+'h'
        directory = os.path.join('pred2', date)
        if not os.path.exists(directory):
            os.makedirs(directory)

        bg_st=obspy.core.Stream()
        print('reading waveforms')
        for net,sta in tqdm(zip(df['network'],df['station'])): 
            bg_st += obspy.read(os.path.join('conti_data',date,net+'_'+sta+'.mseed'))

        print(bg_st,len(bg_st))

        split_points=list(range(3,len(bg_st),3))
        preds = []
        starts = []
        in_samples = 6000
        overlap = 1800
        steps = in_samples - overlap
        trace_stats=[]
        for ii in list(range(0,len(bg_st),3)):
            tr = bg_st[ii]
            trace_stats.append(tr.stats)

        print('predicting')

        for index, windowed_st in enumerate(tqdm(bg_st.slide(window_length=60.0, step=steps/100.0,include_partial_windows=False))): 
            s = index*steps#windowed_st[0].stats.starttime
            if len(windowed_st)!=len(bg_st):
                print('skip')
                preds.append(np.zeros((len(df),2,6000)))
                starts.append(s)
                continue
            start_trim = min([tr.stats.starttime for tr in windowed_st])
            end_trim = max([tr.stats.endtime for tr in windowed_st])
            windowed_st.trim(start_trim, max(start_trim+60, end_trim) , pad=True, fill_value=0)
            temp=np.array(windowed_st)[:,:6000]
            temp = np.split(temp,split_points)
            temp = np.array(temp,dtype=float)
            temp = temp - np.mean(temp,axis=-1,keepdims=True)
            std_data = np.std(temp,axis=-1,keepdims=True)
            std_data[std_data == 0] = 1
            temp = temp/std_data
            temp = torch.tensor(temp, dtype=torch.float)
            res=torch.sigmoid(gnn_model.forward((temp,None,edge_index)))
            preds.append(res.detach().numpy())
            starts.append(s)

        del bg_st
        gc.collect()

        print(preds[0].shape)

        prediction_sample_factor=1
        # Maximum number of predictions covering a point
        coverage = int(
            np.ceil(in_samples / (in_samples - overlap) + 1)
        )
        print('coverage',coverage)
        pred_length = int(
            np.ceil(
                (np.max(starts)+in_samples) * prediction_sample_factor
            )
        )
        pred_merge = (
            np.zeros_like(
                preds[0], shape=( preds[0].shape[0],preds[0].shape[1], pred_length, coverage)
            )
            * np.nan
        )

        for i, (pred, start) in enumerate(zip(preds, starts)):
                        pred_start = int(start * prediction_sample_factor)
                        pred_merge[
                           :,:, pred_start : pred_start + pred.shape[2], i % coverage
                        ] = pred
        print('pred_merge',pred_merge.shape)

        del preds
        gc.collect()

        # del pred_merge

        label_name=["GNN_P","GNN_S"]


        pred_rate =100
        print('writing to mseed')
        import warnings
        # with warnings.catch_warnings():
        #     warnings.filterwarnings(
        #         action="ignore", message="Mean of empty slice"
        #     )
        for k in range(len(df['station'])):
            print(k,trace_stats[k].network+'_'+trace_stats[k].station)
            output = obspy.Stream()
            pred = np.nanmean(pred_merge[k], axis=-1)
        #     trace_stats = bg_st[k].stats
            for i in range(2):

                trimmed_pred, f, _ = _trim_nan(pred[i])
                trimmed_start = trace_stats[k].starttime + f / pred_rate

                output.append(
                    obspy.Trace(
                        trimmed_pred,
                        {
                            "starttime": trimmed_start,
                            "sampling_rate": pred_rate,
                            "network": trace_stats[k].network,
                            "station": trace_stats[k].station,
                            "location": trace_stats[k].location,
                            "channel": label_name[i],
                        },
                    )
                )
            output.write(os.path.join(directory, trace_stats[k].network+'_'+trace_stats[k].station+'.mseed'), format='MSEED')
            del output
            gc.collect()
        del pred_merge
        gc.collect()
