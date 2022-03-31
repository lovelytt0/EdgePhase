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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
import pickle


# def _plotter(X, y, yP, yS, save_figs, evi ):
#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(311)         
#     plt.plot(X[0], 'gray',alpha=0.7)
#     plt.rcParams["figure.figsize"] = (8,5)
#     legend_properties = {'weight':'bold'}  
#     plt.title(str(evi))
#     plt.tight_layout()

#     x = np.linspace(0, len(X[0]),len(X[0]), endpoint=True)

                            
#     ax = fig.add_subplot(312)   
#     plt.title('label')
#     plt.plot(x, y[0], 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
#     plt.plot(x, y[1], 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
#     plt.tight_layout()       
#     plt.ylim((-0.1, 1.1))

                
#     ax = fig.add_subplot(313)
#     plt.title('predict')
#     plt.plot(x, yP, 'b--', alpha = 0.5, linewidth=1.5, label='P_probability')
#     plt.plot(x, yS, 'r--', alpha = 0.5, linewidth=1.5, label='S_probability')
#     plt.tight_layout()       
#     plt.ylim((-0.1, 1.1))
#     plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
#     fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.png')) 
    
def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def pick_picks(ts,tt,min_proba_p,min_prob_s,sample_rate):
    
    
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
    
    prob_S = ts[:,1]
    prob_P = ts[:,0]
    prob_N = ts[:,2]
    itp = detect_peaks(prob_P, mph=min_proba_p, mpd=0.5*sample_rate, show=False)
    its = detect_peaks(prob_S, mph=min_prob_s, mpd=0.5*sample_rate, show=False)
    p_picks=tt[itp]
    s_picks=tt[its]
    p_prob=ts[itp]
    s_prob=ts[its]
    return p_picks,s_picks,p_prob,s_prob

    
output_name ='/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/Eqt_ori_result'
input_testset = 'Dataset_builder/test.npy'
gnn_weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/Try_EQT_2/checkpoints/epoch=72-step=52121.ckpt'


save_dir = os.path.join(os.getcwd(), output_name)


print('Loading the model ...', flush=True)        


test = np.load(input_testset)

params_test = {'file_name': 'Dataset_builder',  
                     'dim': 6000,
                     'batch_size': 1,
                     'n_channels': 3,
                     'shuffle': False,  
                    'coda_ratio': 1.4,
                     'norm_mode': 'std',
                    'label_type': 'triangle',
                     'augmentation': False}         


test_generator = DataGeneratorMulti(list_IDs=test,**params_test) 

eqt_model0 = sbm.EQTransformer.from_pretrained("original")
eqt_model0.eval()

gnn_model = Graphmodel(eqt_model0)
state = torch.load(gnn_weight_path)['state_dict']
gnn_model.load_state_dict( state_dict = state)
gnn_model.eval()

eqt_model1 = sbm.EQTransformer.from_pretrained("original")
eqt_model1.eval()


from EdgeConv.trainerEQTOri import Graphmodel

finetune_weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/EQT_ori/checkpoints/epoch=13-step=39941.ckpt'

finetune_model = Graphmodel(eqt_model1)
state2 = torch.load(finetune_weight_path)['state_dict']
finetune_model.load_state_dict( state_dict = state2)
finetune_model.eval()

# eqt_model = sbm.EQTransformer.from_pretrained("original")
# eqt_model.eval()



folder2 = os.path.join(save_dir,'results')

if not os.path.isdir(folder2):
    os.makedirs(folder2) 

res_p=[]
res_s=[]

subtitle=[
'EdgePhase',
'Baseline-A',

'Baseline-B',
    
]

th_P = [0.39, 0.64, 0.05]
th_S = [0.28, 0.4, 0.06]
 
for idx in [79]:
    X = test_generator.testitem(idx)
    print(X)
    sta_name = X[4]
    y_label = X[1].detach().numpy()
#     y_pre =  torch.sigmoid(gnn_model.forward(X)).detach().numpy()
    index=np.argsort(y_label,axis=0)
    print(len(sta_name))
    order = index[:,0]
    for pivot,temp in enumerate(order):
        if y_label[temp,0]!=0:
            break
    choice=range(pivot,len(order)-6,3)
    select_list = order[choice]
    y_size=1*(len(select_list))
    
    wave=X[0].detach().numpy()

    plt.figure(figsize=(8,y_size))
    x_seq = np.arange(0,60,0.01)

    scale =15
    for model_idx, model in enumerate([gnn_model]):
        
        if model_idx<2:
            y_pre = torch.sigmoid(model.forward(X)).detach().numpy()
            feature1, feature2 = model.out_feature_map(X)
            feature1 = feature1.detach().numpy()
            feature2 = feature2.detach().numpy()
#             feature2 = feature2/np.max(abs(feature2),axis=2)

        else:
            y_pre=  model.forward(X[0])
            P_pred = y_pre[1].detach().numpy()
            S_pred = y_pre[2].detach().numpy()

#         plt.subplot(2,1,1)
#         plt.plot(feature1[0])
#         plt.subplot(2,1,2)
#         plt.plot(feature2[0])
#         plt.title(subtitle[model_idx])
        
        for h,sta in enumerate(select_list[-4:-1]):
            if  model_idx<2:
                P_seq, S_seq = y_pre[sta,0], y_pre[sta,1]
            else:
                P_seq, S_seq = P_pred[sta], S_pred[sta]

            p=_detect_peaks(P_seq, mph=th_P[model_idx], mpd=1)
            s=_detect_peaks(S_seq, mph=th_S[model_idx], mpd=1)
            plt.plot(x_seq,wave[sta,2,:]/wave[sta,2,:].max()+scale*h,'gray',alpha=0.7,linewidth=0.5)

            p_label = y_label[sta,0]
            s_label = y_label[sta,1]
            p_label=p_label[p_label>0]
            s_label=s_label[s_label>0]

            plt.vlines(p_label,scale*h-0.2,scale*h+0.2,'r')
            
            plt.vlines(s_label,scale*h-0.2,scale*h+0.2,'b')
            if model_idx==0:
                plt.text(5.5, scale*h-1,sta_name[sta]+'Z')
                plt.text(5.5, scale*h-scale/2+0.4,'latent representations',color='b')
                plt.text(5.5, scale*h-scale+1.5,'enhanced representations ',color='b')

#             f1 = feature1[sta].T#/np.max(abs(feature1[sta]),axis=1)
            f1 = feature1[sta].T-np.min(feature1[sta],axis=1)
            f1 = f1/np.max(f1,axis=0)

#             f1 = f1[:,np.argsort(f1[10,:])]
            f1 = f1[:,np.argsort(np.mean(f1,axis=0))]

#             plt.plot(np.linspace(0,60,47),0.5*f1[:,:]+8*h-3,'gray')
            plt.imshow(f1[:,:].T, extent=(0, 60, scale*h-scale/2+0.4, scale*h-1.5),
           interpolation='nearest', cmap='hot')     
#             plt.colorbar()

            #f2 = feature2[sta].T#/np.max(abs(feature2[sta]),axis=1)
            f2 = feature2[sta].T-np.min(feature2[sta],axis=1)
            f2 = f2/np.max(f2,axis=0)
            f2 = f2[:,np.argsort(np.mean(f2,axis=0))]

            
#             plt.plot(np.linspace(0,60,47),0.5f2[:,:]+8*h-7,'blue')
            plt.imshow(f2[:,:].T, extent=(0, 60, scale*h-scale+1.5, scale*h-scale/2-0.4),
           interpolation='nearest', cmap='hot')
#             plt.colorbar()
        plt.yticks([])
        
        plt.xlim([5,45])
        plt.ylim([-scale,scale*h+1])

        plt.xlabel('Time(sec)')
        plt.tight_layout()

        plt.savefig('Example/'+str(idx)+'_feature.png')

