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
from EdgeConv.trainerEQTOri import Graphmodel

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

    
output_name ='/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/EQT_ori'
input_testset = 'Dataset_builder/test.npy'

gnn_weight_path = '/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/GNN/EQT_ori/checkpoints/epoch=13-step=39941.ckpt'

save_dir = os.path.join(os.getcwd(), output_name)
save_figs = os.path.join(save_dir, 'figures')

if os.path.isdir(save_figs):
    shutil.rmtree(save_figs)  
os.makedirs(save_figs) 

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

# start_training = time.time()

# sig = nn.Sigmoid()

eqt_model = sbm.EQTransformer.from_pretrained("original")
eqt_model.eval()

gnn_model = Graphmodel(eqt_model)
state = torch.load(gnn_weight_path)['state_dict']
gnn_model.load_state_dict( state_dict = state)
gnn_model.eval()


threshold =0.05
folder2 = os.path.join(save_dir,'results')

if not os.path.isdir(folder2):
    os.makedirs(folder2) 

res_p=[]
res_s=[]



# for idx in [46]:
#     X = test_generator.testitem(idx)
#     sta_name = X[4]
#     y_label = X[1].detach().numpy()
#     y_pre =  torch.sigmoid(gnn_model.forward(X)).detach().numpy()

#     wave=X[0].detach().numpy()
#     index=np.argsort(y_label,axis=0)
#     order = index[:,0]
#     for pivot,temp in enumerate(order):
#         if y_label[temp,0]!=0:
#             break
#     y_size=0.3*(len(order[pivot:]))
#     plt.figure(figsize=(6,y_size))
#     plt.title('Fine-tuned Eqtransformer without EdgeConv')
#     x_seq = np.arange(0,60,0.01)

#     for h,sta in enumerate(order[pivot:-6]):
#         P_seq, S_seq = y_pre[sta,0], y_pre[sta,1]
#         p=_detect_peaks(P_seq, mph=0.64, mpd=1)
#         s=_detect_peaks(S_seq, mph=0.4, mpd=1)
#         plt.plot(x_seq,wave[sta,2,:]/wave[sta,2,:].max()+3*h,'gray',alpha=0.7)

#         plt.plot(x_seq,P_seq+3*h-2,'orange')
#     #     plt.vlines(p,sta,sta+1,'r')
#         p_label = y_label[sta,0]
#         s_label = y_label[sta,1]
#         p_label=p_label[p_label>0]
#         s_label=s_label[s_label>0]

#         plt.vlines(p_label,3*h-0.6,3*h+0.6,'r')
#         plt.vlines(p/100.0,3*h-2,3*h-1,'r',alpha=0.6)

#         plt.plot(x_seq,S_seq+3*h-2,'c')
#     #     plt.vlines(s,sta,sta+1,'b')
#         plt.vlines(s_label,3*h-0.6,3*h+0.6,'b')
#         plt.vlines(s/100.0,3*h-2,3*h-1,'b',alpha=0.6)

#         plt.text(20, 3*h,sta_name[sta]+'Z')
#     plt.yticks([])
#     plt.xlim([20,45])
#     plt.ylim([-3,h*3+3])

#     plt.xlabel('Time(sec)')
#     plt.tight_layout()

#     plt.savefig('Example/'+str(idx)+'_no_edge.png')

for idx in tqdm(range(test_generator.__len__())):
# for idx in tqdm(range(3)):
    
    X = test_generator.testitem(idx)
    y_label = X[1].detach().numpy()
    y_pre =  torch.sigmoid(gnn_model.forward(X)).detach().numpy()
    
    
    for sta in range(len(y_pre)):
        P_seq, S_seq = y_pre[sta,0], y_pre[sta,1]
        p=_detect_peaks(P_seq, mph=threshold, mpd=1)
        s=_detect_peaks(S_seq, mph=threshold, mpd=1)
        
        target_p = y_label[sta,0]
        target_s = y_label[sta,1]
        
        snr = y_label[sta,2]

        res_p.append((np.array(p)/100,np.array(P_seq[p]),target_p,snr))
        res_s.append((np.array(s)/100,np.array(S_seq[s]),target_s,snr))

out_file_nameP= os.path.join(folder2,'p_res.pkl')
out_file_nameS= os.path.join(folder2,'s_res.pkl')

with open(out_file_nameP, 'wb') as handle:
    pickle.dump(res_p,handle)

with open(out_file_nameS, 'wb') as handle2:
    pickle.dump(res_s,handle2)

