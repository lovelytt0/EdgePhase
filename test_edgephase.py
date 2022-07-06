#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 06/17/2022
"""

import os
import sys
sys.path.append('/home/tian_feng/UCLA/gMLP_phase/gMLP_phase/')
import numpy as np
np.warnings.filterwarnings('ignore')
from tqdm import tqdm
import torch

import seisbench.models as sbm
from EdgeConv.trainerEdgePhase import Graphmodel
from EdgeConv.DataGeneratorMulti import DataGeneratorMulti
from EdgeConv.Utils import _detect_peaks
import pickle

dis_th = 0.1
output_name ='./Test_diff_dist_200km'
input_testset = '../gMLP_phase/Dataset_builder/test.npy'
gnn_weight_path = './models/Edgephase_200km_step=24989.ckpt'

print('Loading the model ...', flush=True)        
test = np.load(input_testset)
params_test = {'file_name': '../gMLP_phase/Dataset_builder',  
                     'dim': 6000,
                     'batch_size': 1,
                     'shuffle': False,  
                     'norm_mode': 'std',
                     'augmentation': False,
                    'dis_th':dis_th}         

test_generator = DataGeneratorMulti(list_IDs=test,**params_test) 
eqt_model = sbm.EQTransformer.from_pretrained("original")
eqt_model.eval()
gnn_model = Graphmodel(eqt_model)
state = torch.load(gnn_weight_path)['state_dict']
#state = torch.load(gnn_weight_path, map_location=torch.device('cpu'))['state_dict']

gnn_model.load_state_dict( state_dict = state)
gnn_model.eval()


threshold =0.05
folder2 = os.path.join(output_name,str(int(dis_th*100)))

if not os.path.isdir(folder2):
    os.makedirs(folder2) 

res_p=[]
res_s=[]

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

