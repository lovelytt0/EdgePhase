#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 03/31/2022
"""

from EdgeConv.trainerBaselineA import trainerBaselineA
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

training_generator = trainerBaselineA(input_hdf5='Dataset_builder',
        input_trainset= 'Dataset_builder/training.npy',
        input_validset = 'Dataset_builder/validation.npy'  ,                    
        output_name='GNN/EQT_ori', 
        augmentation = True,
        label_type='triangle',
        shift_event_r = None,
        normalization_mode='std',
        batch_size = 1,
        epochs=200,
        monitor='val_loss',
        patience=15)