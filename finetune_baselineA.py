#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 05/16/2022
"""

from EdgeConv.trainerBaselineA import trainerBaselineA

training_generator = trainerBaselineA(input_hdf5='Dataset_builder',
        input_trainset= 'Dataset_builder/training.npy',
        input_validset = 'Dataset_builder/validation.npy'  ,                    
        output_name='GNN/EQT_ori', 
        augmentation = True,
        shift_event_r = None,
        normalization_mode='std',
        batch_size = 1,
        epochs=200,
        monitor='val_loss',
        patience=15)