#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 05/16/2022
"""

from EdgeConv.trainerBaselineA import trainerBaselineA

training_generator = trainerBaselineA(input_hdf5='../gMLP_phase/Dataset_builder',
        input_trainset= '../gMLP_phase/Dataset_builder/training.npy',
        input_validset = '../gMLP_phase/Dataset_builder/validation.npy'  ,                    
        output_name='model_baselineA', 
        augmentation = True,
        normalization_mode='std',
        batch_size = 1,
        epochs=200,
        monitor='val_loss',
        patience=15)