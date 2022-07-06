#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 05/16/2022
"""


from EdgeConv.trainerEdgePhase import trainerEdgePhase


training_generator = trainerEdgePhase(input_hdf5='../gMLP_phase/Dataset_builder',
        input_trainset= '../gMLP_phase/Dataset_builder/training.npy',
        input_validset = '../gMLP_phase/Dataset_builder/validation.npy'  ,                    
        output_name='model_edgephase_200km',      
        augmentation = True,
        normalization_mode='std',
        batch_size = 32,
        epochs=200,
        monitor='val_loss',
        patience=15,
        dis_th=2)