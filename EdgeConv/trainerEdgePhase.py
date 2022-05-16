#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tianfeng
last update: 03/31/2022
"""
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import time
from EdgeConv.DataGeneratorMulti import DataGeneratorMulti
from torch import nn

from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from pytorch_lightning import loggers as pl_loggers

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
from torch import nn
from torch_geometric.nn import MessagePassing
import seisbench.models as sbm

from tqdm import tqdm
def cycle(loader):
    while True:
        for data in loader:
            yield data
            
def conv_block(n_in, n_out, k, stride ,padding, activation, dropout=0):
    if activation:
        return nn.Sequential(
            nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding),
            activation,
            nn.Dropout(p=dropout),

        )
    else:
        return nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding)

def trainerEdgePhase(input_hdf5=None,
            input_trainset = None,
            input_validset = None,
            output_name = None, 
            input_dimention=(6000, 3),
            shuffle=True, 
            label_type='triangle',
            normalization_mode='std',
            augmentation=True,              
            batch_size=1,
            epochs=200, 
            monitor='val_loss',
            patience=3):
        


    args = {
    "input_hdf5": input_hdf5,
    "input_trainset": input_trainset,
    "input_validset":  input_validset,
    "output_name": output_name,
    "input_dimention": input_dimention,
    "shuffle": shuffle,
    "label_type": label_type,
    "normalization_mode": normalization_mode,
    "augmentation": augmentation,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience                    
    }
    
    save_dir, save_models=_make_dir(args['output_name'])
    training = np.load(input_trainset)
    validation = np.load(input_validset)
    start_training = time.time()

    params_training = {'file_name': str(args['input_hdf5']), 
                      'dim': args['input_dimention'][0],
                      'batch_size': 1,
                      'shuffle': args['shuffle'],  
                      'norm_mode': args['normalization_mode'],
                      'label_type': args['label_type'],
                      'augmentation': args['augmentation']}

    params_validation = {'file_name': str(args['input_hdf5']),  
                         'dim': args['input_dimention'][0],
                         'batch_size': 1,
                         'shuffle': False,  
                         'norm_mode': args['normalization_mode'],
                         'label_type': args['label_type'],
                         'augmentation': False}     



# change into eval mode

#     model.eval() 
    model = sbm.EQTransformer.from_pretrained("original")
    model_GNN = Graphmodel(pre_model=model)


    training_generator = DataGeneratorMulti(list_IDs=training, **params_training)
    validation_generator = DataGeneratorMulti(list_IDs=validation, **params_validation) 
    
#     for i in tqdm(range(len(training))):
#         training_generator.__getitem__(i)
        
#     return 
#     x=training_generator.__getitem__(6207)
#     print(x)
#     for k in x:
#         print(k.shape)
#         print(torch.sum(k))
    
#     return 
    checkpoint_callback = ModelCheckpoint(monitor=monitor,dirpath=save_models,save_top_k=3,verbose=True,save_last=True)
    early_stopping = EarlyStopping(monitor=monitor,patience=args['patience']) # patience=3
    tb_logger = pl_loggers.TensorBoardLogger(save_dir)

    trainer = pl.Trainer(precision=16, gpus=1, gradient_clip_val=0.5, accumulate_grad_batches=32, callbacks=[early_stopping, checkpoint_callback],check_val_every_n_epoch=1,profiler="simple",num_sanity_val_steps=0, logger =tb_logger)

    train_loader  = DataLoader(training_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)

    val_loader   = DataLoader(validation_generator, batch_size = args['batch_size'], num_workers=8, pin_memory=True, prefetch_factor=5)
    
    
    
    print('Started training in generator mode ...') 

    trainer.fit(model_GNN, train_dataloaders = train_loader, val_dataloaders = val_loader)

    end_training = time.time()  
    print('Finished Training')
        
        
    
def _make_dir(output_name):
    
    """ 
    
    Make the output directories.
    Parameters
    ----------
    output_name: str
        Name of the output directory.
                   
    Returns
    -------   
    save_dir: str
        Full path to the output directory.
        
    save_models: str
        Full path to the model directory. 
        
    """   
    
    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name))
        save_models = os.path.join(save_dir, 'checkpoints')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
        shutil.copyfile('EdgeConv/trainerMulti.py',os.path.join(save_dir,'trainerEQT.py'))
    return save_dir, save_models






class EdgeConv(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr='max',node_dim=-3) #  "Max" aggregation.
        activation= nn.GELU() 
        dropout=0.1


        self.deconv1 = conv_block(in_channels*2, in_channels*2, 3, 1, 1, activation=activation, dropout=0.1)
        self.deconv2 = conv_block(in_channels*2, in_channels*2, 3, 1, 1, activation=activation,dropout=0.1)
        self.deconv3 = conv_block(in_channels*2, in_channels, 3, 1, 1,activation=activation,dropout=0.1)
        self.deconv4 = conv_block(in_channels, in_channels, 3, 1, 1,  activation=activation,dropout=0.1)
        self.deconv5 = conv_block(in_channels, in_channels, 3, 1, 1, activation=activation,dropout=0.1)

#         self.gMLPmessage = nn.ModuleList([Residual(PreNorm(in_channels*2, gMLPBlock(dim = in_channels*2, heads = 1, dim_ff = in_channels*2, seq_len = 47, attn_dim = None, causal = False))) for i in range(1)])
#         self.proj_out = nn.Linear(in_channels*2, in_channels)
#         self.conv3 = conv_block(in_channels, out_channels, 3, 1, 1, activation)
    

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

#         tmp = torch.cat([x_i, x_j], dim=2)  # tmp has shape [E, 2 * in_channels]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
#         tmp = nn.Sequential(*self.gMLPmessage)(tmp)
#         return self.proj_out(tmp)
        ans = self.deconv5(self.deconv4(self.deconv3(self.deconv2(self.deconv1(tmp)))))
        return ans
    
    
    
    





class Graphmodel(pl.LightningModule):
    def __init__(
        self,
        pre_model
    ):
        super().__init__()
        
        self.edgeconv1 = EdgeConv(16)
#         for name, p in pre_model.named_parameters():
#             if "encoder" in name  or "gMLPlayers" in name :
#                 p.requires_grad = False
#                 print(name)
#             else:
#                 p.requires_grad = True
        
        self.encoder = pre_model.encoder
        self.res_cnn_stack = pre_model.res_cnn_stack
        self.bi_lstm_stack = pre_model.bi_lstm_stack
        self.transformer_d0 = pre_model.transformer_d0
        self.transformer_d = pre_model.transformer_d
        self.pick_lstms= pre_model.pick_lstms
        self.pick_attentions = pre_model.pick_attentions
        self.pick_decoders = pre_model.pick_decoders
        self.pick_convs = pre_model.pick_convs 
        self.dropout= pre_model.dropout 
        self.criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
        self.loss_weights = [0.5,0.5]
        print('loss weight', self.loss_weights)
        


    def forward(self, data):
        x, edge_index = data[0], data[2]
        x = np.squeeze(x)
        edge_index=np.squeeze(edge_index)
        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)
        x = self.edgeconv1(x,edge_index)

        # Detection part
#         detection = self.decoder_d(x)
#         detection = torch.sigmoid(self.conv_d(detection))
#         detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

#         outputs = [detection]
        
        # Pick parts
        lstm = self.pick_lstms[0]
        attention = self.pick_attentions[0]
        decoder =self.pick_decoders[0]
        conv = self.pick_convs[0]
        px = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        px = lstm(px)[0]
        px = self.dropout(px)
        px = px.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        px, _ = attention(px)
        px = decoder(px)
#             pred = torch.sigmoid(conv(px))
        x_P = conv(px)

        #x_P = pred #torch.squeeze(pred, dim=1)  # Remove channel dimension
            
            
        lstm = self.pick_lstms[1]
        attention = self.pick_attentions[1]
        decoder =self.pick_decoders[1]
        conv = self.pick_convs[1]
        px = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        px = lstm(px)[0]
        px = self.dropout(px)
        px = px.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        px, _ = attention(px)
        px = decoder(px)
#             pred = torch.sigmoid(conv(px))
        x_S = conv(px)

        out = torch.cat((x_P,x_S), 1 )
        #torch.squeeze(pred, dim=1)  # Remove channel dimension
        
        return out

    def out_feature_map(self, data):
        x, edge_index = data[0], data[2]
        x = np.squeeze(x)
        edge_index=np.squeeze(edge_index)
        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)
        
        x2 = self.edgeconv1(x,edge_index)
        
        return x, x2
    
    
    def training_step(self, batch, batch_idx):
    
        y = batch[1][0]
        y = np.squeeze(y)
        
        y_hat = self.forward(batch)
        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]
        
        loss = lossP+lossS

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_lossP", lossP, on_epoch=True, prog_bar=True)
        self.log("train_lossS", lossS, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch[1][0]
        y = np.squeeze(y)

        y_hat = self.forward(batch)
        
        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]
        
        loss = lossP+lossS
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_lossP", lossP, on_epoch=True, prog_bar=True)
        self.log("val_lossS", lossS, on_epoch=True, prog_bar=True)

        return {'val_loss': loss}
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "frequency": 5000
                       },
        
        }
    

