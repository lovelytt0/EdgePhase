#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tianfeng
last update: 05/16/2022

"""

import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os
import pandas as pd


download_folder = 'Data/'

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

df=pd.read_csv('station.csv')
download_start = UTCDateTime("2020-11-16")
days = 1

for day in range(days):
    tmp=download_start+3600*24*day
    print('Downloading',tmp,'-',tmp+3600*24)
    folder=str(tmp).split('T')[0].replace("-", "")
    print(folder)
    try:
        os.mkdir(download_folder+folder)
    except OSError:
        print ("*Creation of the directory %s failed" % folder)
    for i in range(len(df['network'])):
        filling=df['channel'][i][:2]+'*'
        try:
            st=Client(df['client'][i]).get_waveforms(df['network'][i],df['station'][i], "*", filling ,tmp, tmp+3600*24).merge(method=1, fill_value='interpolate')

            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()
            if st[0].meta.sampling_rate!=100.0:
                print('HELP',i,df['network'][i]+'.'+df['station'][i])
            st.write(download_folder+folder+'/'+df['network'][i]+'.'+df['station'][i]+'.mseed',format="MSEED")

        except:
            print('BUG!',i,df['network'][i]+'.'+df['station'][i])
