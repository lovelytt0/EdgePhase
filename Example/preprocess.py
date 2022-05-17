#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tianfeng
last update: 05/16/2022

"""

import pandas as pd
import os
import obspy
import tqdm

df=pd.read_csv('station.csv')

path = './'
datapath = os.path.join(path,'Data2')
date_folder =  [str(i) for i in range(20201116,20201131)]

miss_sta = set()
for date in date_folder[:]:
    directory = os.path.join('conti_data', date)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for net,sta in tqdm.tqdm(zip(df['network'],df['station'])): 
        
        st_temp = obspy.read(os.path.join(datapath,date,net+'.'+sta+'.mseed'))
        st_temp.sort()
        
        st = obspy.core.Stream()
        for tr in st_temp:
            tr.detrend('demean')
            st += tr
               
        st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
        st.taper(max_percentage=0.001, type='cosine', max_length=2) 
        if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
            try:
                st.interpolate(100, method="linear")
            except Exception:
                st=_resampling(st) 
        if len(st)!=3:
            print('error')
        start_trim = obspy.core.UTCDateTime(date[:4]+'-'+date[4:6]+'-'+date[6:]+'T00:00:00')
        end_trim = obspy.core.UTCDateTime(date[:4]+'-'+date[4:6]+'-'+date[6:]+'T23:59:59')
        st.trim(start_trim, end_trim+1 , pad=True, fill_value=0)
        for index, windowed_st in enumerate(tqdm.tqdm(st.slide(window_length=3600.0, step=3600.0,include_partial_windows=True))): 
            hourpath = os.path.join(directory,str(index+1)+'h')
            if not os.path.exists(hourpath):
                os.makedirs(hourpath)
            windowed_st.write(os.path.join(hourpath, net+'_'+sta+'.mseed'), format='MSEED') 
