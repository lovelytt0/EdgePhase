#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tianfeng
last update: 05/16/2022

"""

import numpy as np
import obspy
import os

from EdgeConv.Utils import _detect_peaks

base_folder='phase_30day'
df=pd.read_csv('station.csv')
date_folder = ['20201116']


for net, sta in zip(df['network'],df['station']):
    print(net,sta)
    for date in date_folder[:]:
        tmp_f = os.path.join(base_folder,date)
        if not os.path.exists(tmp_f):
            os.makedirs(tmp_f) 
        fileP=tmp_f+'/'+net+'.'+sta+'.P.txt'
        fileS=tmp_f+'/'+net+'.'+sta+'.S.txt'
        p_picks=[]
        s_picks=[]
        
        starttime = obspy.core.UTCDateTime(date, iso8601=True)  
        print(starttime)
        for hour in range(1,25):
            subfolder=date+'/'+str(hour)+'h'
            st = obspy.read('pred2/'+subfolder+'/'+net+'_'+sta+'.mseed')
            add_on = st[0].stats.starttime-starttime
            P_seq, S_seq = np.array(st[0]), np.array(st[1])
            p=_detect_peaks(P_seq, mph=0.05, mpd=1)
            s=_detect_peaks(S_seq, mph=0.05, mpd=1)

            for i in p:      
                p_picks.append([i/100.0+add_on,P_seq[i],0])
            for i in s:
                s_picks.append([i/100.0+add_on,S_seq[i],0])

        np.savetxt(fileP,np.array(p_picks))
        np.savetxt(fileS,np.array(s_picks))
