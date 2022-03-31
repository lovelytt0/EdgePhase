import pandas as pd
import numpy as np
import os
import obspy
from obspy.core.utcdatetime import  UTCDateTime
import pickle
from tqdm import tqdm
import gc 

df_ori = pd.read_csv('earthquake_catalogs/2021_catalog_index.csv')
path = './'


time_rg = []
trig_on = None
ahead = 5*60
after = 40*60
duration = 5*60
for index, row in (df_ori.iterrows()):
    time = UTCDateTime(row['ORIGIN_DATETIME'])
#     print(time)
    if not trig_on:
        trig_on = time + after
        continue
    else:
        if time - ahead - trig_on > duration:
            time_rg.append((trig_on, time-ahead))
#             print('range',trig_on, time - ahead, time-ahead-trig_on)
        trig_on = time + after
        
download_list=np.load('continous_sta_download.npy')
download_list = list(download_list[:10])+list(download_list[12:])


origin_t = UTCDateTime(2021,1,1)
for i in range(18, 366):
    cur = origin_t+(i-1)*24*3600
    rg_ls = []
    print(cur)
    directory='noise/2021/2021_'+str(i).zfill(3)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
    for rg in time_rg:
        if rg[0]> cur and rg[1] < cur+24*3600:
            rg_ls.append(rg)
    folder = 'continuous_waveforms/2021/2021_'+str(i).zfill(3)+'/'
    for sta in tqdm(download_list):
        gc.collect()
        try:
            st = obspy.read(folder+'CI'+sta+'*')
            st.merge(fill_value=0)
            for index, rg in enumerate(rg_ls):
                st_temp = st.copy().trim(starttime=rg[0], endtime=rg[1], pad=True, fill_value=0)
                outfile = os.path.join(directory,str(sta)+'_'+str(cur.year)+str(cur.month).zfill(2)+str(cur.day).zfill(2)+str(index).zfill(3)+'.mseed')
                st_temp.write(outfile)
        except:
            print('bug!')
            pass
        
        # cut pieces:


            