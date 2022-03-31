import pandas as pd
import numpy as np
import os
import obspy
from obspy.core.utcdatetime import  UTCDateTime
import pickle
from tqdm import tqdm



df_ori = pd.read_csv('earthquake_catalogs/2021_catalog_index.csv')
df = df_ori[:][(pd.notna(df_ori['MS_FILENAME']))& (pd.notna(df_ori['PHASE_FILENAME']))]
df.index= range(len(df['MS_FILENAME']))
path = './'

def read_phase(phase_file):
    ph_list={}
    with open(phase_file) as f:
        lines = f.readlines()
        for line in lines:
                tmp = line.split()
                # event
                if len(tmp) == 10:
                    event_time, event_id = tmp[3],tmp[0]
                # phase
                elif len(tmp) == 13:
                    net, sta, lat, lon, dep, phase_type, delay = tmp[0],tmp[1],tmp[4],tmp[5], tmp[6], tmp[7], tmp[12]
                    if net == 'CI':
                        if not sta in ph_list:
                            ph_list[sta] = {'lat':float(lat), 'lon':float(lon), 'dep': float(dep), 'phase_p':[], 'phase_s':[]}
                        if phase_type=='P':
                            ph_list[sta]['phase_p'].append(UTCDateTime(event_time)+float(delay))
                        elif phase_type=='S':
                            ph_list[sta]['phase_s'].append(UTCDateTime(event_time)+float(delay))
                        else:
                            pass

                else:
                    pass
    return ph_list


for index, row in tqdm(df.iterrows()):
#     if index > 4:
#         break
    phase_file =  os.path.join(path, 'event_phases',row['PREFIX'],row['PHASE_FILENAME'])
    ms_file =  os.path.join(path, 'event_waveforms',row['PREFIX'],row['MS_FILENAME'])
    directory = os.path.join(path, 'Dataset','events',row['PREFIX'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    out_file_name = os.path.join(directory,row['PHASE_FILENAME'].split('.')[0]+'.pkl')
#     if os.path.exists(out_file_name):
#         print('skip')
#         continue
#     print(phase_file)
    ph_list=read_phase(phase_file)
#     print(ph_list)
#     print(ms_file)
    st_temp=obspy.read(ms_file)
    try:
        st_temp.merge(fill_value=0)
    except:
        continue
    for tr in st_temp:
        if tr.meta.station not in ph_list.keys():
            st_temp.remove(tr)

    
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
    if not ph_list or len(st)<1:
        continue
    start_trim = max([tr.stats.starttime for tr in st])
    end_trim = min([tr.stats.endtime for tr in st])
    st.trim(start_trim, max(start_trim+60, end_trim) , pad=True, fill_value=0)
    
    outfile = {}
    
    for tr in st:
        chan = tr.meta.channel[:2]
        sta = tr.meta.station
        key = sta+'_' + chan
        if key not in outfile.keys():
            outfile[key]=ph_list[sta].copy()
            outfile[key]['waveforms']=[tr]
        else:
            outfile[key]['waveforms'].append(tr) 
    del_keys = []
    for key in outfile.keys():
        if len(outfile[key]['waveforms'])<2:
            del_keys.append(key)
    for key in del_keys:
        outfile.pop(key, None)
    if len(outfile)>5:
        with open(out_file_name, 'wb') as handle:
            pickle.dump(outfile,handle)
