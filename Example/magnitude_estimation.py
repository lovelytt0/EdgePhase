import numpy as np, pandas as pd,  matplotlib. pyplot as plt
from datetime import timedelta
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
from obspy.taup import TauPyModel
import time
# load data

df_Hypo = pd.read_csv("df_Hypo.csv")
df_Hypo.time=pd.to_datetime(df_Hypo.time)

paz_wa = {'sensitivity': 2800, 'zeros': [0j], 'gain': 1,
          'poles': [-6.2832 - 4.7124j, -6.2832 + 4.7124j]}

full_sta = pd.read_csv('../station.csv')
# valid_sta = np.load('valid_sta_index.npy') 
# np.save('valid_sta_index.npy',valid_sta)
model = TauPyModel(model="iasp91")


lat0=37.902    #37.897
lon0=26.794 #26.784
dis = 1
select_sta=full_sta[(full_sta['longitude']> lon0-dis) & (full_sta['longitude']< lon0+dis)
         & (full_sta['latitude']> lat0-dis) & (full_sta['latitude']< lat0+dis)]
select_sta.index=range(len(select_sta))

def cal_magnitude(download_start,event_lat,event_lon, event_dep, select_sta, model):
    mags =[]
    for i in range(len(select_sta)):
        # Calculate distance
        sta_lat = float(select_sta['latitude'][i])
        sta_lon = float(select_sta['longitude'][i])

        epi_dist, az, baz = gps2dist_azimuth(event_lat, event_lon, sta_lat, sta_lon)
        epi_dist = epi_dist / 1000
        # Calculate travel time
        
        try:
            arrivals = model.get_travel_times(source_depth_in_km = event_dep, distance_in_degree = epi_dist/111.1,  phase_list=["P","p"])
            travel_time = arrivals[0].time 
        except:
#             display('skip:',i)
            continue
        tmp = download_start + travel_time - 5
        # Download waveform and response
        filling=select_sta['channel'][i][:2]+'*'
#         start = time.time()

        try:
            st=Client(select_sta['client'][i]).get_waveforms(select_sta['network'][i],select_sta['station'][i], "*", filling ,tmp-30, tmp+80).merge(method=1, fill_value='interpolate')
            inv=Client(select_sta['client'][i]).get_stations(starttime=tmp-30, endtime=tmp+80,network=select_sta['network'][i],station=select_sta['station'][i],channel=filling,level="response")
            st.remove_response(inventory=inv)
            st.simulate(paz_remove=None, paz_simulate=paz_wa, water_level=10)
            st.trim(tmp, tmp + 50)
            tr_n = st.select(component="N")[0]
            ampl_n = max(abs(tr_n.data))
            tr_e = st.select(component="E")[0]
            ampl_e = max(abs(tr_e.data))
            ampl = max(ampl_n, ampl_e)
        except:
#             display('skip2:',i)
            continue
    
#         end = time.time()
        # total time taken
#         print(f"Runtime of the program is {end - start}")
        if epi_dist < 60:
            a = 0.018
            b = 2.17
        else:
            a = 0.0038
            b = 3.02
        ml = np.log10(ampl * 1000) + a * epi_dist + b
        mags.append(ml)
        
    net_mag = np.median(mags)
    print("Estimate magnitude:", net_mag)
    return net_mag
        
for i in tqdm(range(len(df_Hypo[:]))):
    download_start = UTCDateTime(df_Hypo.time[i])
    event_lat = df_Hypo.La[i]
    event_lon = df_Hypo.Lo[i]
    event_dep = max(df_Hypo.Dep[i],1)
    mg = cal_magnitude(download_start,event_lat,event_lon,event_dep, select_sta, model)
    df_Hypo['Mw'][i] = mg
    
    
df_Hypo.to_csv('df_Hypo_Mw3.csv')