import obspy
import glob
import numpy as np
import os
import tqdm
import pickle

download_list=np.load('continous_sta_download.npy')

sta_dict={}
for sta in download_list:
    inv=obspy.read_inventory("StationList/CI_"+sta+'.xml', format="STATIONXML")
    lat, lon, dep= inv[0][0].latitude, inv[0][0].longitude, inv[0][0].elevation
    sta_dict[sta] = {'lat':float(lat), 'lon':float(lon), 'dep': float(dep)}

    
    
# for folder in sorted(glob.glob('noise/2021/*')):
for day in range(1, 366):
    folder = 'noise/2021/2021_'+str(day).zfill(3)
    tmp_name = folder.split('/')
    directory = os.path.join('Dataset/noises',tmp_name[1],tmp_name[2])
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    files=glob.glob(folder+'/*')
    print(folder)
    event_set = set()
    for file in files:
        event_set.add(file.split('/')[-1].split('_')[1])
    for index, event in enumerate(event_set):  
#             print(folder+'/*'+event)
        st_temp = obspy.read(folder+'/*'+event)
#             print(st)
        
        try:
            st_temp.merge(fill_value=0)
        except:
            'merge'
            pass
            
        st = obspy.core.Stream()

        for tr in st_temp:
            tr.detrend('demean')
            st += tr

        try:
            st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
        except:
            'filter'
            pass
        st.taper(max_percentage=0.001, type='cosine', max_length=2) 
        if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
            try:
                st.interpolate(100, method="linear")
            except Exception:
                st=_resampling(st) 

        for index, windowed_st in enumerate(st.slide(window_length=60.0, step=60.0,include_partial_windows=False)):
            outfile = {}
            out_file_name = os.path.join(directory,event.split('.')[0]+'_'+str(index).zfill(3)+'.pkl')
            for tr in windowed_st:
                chan = tr.meta.channel[:2]
                sta = tr.meta.station
                key = sta+'_' + chan
                if key not in outfile.keys():
                    outfile[key]=sta_dict[sta].copy()
                    outfile[key]['waveforms']=[tr]
                else:
                    outfile[key]['waveforms'].append(tr) 
            del_keys = []
            for key in outfile.keys():
                if len(outfile[key]['waveforms'])<2:
                    del_keys.append(key)
            for key in del_keys:
                outfile.pop(key, None)  
                
            with open(out_file_name, 'wb') as handle:
                pickle.dump(outfile,handle)