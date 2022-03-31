from matplotlib import pyplot as plt
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read_inventory
from obspy import Stream
from obspy.taup import TauPyModel
from obspy.geodetics import base
from obspy import read, read_inventory
# from obspyh5 import iterh5,writeh5,readh5
from obspy.signal.trigger import classic_sta_lta,plot_trigger,pk_baer

import os
import pandas as pd
# import conda

# conda_file_dir = conda.__file__
# conda_dir = conda_file_dir.split('lib')[0]
# proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
# os.environ["PROJ_LIB"] = proj_lib

# from mpl_toolkits.basemap import Basemap


df=pd.read_csv('station.csv')

download_start = UTCDateTime("2020-11-16")
for day in range(15):
    tmp=download_start+3600*24*day
    print('Downloading',tmp,'-',tmp+3600*24)
    folder=str(tmp).split('T')[0].replace("-", "")
    print(folder)
    try:
        os.mkdir('Data2/'+folder)
    except OSError:
        print ("*Creation of the directory %s failed" % folder)
    for i in [38]:#range(len(df['network'])):
        filling=df['channel'][i][:2]+'*'
        try:
            st=Client(df['client'][i]).get_waveforms(df['network'][i],df['station'][i], "*", filling ,tmp, tmp+3600*24).merge(method=1, fill_value='interpolate')

            for tr in st:
                if isinstance(tr.data, np.ma.masked_array):
                    tr.data = tr.data.filled()
            if st[0].meta.sampling_rate!=100.0:
                print('HELP',i,df['network'][i]+'.'+df['station'][i])
            st.write('Data2/'+folder+'/'+df['network'][i]+'.'+df['station'][i]+'.mseed',format="MSEED")

        except:
            print('BUG!',i,df['network'][i]+'.'+df['station'][i])
