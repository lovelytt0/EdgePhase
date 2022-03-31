import numpy as np
import os

download_list=np.load('continous_sta_download.npy')

for i in range(1,366):
    
    date = str(i).zfill(3)+'/'
    for sta in download_list[:]:
        cmd = 'aws s3 cp s3://scedc-pds/continuous_waveforms/2021/2021_' + date + ' continuous_waveforms/2021/2021_'+ date + ' --recursive --exclude \"*\" --include \"CI'+sta+'*\"' 
        print(cmd)
        os.system(cmd)
