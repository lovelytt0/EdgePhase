from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
import pandas as pd
import obspy
import os

def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind







def pick_picks(ts,tt,min_proba_p,min_prob_s,sample_rate):
    
    
    pp_arr = _detect_peaks(yh2, mph=args['P_threshold'], mpd=1)
    ss_arr = _detect_peaks(yh3, mph=args['S_threshold'], mpd=1)
    
    prob_S = ts[:,1]
    prob_P = ts[:,0]
    prob_N = ts[:,2]
    itp = detect_peaks(prob_P, mph=min_proba_p, mpd=0.5*sample_rate, show=False)
    its = detect_peaks(prob_S, mph=min_prob_s, mpd=0.5*sample_rate, show=False)
    p_picks=tt[itp]
    s_picks=tt[its]
    p_prob=ts[itp]
    s_prob=ts[its]
    return p_picks,s_picks,p_prob,s_prob


base_folder='phase_30day'
df=pd.read_csv('station.csv')
date_folder = ['20201030','20201031']+ [str(i) for i in range(20201101,20201131)]

for net, sta in zip(df['network'],df['station']):
    print(net,sta)
    if sta =='EAG2':
        print ('skip',sta)
        continue
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
#             p_lib=p/100.0+add_on
#             s_lib=s/100.0+add_on

            for i in p:      
                p_picks.append([i/100.0+add_on,P_seq[i],0])
            for i in s:
                s_picks.append([i/100.0+add_on,S_seq[i],0])

        np.savetxt(fileP,np.array(p_picks))
        np.savetxt(fileS,np.array(s_picks))
