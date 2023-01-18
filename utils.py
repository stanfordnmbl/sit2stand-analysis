import os
import json
import numpy as np
import subprocess
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import interp1d, splrep, splev
import pandas as pd
from scipy import interpolate
import csv
import cv2
import pickle
from edits import tofix, tocheck, toremove

FPS = 30.0

NOSE = 0
NECK = 1
RSHO = 2
RELB = 3
RWRI = 4
LSHO = 5
LELB = 6
LWRI = 7
MHIP = 8
RHIP = 9
RKNE = 10
RANK = 11
LHIP = 12
LKNE = 13
LANK = 14
REYE = 15
LEYE = 16
REAR = 17
LEAR = 18
LBTO = 19
LSTO = 20
LHEL = 21
RBTO = 22
RSTO = 23
RHEL = 24
VERT = 25
LAH = 26 # Left ankle horizontal
RAH = 27 # Left ankle horizontal

def get_framerate(filepath, videoid):
    try:
        res = subprocess.check_output("ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {}".format(filepath).split(" "))
        res = tuple(map(float, res.decode().rstrip().split("/")))
        res = res[0]/res[1]
    except:
        print("{}".format(videoid))
        res = None
    return res
    #cam = cv2.VideoCapture(filepath)
    #return cam.get(cv2.CAP_PROP_FPS)

videometa = {}
    
if os.path.exists("videometa.pkl"):
    with open("videometa.pkl", 'rb') as file: 
        videometa = pickle.load(file)
else:
    with open("links.csv", "r") as f:
        csvFile = csv.reader(f)
 
        # displaying the contents of the CSV file
        for lines in csvFile:
            videoid = lines[0]
            filename = "videos/raw/" + lines[1].split("/")[-1]
            videometa[videoid] = {
                "framerate": get_framerate(filename, videoid),
                "filepath": lines[1],
                "videoid": lines[1].split("/")[-1][:-4]
            }
    with open("videometa.pkl", 'wb') as file: 
        pickle.dump(videometa, file)

# Convert OpenPose frames to a numpy array
def json2np(json_dir, subjectid):
    n = len(os.listdir(json_dir))
    res = np.zeros((n,75))
    for frame in range(n):
        test_image_json = '{}/input_{}_keypoints.json'.format(json_dir, str(frame).zfill(12))

        with open(test_image_json) as data_file:  
            data = json.load(data_file)

        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            xcoords = [keypoints[i] for i in range(len(keypoints)) if i % 3 == 0]
            counter = 0
            res[frame-1,:] = keypoints
            break

    return res


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def get_angle(A,B,C,data):
    """
    finds the angle ABC, assumes that confidence columns have been removed
    A,B and C are integers corresponding to different keypoints
    """
    p_A = np.array([data[:,3*A],data[:,3*A+1]]).T
    p_B = np.array([data[:,3*B],data[:,3*B+1]]).T
    p_C = np.array([data[:,3*C],data[:,3*C+1]]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
    dot_products = np.sum(p_BA*p_BC,axis=1)
    norm_products = np.linalg.norm(p_BA,axis=1)*np.linalg.norm(p_BC,axis=1)
    return np.arccos(dot_products/norm_products)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def smooth_ts(ts, framerate = 30, s = 0.01):
    ts = ts.tolist()
    x = range(len(ts))
    f = splrep(x, ts, s = s)
    return np.array(splev(x, f))

def smooth_ts(ts, framerate):
    return butter_lowpass_filter(ts, 6, framerate)

def get_keypoints3d(npz, framerate=30):
    if not os.path.exists(npz):
        return None
    with np.load(npz, allow_pickle=True) as data:
        res = data["results"][()]

    res3d_list = []
    for frame in range(len(res[0])):
        res3d_list.append(res[0][f"{frame:06}"]["j3d_op25"])
        
    kp3d = np.stack(res3d_list)
    for i in range(25):
        for j in range(3):
            kp3d[:,i,j] = smooth_ts(kp3d[:,i,j], framerate=framerate)#,s=0.1)

    return kp3d

SAVE_FIGS = False
MORE_PLOTS = True

def fill_nan(A):
    inds = np.arange(A.shape[0]) 
    good = np.where(np.isfinite(A))
    if(len(good[0]) <= 1):
        return A
   
    # linearly interpolate and then fill the extremes with the mean (relatively similar to)
    # what kalman does 
    f = interpolate.interp1d(inds[good], A[good],kind="linear",bounds_error=False,fill_value="extrapolate")
    B = np.where(np.isfinite(A),A,f(inds))
    B = np.where(np.isfinite(B),B,np.nanmean(B))
    return B

def mean_perc(ts):
    ts = ts[ts > np.percentile(ts,5)]
    ts = ts[ts < np.percentile(ts,95)]
    return np.mean(ts)

def center_ts(res):
    res.shape

    scale = (res[:,(NECK*3):(NECK*3+3)] - res[:,(MHIP*3):(MHIP*3+3)])[:,:2]
    scale = np.sqrt(np.sum(scale**2,axis=1))
    scale = mean_perc(scale)
    
    X = mean_perc(res[:,RANK*3])
    Y = mean_perc(res[:,RANK*3+1])
    
    for i in range(25):
        res[:,(i*3):(i*3+3)] = res[:,(i*3):(i*3+3)] - np.hstack([X,Y,0])[None,:]
    return res /scale #[:,None]

def plot_ts(res):
    # Features to plot for diagnostics
    PLOT_COLS = {
        "Left knee": LKNE,
        "Right knee": RKNE,
        "Left hip": LHIP,
        "Right hip": RHIP,
        "Nose": NOSE,
    }

    for name, col in PLOT_COLS.items():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.title(name,fontsize=24)
        plt.xlabel("frame",fontsize=17)
        plt.ylabel("position",fontsize=17)
        plt.plot(res[:,[col*3,]], linestyle="-", linewidth=2.5)
        plt.plot(res[:,[col*3+1,]], linestyle="-", linewidth=2.5)
        plt.legend(['x', 'y'],loc=1)

def get_angle_stats(A, B, C, res, breaks, framerate = 30, name = None, alternate = 0, breaks_alt=None):
    is3d = len(res.shape) == 3
    
    if is3d:
        name += "_3d"
        langle = get_angle3d(A, B, C, res)
    else:
        langle = get_angle(A, B, C, res)
        
    minv = []
    maxv = []
    vel = []
    acc = []
    vel_max = []
    vel_min = []
    acc_max = []
    acc_min = []
    diffs = []
    sds = []

#    langle = smooth_ts(langle)

    if name == "trunk_lean" and (alternate == 1) and MORE_PLOTS:
        plt.title("Right knee angle",fontsize=24)
        plt.xlabel("time [s]",fontsize=17)
        plt.ylabel("angle",fontsize=17)
        
        ts = langle #smooth_ts(langle)
        grid = [x/framerate for x in range(ts.shape[0])]
        
        plt.plot(grid, ts*180/np.pi, linestyle="-", linewidth=2.5)
        plt.title("Trunk lean" + (" (3D)" if is3d else ""),fontsize=24)
        for i in range(breaks.shape[0]):
            plt.axvline(x=breaks[i]/framerate,linewidth=2, color='g', linestyle="--")
            if i % 2 == 1:
                plt.axvspan(breaks[i-1]/framerate, breaks[i]/framerate, alpha=0.1, color='green')
        if SAVE_FIGS:
            plt.savefig("plots/hip-angle.pdf", bbox_inches='tight')
        plt.show()
        
        # Single hip angle
        plt.title("Trunk lean" + (" (3D)" if is3d else ""),fontsize=24)
        plt.xlabel("time [s]",fontsize=17)
        plt.ylabel("angle",fontsize=17)
        
        ts = langle #smooth_ts(langle)
        
        grid = [x/framerate for x in range(breaks[2] - breaks[1])]
        plt.plot(grid, ts[breaks[1]:breaks[2]]*180/np.pi, linestyle="-", linewidth=2.5)
#        plt.axvline(x=(breaks_alt[1] - breaks[1])/framerate, linewidth=2, color='g', linestyle="--")
        plt.axvline(x=(breaks[2] - breaks[1])/framerate, linewidth=2, color='r', linestyle="--")
        plt.axvline(x=(breaks[1] - breaks[1])/framerate, linewidth=2, color='r', linestyle="--")
        if SAVE_FIGS:
            plt.savefig("plots/single-hip-angle.pdf", bbox_inches='tight')
        plt.show()
        
    for i in range(len(breaks)-1):
        if (alternate==1) and i % 2 == 1:
            continue
        if (alternate==-1) and i % 2 == 0:
            continue
        lang = langle[breaks[i]:breaks[i+1]]
        y = (lang)*180/np.pi

        n = y.shape[0]

        minv.append(np.quantile(y, 0.05))
        maxv.append(np.quantile(y, 0.95))
            
        v = (y[1:n] - y[0:(n-1)])*framerate
        a = (v[1:(n-1)] - v[0:(n-2)])*framerate
        
        diffs.append( np.quantile(y, 0.95) - np.quantile(y, 0.05) )
        sds.append( np.std(y) )
        
        vel.append( np.median(v) )
        acc.append( np.median(a) )

        vel_max.append( np.quantile(v, 0.95) )
        acc_max.append( np.quantile(a, 0.95) )

        vel_min.append( np.quantile(v, 0.05) )
        acc_min.append( np.quantile(a, 0.05) )
    
    for i in range(len(breaks)-1):
        if (alternate==1) and i % 2 == 1:
            continue
        if (alternate==-1) and i % 2 == 0:
            continue
                
        nlen = breaks[i+1] - breaks[i]
        nfrom = int(breaks[i] - nlen*1/3)
        nto = int(breaks[i] + nlen*1/3)
            
        lang = langle[nfrom:nto]
            
        diffs.append( np.quantile(y, 0.95) - np.quantile(y, 0.05) )

    
    sts = ""
    if alternate == 1:
        sts = "_sit2stand"
    if alternate == -1:
        sts = "_stand2sit"

    return {
        "{}_range_mean{}".format(name,sts): np.mean(diffs),
        "{}_sd{}".format(name,sts): np.mean(sds),
        "{}_max{}".format(name,sts): max(maxv),
        "{}_min{}".format(name,sts): min(minv),
        "{}_max_mean{}".format(name,sts): np.array(maxv).mean(),
        "{}_min_mean{}".format(name,sts): np.array(minv).mean(),
        "{}_max_sd{}".format(name,sts): np.array(maxv).std(),
        "{}_min_sd{}".format(name,sts): np.array(minv).std(),
        "{}_ang_vel{}".format(name,sts): np.array(vel).mean(),
        "{}_ang_acc{}".format(name,sts): np.array(vel).mean(),
        "{}_max_ang_vel{}".format(name,sts): np.array(vel_max).mean(),
        "{}_max_ang_acc{}".format(name,sts): np.array(acc_max).mean(),
        "{}_min_ang_vel{}".format(name,sts): np.array(vel_min).mean(),
        "{}_min_ang_acc{}".format(name,sts): np.array(acc_min).mean(),
#        "{}_ts{}".format(name,sts): langle,
    }
    
def get_angles_results(res, breaks, framerate = 30, alternate = 0, breaks_alt = None):
    is3d = len(res.shape)==3
    
    res = res.copy()
    
    if is3d:
        vert = res[:,MHIP,:].copy()
        vert[:,1] = vert[:,1] + 10
    else:
        vert = res[:,(MHIP*3):(MHIP*3+3)].copy()
        vert[:,1] = vert[:,1] - 10

    if is3d:
        lah = res[:,LKNE,:].copy()
        lah[:,1] = res[:,LANK,1]
    
        rah = res[:,RKNE,:].copy()
        rah[:,1] = res[:,RANK,1]
    
        extras = np.stack([vert, lah, rah],axis=1)
        res = np.hstack([res.copy(), extras])
    else:
        orientation = res[breaks[0], LKNE*3] > res[breaks[0], RKNE*3]
    
        lah = res[:,(LANK*3):(LANK*3+3)].copy()
        lah[:,0] = lah[:,0] + orientation*10
    
        rah = res[:,(RANK*3):(RANK*3+3)].copy()
        rah[:,0] = rah[:,0] + orientation*10
    
        res = np.hstack([res.copy(), vert, lah, rah])
    
    results = {}
    results.update(get_angle_stats(LANK, LKNE, LHIP, res, breaks, name="left_knee", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(RANK, RKNE, RHIP, res, breaks, name="right_knee", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(NECK, LHIP, LKNE, res, breaks, name="left_hip", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(NECK, RHIP, RKNE, res, breaks, name="right_hip", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(LBTO, LANK, LKNE, res, breaks, name="left_ankle", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(RBTO, RANK, RKNE, res, breaks, name="right_ankle", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(VERT, MHIP, NECK, res, breaks, name="trunk_lean", framerate = framerate, alternate = alternate, breaks_alt = breaks_alt))
    results.update(get_angle_stats(LKNE, LANK, LAH, res, breaks, name="left_shank_angle", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(RKNE, RANK, RAH, res, breaks, name="right_shank_angle", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(NECK, RKNE, RANK, res, breaks, name="alignment", framerate = framerate, alternate = alternate))
    results.update(get_angle_stats(NECK, MHIP, 25, res, breaks, name="trunk"))
    
    return results

def get_time_results(res, breaks, framerate = 30, alternate=0):
    times = []
    speeds = []
    diffs = []

    last_time = []
    
    for i in range(len(breaks)-1):
        time = float((breaks[i+1] - breaks[i])/framerate)
        speed = 1/time
        
        if (alternate == 1) and i % 2 == 1:
            last_time = time
            continue
        if (alternate == -1) and i % 2 == 0:
            last_time = time
            continue

        times.append(time)
        speeds.append(speed)
        
        if i > 0:
            diffs.append(time-last_time)        
        last_time = time
        

    total_time = sum(times)
    
    sts = ""
    if alternate == 1:
        sts = "_sit2stand"
    if alternate == -1:
        sts = "_stand2sit"
        
    return {
        "n{}".format(sts): len(times),
        "time{}".format(sts): total_time,
        "time_diff{}".format(sts): np.array(diffs).mean(),
        "speed{}".format(sts): round(len(times)/total_time,2),
        "time_sd{}".format(sts): np.array(times).std(),
        "speed_sd{}".format(sts): np.array(speeds).std(),
    }

def get_joint_speed(joint, res):
    n = res.shape[0]
    return res[1:n,(joint*3):(joint*3+3)] - res[0:(n-1),(joint*3):(joint*3+3)]

def get_joint_speed3d(joint, res):
    n = res.shape[0]
    return res[1:n,joint,:] - res[0:(n-1),joint,:]

def get_static(res, down, up):
    is3d = len(res.shape) == 3
    is3d_str = "_3d" if is3d else ""
    
    if is3d:
        ank_dist = res[:,RANK,:] - res[:,LANK,:]
        ank_dist_mag = np.sqrt(np.sum(ank_dist**2, axis=1))

        knee_dist = res[:,RKNE,:] - res[:,LKNE,:]
        knee_dist_mag = np.sqrt(np.sum(knee_dist**2, axis=1))

        hip_dist = res[:,RHIP,:] - res[:,LHIP,:]
        hip_dist_mag = np.sqrt(np.sum(hip_dist**2, axis=1))

        height = (res[:,RANK,:] + res[:,LANK,:])/2 - res[:,NOSE,:]
        height_mag = np.sqrt(np.sum(height**2, axis=1))

        lkeee_angle = get_angle3d(LANK, LKNE, LHIP, res)*180/np.pi
        rkeee_angle = get_angle3d(RANK, RKNE, RHIP, res)*180/np.pi
    else:
        ank_dist = res[:,RANK*3:(RANK*3+2)] - res[:,LANK*3:(LANK*3+2)]
        ank_dist_mag = np.sqrt(np.sum(ank_dist**2, axis=1))

        knee_dist = res[:,RKNE*3:(RKNE*3+2)] - res[:,LKNE*3:(LKNE*3+2)]
        knee_dist_mag = np.sqrt(np.sum(knee_dist**2, axis=1))

        hip_dist = res[:,RHIP*3:(RHIP*3+2)] - res[:,LHIP*3:(LHIP*3+2)]
        hip_dist_mag = np.sqrt(np.sum(hip_dist**2, axis=1))

        height = (res[:,RANK*3:(RANK*3+2)] + res[:,LANK*3:(LANK*3+2)])/2 - res[:,NOSE*3:(NOSE*3+2)]
        height_mag = np.sqrt(np.sum(height**2, axis=1))

        lkeee_angle = get_angle(LANK, LKNE, LHIP, res)*180/np.pi
        rkeee_angle = get_angle(RANK, RKNE, RHIP, res)*180/np.pi
    
    return {
        "ank_to_hip_dist_sit"+is3d_str: (ank_dist_mag / hip_dist_mag)[down].mean(),
        "ank_to_hip_dist_stand"+is3d_str: (ank_dist_mag / hip_dist_mag)[up].mean(),
        "knee_to_hip_dist_sit"+is3d_str: (knee_dist_mag / hip_dist_mag)[down].mean(),
        "knee_to_hip_dist_stand"+is3d_str: (knee_dist_mag / hip_dist_mag)[up].mean(),
        "height"+is3d_str: height_mag[up].mean(),
        "lknee_angle_first_sit"+is3d_str: lkeee_angle[down[0]],
        "rknee_angle_first_sit"+is3d_str: rkeee_angle[down[0]],
        "lknee_angle_first_stand"+is3d_str: lkeee_angle[up[0]],
        "rknee_angle_first_stand"+is3d_str: rkeee_angle[up[0]],
    }

def get_speed_stats(joint, res, breaks, framerate=30, name="pelvic", alternate=False):
    is3d = len(res.shape) == 3
    if is3d:
        trunk_speed = get_joint_speed3d(joint, res)[:,0:3] * framerate
        name += "_3d"
    else:
        trunk_speed = get_joint_speed(joint, res)[:,0:2] * framerate

    trunk_speed_mag = np.sqrt(np.sum(trunk_speed**2, axis=1))
    
    if joint == MHIP and alternate and MORE_PLOTS:
        plt.title("Pelvic vertical velocity",fontsize=24)
        plt.xlabel("time (s)",fontsize=17)
        plt.ylabel("position",fontsize=17)
        
    
#        ts = smooth_ts(trunk_speed[:,1])
        ts = trunk_speed[:,1]
        grid = [x/framerate for x in range(ts.shape[0])]
        
        plt.plot(grid, ts, linestyle="-", linewidth=2.5)
        for i in range(breaks.shape[0]):
            plt.axvline(x=breaks[i]/framerate,linewidth=2, color='g', linestyle="--")
            if i % 2 == 1:
                plt.axvspan(breaks[i-1]/framerate, breaks[i]/framerate, alpha=0.1, color='green')
        if SAVE_FIGS:
            plt.savefig("plots/pelvic.pdf", bbox_inches='tight')
        plt.show()
    
    
    n = trunk_speed.shape[0]
    trunk_acc = (trunk_speed[1:n,:] - trunk_speed[0:(n-1),:]) * framerate
    trunk_acc_mag = np.sqrt(np.sum(trunk_acc**2, axis=1))
    
    if alternate !=0:
        slices = []
        for i in range(len(breaks)-1):
            if alternate == 1 and i % 2 == 1:
                continue
            if alternate == -1 and i % 2 == 0:
                continue
            slices += list(range(breaks[i], breaks[i+1]))
    else:
        slices = list(range(breaks[0], breaks[-1]))
    
    trunk_speed_mag = trunk_speed_mag[slices]
    trunk_speed_mag = trunk_speed_mag[trunk_speed_mag < np.percentile(trunk_speed_mag, 95)] # remove outliers
    
    trunk_acc_mag = np.append(trunk_acc_mag, trunk_acc_mag[-1])
    trunk_acc_mag = trunk_acc_mag[slices]
    trunk_acc_mag = trunk_acc_mag[trunk_acc_mag < np.percentile(trunk_acc_mag, 95)] # remove outliers
    
    sts = ""
    if alternate == 1:
        sts = "_sit2stand"
    if alternate == -1:
        sts = "_stand2sit"

    return {
        "{}_avg_speed{}".format(name,sts): np.median(trunk_speed_mag),
        "{}_min_speed{}".format(name,sts): np.quantile(trunk_speed_mag, 0.05),
        "{}_max_speed{}".format(name,sts): np.quantile(trunk_speed_mag, 0.95),
        "{}_avg_acc{}".format(name,sts): np.median(trunk_acc_mag),
        "{}_min_acc{}".format(name,sts): np.quantile(trunk_acc_mag, 0.05),
        "{}_max_acc{}".format(name,sts): np.quantile(trunk_acc_mag, 0.95),
        "{}_avg_y_speed{}".format(name,sts): np.median(trunk_speed[:,1]),
        "{}_min_y_speed{}".format(name,sts): np.quantile(trunk_speed[:,1], 0.05),
        "{}_max_y_speed{}".format(name,sts): np.quantile(trunk_speed[:,1], 0.95),
        "{}_avg_y_acc{}".format(name,sts): np.median(trunk_acc[:,1]),
        "{}_min_y_acc{}".format(name,sts): np.quantile(trunk_acc[:,1], 0.05),
        "{}_max_y_acc{}".format(name,sts): np.quantile(trunk_acc[:,1], 0.95),
    }
    
def get_acceleration_results(res, breaks, framerate=30, alternate=0):
    results = {}
    results.update(get_speed_stats(MHIP, res, breaks, name="pelvic", framerate = framerate, alternate=alternate))
    results.update(get_speed_stats(NECK, res, breaks, name="neck", framerate = framerate, alternate=alternate))
    return results

def get_angle(A,B,C,data):
    """
    finds the angle ABC, assumes that confidence columns have been removed
    A,B and C are integers corresponding to different keypoints
    """
    p_A = np.array([data[:,3*A],data[:,3*A+1]]).T
    p_B = np.array([data[:,3*B],data[:,3*B+1]]).T
    p_C = np.array([data[:,3*C],data[:,3*C+1]]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
   
    dot_products = np.sum(p_BA*p_BC,axis=1)
    det = np.sign(-p_BA[:,0]*p_BC[:,1] +p_BA[:,1]*p_BC[:,0])

    norm_products = np.abs(np.linalg.norm(p_BA,axis=1)*np.linalg.norm(p_BC,axis=1))

    
    M = dot_products.copy()
    M[np.abs(M)>1e-5] = (det[np.abs(M)>1e-5]*np.arccos(dot_products[np.abs(M)>1e-5]/norm_products[np.abs(M)>1e-5]))
    
    M[M < 0] = 2*np.pi + M[M < 0]
    return M

def get_angle3d(A,B,C,data):
    """
    finds the angle ABC, assumes that confidence columns have been removed
    A,B and C are integers corresponding to different keypoints
    """
    p_A = data[:,A,:]
    p_B = data[:,B,:]
    p_C = data[:,C,:]
    p_BA = p_A - p_B
    p_BC = p_C - p_B
   
    dot_products = np.sum(p_BA*p_BC,axis=1)
    det = np.sign(-p_BA[:,0]*p_BC[:,1] +p_BA[:,1]*p_BC[:,0])
    
    norm_products = np.abs(np.linalg.norm(p_BA,axis=1)*np.linalg.norm(p_BC,axis=1))
    
    M = dot_products.copy()
    M[np.abs(M)>1e-5] = -(det[np.abs(M)>1e-5]*np.arccos(dot_products[np.abs(M)>1e-5]/norm_products[np.abs(M)>1e-5]))
    
    M[M < 0] = 2*np.pi + M[M < 0]
    return M

def get_segments(res, magnitude = 1, magnitude_loc = 1, framerate = 30):
    nose_y = res[:,[NOSE*3+1,]]
    neck_y = res[:,[NECK*3+1,]]
    ind_y = (neck_y + nose_y)/2
    
    knee_angle = smooth_ts(get_angle(RANK, RKNE, RHIP, res), framerate = framerate)

    x=range(len(ind_y))
    f = splrep(x, ind_y, s=magnitude)
    ind_y_smooth = splev(x, f)
    
    vmax = np.quantile(ind_y,0.99) #ind_y.max()
    vmin = np.quantile(ind_y,0.01) #ind_y.min()
    
    vmax_knee = np.quantile(knee_angle, 0.97) #ind_y.max()
    vmin_knee = np.quantile(knee_angle, 0.03) #ind_y.min()
    
    plt.title("Peaks of the nose",fontsize=24)
    plt.xlabel("time [s]",fontsize=17)
    plt.ylabel("position",fontsize=17)

    grid = [x for x in range(ind_y.shape[0])]

    ups,downs = peakdet(ind_y_smooth, np.sqrt(magnitude)*(vmax - vmin)/2  )
#    shift = 5
#    ups,_ = peakdet(knee_angle[shift:], np.sqrt(magnitude)*(vmax_knee - vmin_knee)/2  )

    n = ups.shape[0]

    # TODO we can get a better estimate of breaks from multiple signals at once
    ups = ups[:,0].astype(np.uint16)
    ups.sort()
#    ups = ups# + shift
    
    downs = downs[:,0].astype(np.uint16)
    downs.sort()
    
    # Remove duplicates
    # downs = downs[np.append(downs[1:] - downs[:-1] > 5,True)]
    # ups = ups[np.append(ups[1:] - ups[:-1] > 5,True)]
    
    print(downs)
    print(ups)
    
    if (len(downs) <= 5 and len(ups) == 5) or (len(downs) <= 4 and len(ups) == 4):
        if max(ups) > max(downs):
            downs = np.append(downs, max(ups) + np.argmin(ind_y_smooth[max(ups):(max(ups) + ups[-1] - ups[-2])]))
        if min(ups) < min(downs):
            start_idx = int(min(ups) - (int(ups[1]) - int(ups[0]))/2)
            if start_idx < 0:# and start_idx >= -5:
                start_idx = 0
        
            if start_idx >= 0 and int(ups[0]) > start_idx:
                mina = int(np.argmin(ind_y_smooth[start_idx:int(ups[0])]))
                
                
                # check if new dip is really before the first peak (heuristic)
                if abs(start_idx + mina - min(ups)) > 10:
                    downs = np.concatenate([[start_idx + mina], downs.tolist()])
                
    for i in range(len(ups)-1):
        
        segment = ind_y_smooth[ups[i]:ups[i+1]]
        vmax = np.quantile(segment,0.99) #ind_y.max()
        vmin = np.quantile(segment,0.01) #ind_y.min()
        
        dd = None
        for j in range(len(downs)):
            if downs[j] > ups[i] and downs[j] < ups[i+1]:
                dd = j
                break

        _,loc_downs = peakdet(segment, np.sqrt(magnitude_loc)*(vmax - vmin)/12.5  )
        
        if loc_downs.shape[0]>=2:
            downs[dd] = ups[i] + loc_downs[-1,0]
        if loc_downs.shape[0]>2:
            print("ERROR, to mane dips")
           
#    if downs[-1] > ups[-1]:
#        downs = downs[:-1]
        
    print(downs)
    print(ups)
        
    scale = framerate
    
    plt.plot(np.array(grid)/scale, knee_angle, linestyle="-", linewidth=2.5)
    plt.plot(np.array(grid)/scale, ind_y, linestyle="-", linewidth=2.5)
    plt.plot(np.array(grid)/scale, ind_y_smooth, linestyle="-", linewidth=2.5)

    for i in range(ups.shape[0]):
        plt.axvline(x=ups[i]/scale,linewidth=2, color='g', linestyle="--")
    
    for i in range(downs.shape[0]):
        plt.axvline(x=downs[i]/scale,linewidth=2, color='r', linestyle="--")
        
    
    if SAVE_FIGS:
        plt.savefig("plots/nose.pdf", bbox_inches='tight')
    plt.show()
    
    return ups, downs

def swap_columns(res, A, B):
    tmp = res[:,(A*3):(A*3+3)].copy()
    res[:,(A*3):(A*3+3)] = res[:,(B*3):(B*3+3)]
    res[:,(B*3):(B*3+3)] = tmp
    
toswap = [
    [RSHO, LSHO],
    [RELB, LELB],
    [RWRI, LWRI],
    [RHIP, LHIP],
    [RKNE, LKNE],
    [RANK, LANK],
    [REYE, LEYE],
    [REAR, LEAR],
    [RHEL, LHEL],
    [RSTO, LSTO],
    [RBTO, LBTO],    
]

realign = {
    "5gtBtMlE": {1: 110},
    "e13bsM5a": {3: 205},
    "P5wlKd0H": {1: 82, 4: 283},
    "VPQJGG4D": {2: 160},
    "zSWVNAon": {1: 79},
    "eOBg4mwH": {0: 120},
    "5illwZ0w": {1: 118},
    "YuPB2PLf": {1: 710},
    "QFWxKvJ3": {1: 139},
    "Zj2jtm25": {1: 105},
    "fayp3GUT": {1: 143},
    "ztKJoXiw": {1: 320},
    "0nUjlcd7": {2: 200},
    "GESYi2xq": {0: 20},
    "ULS1fTmQ": {1: 120, 3: 320, 4: 420},
    "i9oVbbz6": {4: 550},
    "y3ET3wKE": {2: 160, 3: 235},
    "FsNM5n5s": {1: 50},
    "UDXlpEgF": {1: 85},
    "Ytlu6T69": {0: 115},
    "8dLqK0KT": {1: 65},
    "bFIh8shd": {2: 150},
    "GwvQ3hpI": {1: 120},
    "Je470A3u": {2: 160},
    "uQaqBtOs": {1: 115},
    "MfzjZe52": {1: 123, 2: 236},
    "RhRnfeBp": {2: 210},
    "a59COYtO": {1: 128, 2: 200, 3: 260},
    "GITsdVy7": {4: 280},
    "A5ya7RsN": {1: 105},
    "T3aA8TCd": {1: 100, 2: 160, 3: 225},
    "gXhE9VUw": {1: 75, 4: 260},
    "0jYyyP9R": {1: 200, 4: 650},
    "K7pXSGJ9": {1: 130, 2: 240},
    "o9xCf7YI": {2: 165},
    "ut7ckdyI": {1: 120, 2: 200, 4: 360},
    "k3YTjMU4": {1: 80, 4: 300},
    "HpYl7dTS": {1: 100, 3: 240},
    "W0BHMtXT": {1: 110, 3: 240},
    "oSFbRH4g": {3: 225},
    "2SV6hYB2": {1: 110, 2: 180},
}

def run_openpose(path, slug):
    os.makedirs("/tmp/openpose", exist_ok=True)
    keypoints_dir = "data/lab/keypoints/"
    os.makedirs(keypoints_dir, exist_ok=True)

    if os.path.isdir(keypoints_dir+slug):
        print("{} is processed".format(keypoints_dir+slug))
        return None
    return 1

    CMD = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 -i \"{}\"".format(path)
    rotate = os.popen(CMD).read().strip()

    if rotate:
        _, file_extension = os.path.splitext(path)
        path_tmp = "/tmp/openpose/tmp{}".format(file_extension)
        os.system("mv \"{}\" {}".format(path, path_tmp))
        
        path = "/tmp/openpose/input.mp4"
        CMD = "rm {path} ; ffmpeg -y -i {path_tmp} {path}".format(path_tmp = path_tmp, path = path)
        
        print(CMD)
        os.system(CMD)
        
    dirpath = os.path.dirname(os.path.abspath(path))
    filepath = os.path.basename(path)

    CMD = "rm {dirpath}/keypoints -r ; mkdir {dirpath}/keypoints ; docker run --gpus=1 -v \"{dirpath}\":/openpose/data stanfordnmbl/openpose-gpu /openpose/build/examples/openpose/openpose.bin\
  --video \"/openpose/data/{filepath}\"\
  --display 0\
  --write_json /openpose/data/keypoints\
  --render_pose 0 ; cp -r {dirpath}/keypoints {slug_dir}".format(dirpath=dirpath, filepath=filepath, slug_dir=keypoints_dir+slug)
    print(CMD)
    os.system(CMD)
    return CMD

def process_raw_video(video_path, processed_npy_path="videos/np/"):
    # Run OpenPose
    # Convert frame jsons to npy
    # Save numpy to "videos/np/" with some subjectid based on the path
    # process_subject(subjectid)
    pass

def process_subject(subjectid, processed_npy_path="videos/np/", framerate = None):
    res = np.load("{}{}.npy".format(processed_npy_path, subjectid))
    
    if subjectid == "pmYdj2Zc":
        res = res[:-10,:]
    
    # make it more intuitive by inverting Y
    res[:,1::3] = 50 + res[:,1::3].max() - res[:,1::3]
    
    md = np.median((res[:,MHIP*3] - (res[:,LKNE*3] + res[:,RKNE*3])/2 ))
    
    orientation = "R" if md < 0 else "L"
    
    print(orientation)
    if orientation == "L":
        res[:,0::3] = 1 + res[:,0::3].max()-res[:,0::3]
        # TODO: swap left and right
        
        for cols in toswap:
            swap_columns(res, cols[0], cols[1])
    
    first = 0
    last = res.shape[0]-0
    magnitude = 1
    
    if subjectid == "k4Zz5q1I":
        first = 75
        last = 240
        magnitude = 0.1
    if subjectid == "hozGKSGr":
        first = 60
        last = 250
        magnitude = 0.1
    if subjectid == "8iHK3CGi":
        first = 550
        last = 1000
        magnitude = 0.1
    if subjectid == "9qluCnOn":
        first = 0
        last = 400
        magnitude = 0.2
    if subjectid == "zyW3PPtt":
        res[res[:,NOSE*3+1] < -1,NOSE*3+1] = np.NaN

    if subjectid in tofix:
        first = tofix[subjectid][0]
        last = tofix[subjectid][1]
        
    if not framerate:
        framerate = videometa[subjectid]["framerate"]

    res = res[first:last,:]

    res[res < 0.5] = np.NaN 
    res = np.apply_along_axis(fill_nan,arr=res,axis=0)
    
    #plt.plot(res[:,RANK])

    res = center_ts(res)
    ups, downs = get_segments(res, magnitude=magnitude, framerate = framerate)
    
    if subjectid in realign:
        print(downs)
        for k,v in realign[subjectid].items():
            downs[k] = v
        print(downs)
    
    # TODO: assert alternating
    allbreaks = sorted(ups.tolist() + downs.tolist())
    if allbreaks[1] == downs[0]:
        allbreaks = allbreaks[1:]
    if allbreaks[0] != downs[0]:
        return None
    if len(allbreaks)%2 == 1:
        allbreaks=allbreaks[:(len(allbreaks)-1)]
    allbreaks = np.array(allbreaks)
    
    results = {
        "subjectid": subjectid,
        "orientation": orientation,
        "framerate": framerate,
    }
    
    # estimate height
    lengths = res[ups[1]:ups[-2],3*NOSE:(3*NOSE+2)] - res[ups[1]:ups[-2],3*RANK:(3*RANK+2)]
    lengths = np.sqrt(np.sum(lengths**2, axis=1))
    height = np.quantile(lengths, 0.95)
    print(height)
    
    for i in range(3*25):
        res[:,i] = smooth_ts(res[:,i], framerate)
        
    # Normalize to r foot
    delta = res[:,3*RANK:(3*RANK+2)].copy()
    for i in range(res.shape[1]//3):
        res[:,3*i:(3*i+2)] = (res[:,3*i:(3*i+2)] - delta)/height
        
    results.update(get_time_results(res, downs, framerate = framerate))
    results.update(get_time_results(res, allbreaks, framerate = framerate, alternate=1))
    results.update(get_time_results(res, allbreaks, framerate = framerate, alternate=-1))

    results.update(get_angles_results(res, downs, framerate = framerate, breaks_alt = ups))
    results.update(get_angles_results(res, allbreaks, framerate = framerate, alternate=1))
    results.update(get_angles_results(res, allbreaks, framerate = framerate, alternate=-1))

    results.update(get_acceleration_results(res, downs, framerate = framerate))
    results.update(get_acceleration_results(res, allbreaks, framerate = framerate, alternate=1))
    results.update(get_acceleration_results(res, allbreaks, framerate = framerate, alternate=-1))

#    kp3d = get_keypoints3d("npz/{}_ts_results.npz".format(videometa[subjectid]["videoid"]), framerate=framerate)
    kp3d = None
    results.update(get_static(res, downs, ups))
    
    if kp3d is not None:
        try:
            results.update(get_angles_results(kp3d, downs, framerate = framerate))
            results.update(get_angles_results(kp3d, allbreaks, framerate = framerate, alternate=1))
            results.update(get_angles_results(kp3d, allbreaks, framerate = framerate, alternate=-1))
        except:
            print("E")

        try:
            results.update(get_acceleration_results(kp3d, downs, framerate = framerate))
            results.update(get_acceleration_results(kp3d, allbreaks, framerate = framerate, alternate=1))
            results.update(get_acceleration_results(kp3d, allbreaks, framerate = framerate, alternate=-1))
        except:
            print("E")

        try:
            results.update(get_static(kp3d, downs, ups))
        except:
            print("E")

    return results


