import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from os import listdir
import librosa
import scipy.signal as sg
from scipy.signal import wiener, butter, sosfilt

# Paramètres
mp3_dir = ""
f_sub = 22000       

c = 0
for filename in listdir(mp3_dir):
    c+=1
    print("Etape : ", c, " / ", len(listdir(mp3_dir)))

    signal, f_s = librosa.load(filename)
    if len(signal.shape) == 2:
        signal = signal[:,0]


    delta_ts = 1/f_s 
    delta_tr = 1/f_sub

    ts = np.array([k *delta_ts for k in range(len(signal))])
    tr = np.arange(0, max(ts), delta_tr)

    # ### Filtrage passe-bas et sous-échantillonnage
    r = f_s/f_sub
    n = 50
    t = np.array([k for k in range(-n, n+1)])
    sinc = np.sinc(t/r) # sinus cardinal

    h = sinc * np.hanning(2*n+1) # Apodisation
    alpha = 1/np.sum(h)
    h = alpha * h

    signal_antialiasing = sg.convolve(signal, h, mode="same") # passe-bas


    # sub-sampling
    f = interp1d(ts, signal_antialiasing, kind = "cubic")
    signal_subsampled = f(tr)


    plt.plot(ts, signal)
    plt.xlim(4,5)


    plt.plot(tr, signal_subsampled)
    plt.xlim(4, 5)


    filterb = butter(8, 4000, "highpass", fs = f_sub, output='sos')
    signal_filtered = sosfilt(filterb, signal_subsampled)
    signal_filtered = wiener(signal_filtered, noise =250)


    plt.plot(tr, signal_filtered)
    plt.xlim(4, 5)


