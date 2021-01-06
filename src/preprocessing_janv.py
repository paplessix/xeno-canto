import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from os import listdir
import librosa
import scipy.signal as sg
import pywt

# Paramètres
mp3_dir = ""
f_sub = 22000

c = 0


def preprocessing(signal, f_ech):
    f_sub = 22000
    if len(signal.shape) == 2:
        signal = signal[:, 0]
        
    N = len(signal)
    delta_ts = 1 / f_ech
    delta_tr = 1 / f_sub
    ts = np.linspace(0, N*delta_ts, N, endpoint=False)
    tr = np.arange(0, max(ts), delta_tr)

    # ### Filtrage passe-bas et sous-échantillonnage
    r = f_ech / f_sub
    n = 50
    t = np.array([k for k in range(-n, n + 1)])
    sinc = np.sinc(t / r)  # sinus cardinal

    h = sinc * np.hanning(2 * n + 1)  # Apodisation
    alpha = 1 / np.sum(h)
    h = alpha * h

    signal_antialiasing = sg.convolve(signal, h, mode="same")  # passe-bas

    # sub-sampling
    f = interp1d(ts, signal_antialiasing, kind="cubic")
    signal_subsampled = f(tr)

    # wavelet denoising
    coeffs = pywt.dwt(signal_subsampled, "dmey")
    thres = np.percentile(coeffs, 95, interpolation="linear") # on garde non nul 5% des coefficients
    new_coeffs = pywt.threshold(coeffs, thres, mode="hard")

    signal_denoised = pywt.idwt(new_coeffs[0], new_coeffs[1], wavelet="dmey")

    # HP and Wiener filter
    filterb = sg.butter(8, 2500, "highpass", fs = f_sub, output='sos')
    signal_filtered = sg.sosfilt(filterb, signal_denoised)
    signal_preprocessed = sg.wiener(signal_filtered)
    
    return signal_preprocessed, f_sub



