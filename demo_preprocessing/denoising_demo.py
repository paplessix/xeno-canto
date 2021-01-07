import numpy as np
from os import listdir
from tqdm import tqdm
import warnings
import librosa
import soundfile as sf
import scipy.signal as sg
from scipy.interpolate import interp1d
import pywt

warnings.filterwarnings("ignore")
     

def preprocessing(filename, f_sub):
    """sub sample and denoise the audio signal

    Args:
        filename (str): the path of the .mp3 file
        f_sub (float): the new sampling rate

    Returns:
        numpy array: the preprocessed signal
    """
    signal, f_s = librosa.load(mp3_dir+filename)
    # si le signal est en stéréo :
    if len(signal.shape) == 2:
        signal = signal[:,0]

    signal = signal.astype("float32")
    N = len(signal)
    delta_ts = 1/f_s 
    delta_tr = 1/f_sub

    ts = np.linspace(0, N*delta_ts, N, endpoint=False)
    tr = np.arange(0, max(ts), delta_tr)

    # ### Filtrage passe-bas puis sous-échantillonnage
    r = f_s/f_sub
    n = 50
    t = np.arange(-n, n+1)
    sinc = np.sinc(t/r) # sinus cardinal

    h = sinc * np.hanning(2*n+1) # Apodisation
    alpha = 1/np.sum(h)
    h = alpha * h

    signal_antialiasing = sg.convolve(signal, h, mode="same") # passe-bas


    # sub-sampling
    f = interp1d(ts, signal_antialiasing, kind = "cubic")
    signal_subsampled = f(tr)

    # wavelet denoising
    coeffs = pywt.dwt(signal_subsampled, "dmey")
    thres = np.percentile(coeffs, 95, interpolation="linear") # on garde non nul 5% des coefficients
    new_coeffs = pywt.threshold(coeffs, thres, mode="hard")

    signal_denoised = pywt.idwt(new_coeffs[0], new_coeffs[1], wavelet="dmey")

    # HP and Wiener filter
    filterb = sg.butter(8, 2500, "highpass", fs = f_sub, output='sos')
    signal_filtered = sg.sosfilt(filterb, signal_denoised)
    signal_filtered2 = sg.wiener(signal_filtered)
    
    # Replace 0. values with the min of non_zero values
    masked_a = np.ma.masked_equal(signal_filtered2, 0.0, copy=False)
    sig_min = np.min(np.abs(masked_a))
    signal_final = signal_filtered2.copy()
    signal_final[signal_final == 0.] = sig_min

    print("signal preprocessed")
    return signal_final


# # Paramètres
# mp3_dir = "data/mp3/"
# f_sub = 22000  

# for filename in tqdm(listdir(mp3_dir)[0:171:10]):
#     signal_processed = preprocessing(filename, f_sub=f_sub)

#     # saving the signal
#     path_new_npy = "data/npy_denoised/" + filename[:-4]
#     raw_wav_path = "data/wav/" + filename[:-4]
#     wav_path = "data/wav_denoised" + filename[:-4]

#     np.save(path_new_npy, signal_processed)
#     # sf.write(path_new + ".wav", signal_final, f_sub)
#     # sf.write(raw_wav_path + ".wav", signal, f_s)
