import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

def mfcc(array, f_ech, M = 10, fmin, fmax):
    fft = np.fft.fft(array)
    energy = np.abs(fft)**2
    list_f = 700*(np.exp(np.arange(M)/2595)-1)
    def H(m,k):
        if k < list_f[m-1] or k > list_f[m+1] : 
            return 0
        elif list_f[m-1] <= k <= list_f[m] : 
            return (2*(k-list_f[m-1])/(list_f[m]-list_f[m-1]))
        elif list_f[m] < k <= list_f[m+1] : 
            return (2*(list_f[m+1]-k)/(list_f[m+1]-list_f[m]))
    fft_filtered = [np.sum([H(m,k) for k in range(len(fft))]*energy**2) for m in range(1, M-1)]
    inv_transfo = np.fft.ifft(np.log(fft_filtered))
    return inv_transfo