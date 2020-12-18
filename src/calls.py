import numpy as np
import matplotlib.pyplot as plt
import librosa

def signal_to_calls(signal):
    """Retourne une liste de calls à partir du signal mis en entrée"""

    kernel_size = 1000
    h = np.ones(2*kernel_size+1)/(kernel_size*2+1)
    mean = np.convolve(signal, h, mode='same')
    var = np.convolve( np.square(y-mean),h, mode='same')

    var_seuil = np.array(var>0.0002)

    kernel_size2 = 10000
    h2 = np.ones(2*kernel_size2+1)/(kernel_size2*2+1)
    var_mean = np.convolve(var_seuil, h2, mode ='same')

    sounds, *_ = np.where(var_mean>0.000001)

    n = len(sounds)
    indices_intervalles = []
    a = 0
    b = 1
    while b < n:
        if liste[b] != liste[b-1] + 1:
            l.append((liste[a] + 5000,liste[b-1] - 5000))
            a = b
        b+=1
    l.append((liste[a] + 5000,liste[b-1] - 5000))
    calls = [signal[a:b+1] for (a,b) in indices_intervalles]
    return calls