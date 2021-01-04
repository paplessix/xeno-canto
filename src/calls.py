import numpy as np
import matplotlib.pyplot as plt
import librosa

def signal_to_calls(signal, calls =True):
    """Retourne une liste de calls à partir du signal mis en entrée"""

    intensity = signal*signal
    puissance_moy = np.sum(intensity)*10000/len(intensity)

    kernel_size = 1000
    h = np.ones(2*kernel_size+1)/(kernel_size*2+1)
    mean = np.convolve(signal, h, mode='same')
    var = np.convolve(np.square(signal - mean),h, mode='same')
    var = var/puissance_moy

    var_seuil = np.array(var> np.mean(var))

    kernel_size2 = 10000
    h2 = np.ones(2*kernel_size2+1)/(kernel_size2*2+1)
    var_mean = np.convolve(var_seuil, h2, mode ='same')

    sounds, *_ = np.where(var_mean>0.000001)

    n = len(sounds)
    indices_intervalles = []
    a = 0
    b = 1
    while b < n:
        if sounds[b] != sounds[b-1] + 1:
            indices_intervalles.append((sounds[a] + 5000,sounds[b-1] - 5000))
            a = b
        b+=1
    indices_intervalles.append((sounds[a] + 5000,sounds[b-1] - 5000))

    if len(indices_intervalles) > 10:
        return None

    if calls :
        calls = [signal[a:b+1] for (a,b) in indices_intervalles]
        return calls

    return indices_intervalles
