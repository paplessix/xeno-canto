import numpy as np
import matplotlib.pyplot as plt
import librosa



path = "C:\\Users\\Timothée Blondel\\Documents\\challenge_birds\\Acrocephalus-arundinaceus-131536.mp3"


signal, sr = librosa.load(path)

def signal_to_calls(signal, calls =True):
    """Retourne une liste de calls à partir du signal mis en entrée"""

    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:] / n

    intensity = signal*signal
    puissance_moy = np.sum(intensity)*10000/len(intensity)


    kernel_size = 20000
    mean = moving_average(signal, kernel_size)
    var = moving_average(np.square(signal[kernel_size - 1:] - mean), kernel_size)
    var = var/puissance_moy


    sounds, *_ = np.where(var>0.00001)

    n = len(sounds)
    indices_intervalles = []
    a = 0
    b = 1
    while b < n:
        if sounds[b] != sounds[b-1] + 1:
            indices_intervalles.append((sounds[a] + kernel_size,sounds[b-1] + kernel_size))
            a = b
        b+=1
    indices_intervalles.append((sounds[a] + kernel_size,sounds[b-1] + kernel_size))

    if len(indices_intervalles) > 1000:
        return None

    if calls :
        calls = [signal[a:b+1] for (a,b) in indices_intervalles]
        return calls

    return indices_intervalles

a = signal_to_calls(signal)

plt.plot(a[3])
plt.show()



