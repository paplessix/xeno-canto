import numpy as np


def signal_to_calls(signal, calls =True):
    """extrait les calls à partir d'un signal

    Args:
        signal (numpy array): le signal dont on veut extraire les calls
        calls (bool, optional): si True, renvoie les calls sous forme de numpy array. 
                                Defaults to True. 
    """

    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:] / n

    intensity = signal*signal
    puissance_moy = np.sum(intensity)*10000/len(intensity)



    kernel_size = 10000
    mean = moving_average(signal, kernel_size)
    var = moving_average(np.square(signal[kernel_size - 1:] - mean), kernel_size)
    var = var/puissance_moy


    sounds, *_ = np.where(var>0.00001)

    n = len(sounds)
    indices_intervalles = []
    a = 0
    b = 1
    while b < n:
        if sounds[b] != sounds[b-1] + 1 and b-a > 2*kernel_size :
            indices_intervalles.append((sounds[a] + int(kernel_size/2),sounds[b-1] - int(kernel_size/2)))
            a = b
        b+=1
    if b-a > 2*kernel_size :
        indices_intervalles.append((sounds[a] + int(kernel_size/2),sounds[b-1] - int(kernel_size/2)))


    calls = np.array([signal[a:b+1] for (a,b) in indices_intervalles])
    
    if len(indices_intervalles) < 1000:
        return indices_intervalles, calls
    else :
        return "Too many intervalles", calls



