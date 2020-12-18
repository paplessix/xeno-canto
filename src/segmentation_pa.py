import librosa as lib
import librosa.display
import soundfile as sf
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

#import warnings
#warnings.filterwarnings('ignore')

ROOT_MP3 = "../mp3/"
FILENAME = "Acrocephalus-arundinaceus-136005.mp3"
FILENAME2 = 'Acrocephalus-schoenobaenus-120666.mp3'

y, sr = lib.load(Path(ROOT_MP3+FILENAME))
print('échantillons', len( y))
print( 'sampling rate', sr)
kernel_size = 1000
h = np.ones(2*kernel_size+1)/(kernel_size*2+1)
mean = np.convolve(y, h, mode='same')
var = np.convolve( np.square(y-mean),h, mode='same')

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
times = librosa.times_like(pulse, sr=sr)
times2 = np.cumsum(1/sr*np.ones_like(y))
print(times2)
print(times)
fig, ax = plt.subplots()
ax.plot(times, librosa.util.normalize(pulse),label='PLP')
ax.plot(times2,var>= 0.001)
ax.vlines(times[beats_plp],0,1, color = 'r')

plt.show()



