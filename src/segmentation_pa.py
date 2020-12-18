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

onset_env = librosa.onset.onset_strength(y=y, sr=sr*2, center = False)
pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
factor = len(y)//len(pulse)
beats_plp = np.flatnonzero(librosa.util.localmax(pulse))*factor
times = np.cumsum(1/sr*np.ones_like(y))

def get_pulse(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr * 2, center=False)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    factor = len(y) // len(pulse)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse)) * factor
    return beats_plp

def get_optimal_sampling_size(pulse, mode ='mean'):
    modes = {"mean": np.mean, "min" : np.min, "max" : np.max}
    function = modes[mode]
    return round(function(pulse[1:]-pulse[:-1]))

def sample_signal(y, sr, pulse,species_name):
    size = get_optimal_sampling_size(pulse)
    for index in pulse:
        sampled_pulse = y[index:index+size]
        sf.write(f'{species_name}', y, sr)
fig, ax = plt.subplots()
ax.plot(times, y* ( var>= 0.001))
final_pulse = []
print(len(beats_plp))
for i in beats_plp:
    if (var>=0.001)[i]:
        final_pulse.append(i)
final_pulse = np.array(final_pulse)
print(len(final_pulse))
ax.vlines(times[final_pulse],0,1)
ax.set(title='librosa.beat.plp', xlim=[10,20])
plt.show()
fig, ax = plt.subplots()
ax.plot(times, y)
ax.vlines(times[beats_plp],0,1)
ax.set(title='librosa.beat.plp2', xlim=[20,30])
plt.show()

pulse_size = final_pulse[1:]-final_pulse[:-1]
print(pulse_size*1/sr)
print(np.mean(pulse_size) * 1/sr)



