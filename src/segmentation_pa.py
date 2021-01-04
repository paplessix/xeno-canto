import librosa as lib
import librosa.display
import soundfile as sf
from pathlib import Path
from calls import signal_to_calls
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def get_slided_var(y,sr, kernel_size = 1000 ):

    h = np.ones(2*kernel_size+1)/(kernel_size*2+1)
    mean_signal = np.convolve(y, h, mode='same')
    var = np.convolve(np.square(y - mean_signal), h, mode='same')
    return var


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

def save_call(call,sr, species_name,call_number, path ):
    np.savetxt(path + f'call_{sr}_{species_name}_{call_number}.csv', call, delimiter=",")
    print('SAVED : ',path + f'call_{sr}_{species_name}_{call_number}.csv' )


def sample_pulse(y , sr , pulse, species_name , call_number, path  ):
    size = 5000
    for i in range (len(pulse)):
        index = pulse[i]
        if index+size < len(y):
            sampled_pulse = y[index:index+size]
            print(len(sampled_pulse))
            np.savetxt(path+f'pulse_{species_name}_{call_number}_{i}.csv', sampled_pulse, delimiter=",")
            print('SAVED :', path+f'pulse_{species_name}_{call_number}_{i}.csv')
        else :
            pass

def sample_signal(y, sr, species_name, path ):
    call_indexes = signal_to_calls(y, calls = False)
    if call_indexes == None :
        print('Too much call detected, unusable file')
    else :
        for i in range(len(call_indexes)):
            indexes = call_indexes[i]
            call = y[indexes[0]:indexes[1]]
            save_call(call, sr, species_name,i+1, path)
            pulse = get_pulse(call, sr)

            sample_pulse(call, sr, pulse, species_name, i+1,path)
def main():
    ROOT_MP3 = '../mp3/'
    DATAFILE = '../data/'
    files = os.listdir(ROOT_MP3)
    known_species = set(os.listdir(DATAFILE ))
    for FILENAME in files[:100]:
        y, sr = lib.load(Path(ROOT_MP3 + FILENAME))
        parser = FILENAME.split('-')
        species_name = parser[0]+'-'+parser[1]
        # TODO : S'assurer que y'a aps de galère si le nom d'espèce est fait que de une partie
        species_key = parser[0]+'-'+parser[1] + '-' +parser[2][:-4]
        if not species_name in known_species:
            os.mkdir('../data/'+species_name)
            known_species.add(species_name)
        print(species_name)

        TARGETFILE = DATAFILE+ species_name+'/'
        sample_signal(y, sr, species_key, TARGETFILE)




if __name__ == '__main__':
    sys.exit(main())




