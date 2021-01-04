import librosa as lib
import librosa.display
import soundfile as sf
from pathlib import Path
from calls import signal_to_calls
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from preprocessing_janv import preprocessing

import warnings
warnings.filterwarnings('ignore')

SLICE_SIZE = 22000


def slice_signal (signal, fech, species_name, path ) :
    def slicer(signal):
        for i in range(len(signal)//SLICE_SIZE):
            yield signal[SLICE_SIZE*i:SLICE_SIZE*(i+1)]
    for number, sample  in enumerate(slicer(signal)):
        np.save(path + f'pulse_{species_name}_{number}.npy', sample)
        print('SAVED :', path + f'pulse_{species_name}_{number}.npy')

def main():
    ROOT_MP3 = '../mp3/'
    DATAFILE = '../data/'
    files = os.listdir(ROOT_MP3)
    known_species = set(os.listdir(DATAFILE ))
    for FILENAME in files[:1]:
        y, sr = lib.load(Path(ROOT_MP3 + FILENAME))
        parser = FILENAME.split('-')
        species_name = parser[0]+'-'+parser[1]
        # TODO : S'assurer que y'a aps de galère si le nom d'espèce est fait que de une partie
        species_key = parser[0]+'-'+parser[1] + '-' +parser[2][:-4]
        if not species_name in known_species:
            os.mkdir('../data/'+species_name)
            known_species.add(species_name)
        print(species_name)

        TARGETFILE = DATAFILE + species_name+'/'
        plt.plot(y)
        plt.show()
        y, sr = preprocessing(y, sr)
        plt.plot(y)
        plt.show()
        slice_signal(y, sr, species_key, TARGETFILE)
    print('end process')
    return 0



if __name__ == '__main__':
    sys.exit(main())
