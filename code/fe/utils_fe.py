import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

def mfcc(file):
    signal, sample_rate = librosa.load(file, sr = None)
    print(signal)
