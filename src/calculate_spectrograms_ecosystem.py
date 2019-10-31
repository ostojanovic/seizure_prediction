
import os
import pandas as pd, numpy as np, spectrum as sp, pickle as pkl
import scipy.signal as signal
from collections import defaultdict

patient_id = "*"    # patient id goes here
set = "Test"    # 'Train' or 'Test'
path = ""   # path goes here

fs = 400
nyq = 0.5*fs

if set == "Test":

    files = os.listdir(path)
    files.sort()
    files = files[:-3]

    for file in files:

        with open(path+file, "rb") as f:
            channels = pkl.load(f)

        N, Wn = signal.buttord(wp=np.array([49, 51])/nyq, ws=np.array([49.5, 50.5])/nyq, gpass=3.0, gstop=40.0)
        b, a = signal.butter(N, Wn, 'bandstop')
        filtered = signal.lfilter(b, a, channels.values, axis=0)
        f, t, Sxx = signal.spectrogram(filtered.T, fs=fs, nfft=2*fs)
        print("calculated psd for ", file)

        dictionary = defaultdict(dict)
        dictionary["f"] = f
        dictionary["t"] = t
        dictionary["Sxx"] = Sxx

        print("saving data")
        with open("*"+file.split(".pkl")[0]+"_spec.pkl", "wb") as f:    # saving directory goes here before '+'
            pkl.dump(dictionary, f)

else:

    labels = ['interictal', 'preictal']

    for label in labels:

        with open(path+patient_id+"_"+label+".pkl", "rb") as file:
            file_dict = pkl.load(file)

            for key, val in file_dict.items():

                dictionary = defaultdict(dict)
                filename = key.split(".")[0]

            with open(path+filename+".pkl", "rb") as file:
                channels = pkl.load(file)

            N, Wn = signal.buttord(wp=np.array([49, 51])/nyq, ws=np.array([49.5, 50.5])/nyq, gpass=3.0, gstop=40.0)
            b, a = signal.butter(N, Wn, 'bandstop')
            filtered = signal.lfilter(b, a, channels.values, axis=0)
            f, t, Sxx = signal.spectrogram(filtered.T, fs=fs, nfft=2*fs)
            print("calculated psd for ", filename)

            dictionary["f"] = f
            dictionary["t"] = t
            dictionary["Sxx"] = Sxx

            print("saving data")
            with open('*'+filename+"_spec.pkl", "wb") as file: # saving directory goes here before '+'
                pkl.dump(dictionary, file)
