
import os
import pandas as pd, numpy as np, pickle as pkl

patient_id = "*"    # patient id goes here
set = "*"           # 'Train' or 'Test'
path = "*"          # path goes here

with open("*.pkl", "rb") as file: # path to the mean interictal spectrogram goes here
    mean_spectra = pkl.load(file)

if set == "Test":

    files = os.listdir(path)
    files.sort()

    for idx, val in enumerate(files):
        with open(path+val, "rb") as file:
            spectra = pkl.load(file)

        normalized_spectra = spectra["Sxx"]/mean_spectra
        spectra["normalized_psd"] = normalized_spectra

        print("saving data: ", val)
        with open(path+val, "wb") as file:
            pkl.dump(spectra, file)

else:

    labels = ['interictal', 'preictal']

    for label in labels:

        files = os.listdir(path+label+"/")
        files.sort()

        if label == "interictal":
            files = files[:-1]

        for idx, val in enumerate(files):

            with open(path+label+"/"+val, "rb") as file:
                spectra = pkl.load(file)

            normalized_spectra = spectra["Sxx"]/mean_spectra
            spectra["normalized_psd"] = normalized_spectra

            print("saving data: ", val)
            with open(path+label+"/"+val, "wb") as file:
                pkl.dump(spectra, file)
