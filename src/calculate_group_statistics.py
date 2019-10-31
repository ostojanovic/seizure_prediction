
import os
import scipy.io as sio
import numpy as np
import pickle as pkl
from collections import OrderedDict

path = "*"         # path goes here
dataset = "Epilepsyecosystem"   # EPILEPSIAE or Epilepsyecosystem

patients = OrderedDict() # patient ids go here in an ordered dictionary where keys are patient id's and values are the number of channels
frequencies = OrderedDict() # frequencies for each patient go here in an ordered dictionary where keys are patient id's and values are sampling frequencies

W_preictal_channels = []
H_preictal_channels = []
W_interictal_channels = []
H_interictal_channels = []

W_preictal_measurements = []
H_preictal_measurements = []
W_interictal_measurements = []
H_interictal_measurements = []

for idx, (patient_id, num_ch) in enumerate(patients.items()):

    files_preictal = os.listdir("*")   # path to a folder with preictal models goes here
    files_interictal = os.listdir("*")   # path to a folder with interictal models goes here

    if dataset == "EPILEPSIAE":
        W_preictal = np.zeros((len(files_preictal),num_ch,29))
        H_preictal = np.zeros((len(files_preictal),num_ch,501))
        W_interictal = np.zeros((len(files_interictal),num_ch,29))
        H_interictal = np.zeros((len(files_interictal),num_ch,501))

    elif dataset == "Epilepsyecosystem":
        W_preictal = np.zeros((len(files_preictal),num_ch,401))
        H_preictal = np.zeros((len(files_preictal),num_ch,1071))
        W_interictal = np.zeros((len(files_interictal),num_ch,401))
        H_interictal = np.zeros((len(files_interictal),num_ch,1071))

    for n in range(len(files_preictal)):

        if dataset == "EPILEPSIAE":
            dict_models_preictal = sio.loadmat("*"+files_preictal[n]) # path to preictal models goes here before '+'

            W_preictal[n,:,:] = dict_models_preictal["W_model_preictal"]
            H_preictal[n,:,:] = dict_models_preictal["H_model_preictal"]

        elif dataset == "Epilepsyecosystem":
            with open("*"+files_preictal[n], "rb") as f:   # path to preictal models goes here before '+'
                dict_models_preictal = pkl.load(f)

            W_preictal[n,:,:] = np.squeeze(dict_models_preictal["W_predict"])
            H_preictal[n,:,:] = np.squeeze(dict_models_preictal["H_predict"])

    for n in range(len(files_interictal)):

        if dataset == "EPILEPSIAE":
            dict_models_interictal = sio.loadmat("*"+files_interictal[n])    # path to interictal models goes here before '+'

            H_interictal[n,:,:] = dict_models_interictal["H_model_interictal"]
            W_interictal[n,:,:] = dict_models_interictal["W_model_interictal"]

        elif dataset == "Epilepsyecosystem":

            with open("*"+files_interictal[n], "rb") as f:   # path to interictal models goes here before '+'
                dict_models_interictal = pkl.load(f)

            H_interictal[n,:,:] = np.squeeze(dict_models_interictal["H_predict"])
            W_interictal[n,:,:] = np.squeeze(dict_models_interictal["W_predict"])

    W_preictal_channels.append(np.mean(W_preictal, axis=0))
    H_preictal_channels.append(np.mean(H_preictal, axis=0))
    W_interictal_channels.append(np.mean(W_interictal, axis=0))
    H_interictal_channels.append(np.mean(H_interictal, axis=0))

    W_preictal_measurements.append(np.mean(W_preictal, axis=1))
    H_preictal_measurements.append(np.mean(H_preictal, axis=1))
    W_interictal_measurements.append(np.mean(W_interictal, axis=1))
    H_interictal_measurements.append(np.mean(H_interictal, axis=1))

### measurements ####
W_preictal_max_measurements = [np.max(W_preictal_measurements[x], axis=1) for x in range(len(patients))]
W_preictal_argmax_measurements = [np.argmax(W_preictal_measurements[x], axis=1) for x in range(len(patients))]
W_preictal_min_measurements = [np.min(W_preictal_measurements[x], axis=1) for x in range(len(patients))]
W_preictal_argmin_measurements = [np.argmin(W_preictal_measurements[x], axis=1) for x in range(len(patients))]
W_preictal_mean_measurements = [np.mean(W_preictal_measurements[x], axis=1) for x in range(len(patients))]

H_preictal_max_measurements = [np.max(H_preictal_measurements[x], axis=1) for x in range(len(patients))]
H_preictal_argmax_measurements = [np.argmax(H_preictal_measurements[x], axis=1) for x in range(len(patients))]
H_preictal_min_measurements = [np.min(H_preictal_measurements[x], axis=1) for x in range(len(patients))]
H_preictal_argmin_measurements = [np.argmin(H_preictal_measurements[x], axis=1) for x in range(len(patients))]
H_preictal_mean_measurements = [np.mean(H_preictal_measurements[x], axis=1) for x in range(len(patients))]

W_interictal_max_measurements = [np.max(W_interictal_measurements[x], axis=1) for x in range(len(patients))]
W_interictal_argmax_measurements = [np.argmax(W_interictal_measurements[x], axis=1) for x in range(len(patients))]
W_interictal_min_measurements = [np.min(W_interictal_measurements[x], axis=1) for x in range(len(patients))]
W_interictal_argmin_measurements = [np.argmin(W_interictal_measurements[x], axis=1) for x in range(len(patients))]
W_interictal_mean_measurements = [np.mean(W_interictal_measurements[x], axis=1) for x in range(len(patients))]

H_interictal_max_measurements = [np.max(H_interictal_measurements[x], axis=1) for x in range(len(patients))]
H_interictal_argmax_measurements = [np.argmax(H_interictal_measurements[x], axis=1) for x in range(len(patients))]
H_interictal_min_measurements = [np.min(H_interictal_measurements[x], axis=1) for x in range(len(patients))]
H_interictal_argmin_measurements = [np.argmin(H_interictal_measurements[x], axis=1) for x in range(len(patients))]
H_interictal_mean_measurements = [np.mean(H_interictal_measurements[x], axis=1) for x in range(len(patients))]

##### calculating frequencies ######

H_preictal_argmax_frequencies = []
H_preictal_argmin_frequencies = []
H_interictal_argmax_frequencies = []
H_interictal_argmin_frequencies = []
for idx, (patient, frequency) in enumerate(frequencies.items()):


    if dataset == "EPILEPSIAE":

        conversion_factor = np.round(frequencies[patient]/2/501, decimals=2)

    elif dataset == "Epilepsyecosystem":

        conversion_factor = np.round(frequencies[patient]/2/1071, decimals=2)

    H_preictal_argmax_frequencies.append(H_preictal_argmax_measurements[idx]*conversion_factor)
    H_preictal_argmin_frequencies.append(H_preictal_argmin_measurements[idx]*conversion_factor)
    H_interictal_argmax_frequencies.append(H_interictal_argmax_measurements[idx]*conversion_factor)
    H_interictal_argmin_frequencies.append(H_interictal_argmin_measurements[idx]*conversion_factor)

##### channels ####
W_preictal_max_channels = [np.max(W_preictal_channels[x], axis=1) for x in range(len(patients))]
W_preictal_argmax_channels = [np.argmax(W_preictal_channels[x], axis=1) for x in range(len(patients))]
W_preictal_min_channels = [np.min(W_preictal_channels[x], axis=1) for x in range(len(patients))]
W_preictal_argmin_channels = [np.argmin(W_preictal_channels[x], axis=1) for x in range(len(patients))]

H_preictal_max_channels = [np.max(H_preictal_channels[x], axis=1) for x in range(len(patients))]
H_preictal_argmax_channels = [np.argmax(H_preictal_channels[x], axis=1) for x in range(len(patients))]
H_preictal_min_channels = [np.min(H_preictal_channels[x], axis=1) for x in range(len(patients))]
H_preictal_argmin_channels = [np.argmin(H_preictal_channels[x], axis=1) for x in range(len(patients))]

W_interictal_max_channels = [np.max(W_interictal_channels[x], axis=1) for x in range(len(patients))]
W_interictal_argmax_channels = [np.argmax(W_interictal_channels[x], axis=1) for x in range(len(patients))]
W_interictal_min_channels = [np.min(W_interictal_channels[x], axis=1) for x in range(len(patients))]
W_interictal_argmin_channels = [np.argmin(W_interictal_channels[x], axis=1) for x in range(len(patients))]

H_interictal_max_channels = [np.max(H_interictal_channels[x], axis=1) for x in range(len(patients))]
H_interictal_argmax_channels = [np.argmax(H_interictal_channels[x], axis=1) for x in range(len(patients))]
H_interictal_min_channels = [np.min(H_interictal_channels[x], axis=1) for x in range(len(patients))]
H_interictal_argmin_channels = [np.argmin(H_interictal_channels[x], axis=1) for x in range(len(patients))]

group_measures = {"W_preictal_max_measurements": W_preictal_max_measurements,
"W_preictal_argmax_measurements": W_preictal_argmax_measurements,
"W_preictal_min_measurements": W_preictal_min_measurements,
"W_preictal_argmin_measurements": W_preictal_argmin_measurements,
"W_preictal_mean_measurements": W_preictal_mean_measurements,
"H_preictal_max_measurements": H_preictal_max_measurements,
"H_preictal_argmax_measurements": H_preictal_argmax_measurements,
"H_preictal_min_measurements": H_preictal_min_measurements,
"H_preictal_argmin_measurements": H_preictal_argmin_measurements,
"H_preictal_mean_measurements": H_preictal_mean_measurements,
"W_interictal_max_measurements": W_interictal_max_measurements,
"W_interictal_argmax_measurements": W_interictal_argmax_measurements,
"W_interictal_min_measurements": W_interictal_min_measurements,
"W_interictal_argmin_measurements": W_interictal_argmin_measurements,
"W_interictal_mean_measurements": W_interictal_mean_measurements,
"H_interictal_max_measurements": H_interictal_max_measurements,
"H_interictal_argmax_measurements": H_interictal_argmax_measurements,
"H_interictal_min_measurements": H_interictal_min_measurements,
"H_interictal_argmin_measurements": H_interictal_argmin_measurements,
"H_interictal_mean_measurements": H_interictal_mean_measurements,
"W_preictal_max_channels": W_preictal_max_channels,
"W_preictal_argmax_channels": W_preictal_argmax_channels,
"W_preictal_min_channels": W_preictal_min_channels,
"W_preictal_argmin_channels": W_preictal_argmin_channels,
"H_preictal_max_channels": H_preictal_max_channels,
"H_preictal_argmax_channels": H_preictal_argmax_channels,
"H_preictal_min_channels": H_preictal_min_channels,
"H_preictal_argmin_channels": H_preictal_argmin_channels,
"W_interictal_max_channels": W_interictal_max_channels,
"W_interictal_argmax_channels": W_interictal_argmax_channels,
"W_interictal_min_channels": W_interictal_min_channels,
"W_interictal_argmin_channels": W_interictal_argmin_channels,
"H_interictal_max_channels": H_interictal_max_channels,
"H_interictal_argmax_channels": H_interictal_argmax_channels,
"H_interictal_min_channels": H_interictal_min_channels,
"H_interictal_argmin_channels": H_interictal_argmin_channels,
"H_preictal_argmax_frequencies": H_preictal_argmax_frequencies,
"H_preictal_argmin_frequencies": H_preictal_argmin_frequencies,
"H_interictal_argmax_frequencies": H_interictal_argmax_frequencies,
"H_interictal_argmin_frequencies": H_interictal_argmin_frequencies}

with open('../data/group_measures.pkl', 'wb') as f:  # path for saving goes here 
    pkl.dump(group_measures, f)
