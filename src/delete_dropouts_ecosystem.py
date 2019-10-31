
import os
import pandas as pd, numpy as np
import pickle as pkl
from collections import defaultdict

patient_id = "*" # patient id goes here
set = "Test" # 'Train' or 'Test'
path = "*"  # path goes here

files = os.listdir(path+patient_id+set+"/")
files.sort()
files = files[:-1]      # exclude the "deleted_dropouts" folder

preictal = defaultdict(dict)
interictal = defaultdict(dict)

for idx, name in enumerate(files):

    df=pd.read_csv("*"+name)    # path to a specific .csv file goes here before the '+'
    channels = df.loc[:,"ch0":"ch15"]

    num_of_drops = channels[channels.values == 0.034].count()
    mean_num_of_drops = np.mean(num_of_drops)
    percentage_per_channel = np.round(mean_num_of_drops/channels.shape[0]/channels.shape[1]*100, decimals=2)

    if percentage_per_channel <= 50.00:
        bad_indexes = []
        for idx_time in channels.index.values:
            if (channels.values[idx_time,:] == 0.034).any():
                bad_indexes.append(idx_time)

        no_dropouts = np.delete(channels.values, bad_indexes, axis=0)
        mean_recording = np.mean(no_dropouts, axis=0)

        for idx_bad in bad_indexes:
            channels.values[idx_bad,:] = mean_recording

        final = channels
        filename = name.split(".")[0]

        print("saving data: ", name)
        with open("*"+filename+".pkl", "wb") as file: # path to a saving directory goes here instead of '*'
            pkl.dump(final, file)

        if set == "Train":

            label = name.split("_")[-1].split(".")[0]

            if label == str(0):
                interictal[name] = label
                print("added {} to the interictal dictionary".format(name))
            elif label == str(1):
                preictal[name] = label
                print("added {} to the preictal dictionary".format(name))

    else:
        print("Too many dropouts: ", percentage_per_channel, "for measurement: ", name)

if set == "Train":

    print("saving interictal and preictal dictionaries")
    dict_list = ['interictal', 'preictal']

    for (dictionary, label) in zip((interictal, preictal), dict_list):
        with open("*"+"{}_{}.pkl".format(patient_id,label), "wb") as file:    # path to a saving directory goes here before '+'
            pkl.dump(dictionary, file)
