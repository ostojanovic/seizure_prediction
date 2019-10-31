
import os
import pandas as pd, numpy as np, pickle as pkl
from collections import defaultdict

patient_id = "*"    # patient id goes here
path = "*"  # path goes here

files = os.listdir(path)
files.sort()

interictal_batch_one = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[:int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_one[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_one = np.mean(interictal_batch_one, axis=0)
del interictal_batch_one

interictal_batch_two = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[int(len(files)/18):2*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_two[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_two = np.mean(interictal_batch_two, axis=0)
del interictal_batch_two

interictal_batch_three = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[2*int(len(files)/18):3*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_three[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_three = np.mean(interictal_batch_three, axis=0)
del interictal_batch_three

interictal_batch_four = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[3*int(len(files)/18):4*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_four[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_four = np.mean(interictal_batch_four, axis=0)
del interictal_batch_four

interictal_batch_one = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[:int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_one[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_one = np.mean(interictal_batch_one, axis=0)
del interictal_batch_one

interictal_batch_two = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[int(len(files)/18):2*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_two[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_two = np.mean(interictal_batch_two, axis=0)
del interictal_batch_two

interictal_batch_three = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[2*int(len(files)/18):3*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_three[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_three = np.mean(interictal_batch_three, axis=0)
del interictal_batch_three

interictal_batch_four = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[3*int(len(files)/18):4*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_four[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_four = np.mean(interictal_batch_four, axis=0)
del interictal_batch_four

interictal_batch_one = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[:int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_one[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_one = np.mean(interictal_batch_one, axis=0)
del interictal_batch_one

interictal_batch_two = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[int(len(files)/18):2*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_two[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_two = np.mean(interictal_batch_two, axis=0)
del interictal_batch_two

interictal_batch_three = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[2*int(len(files)/18):3*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_three[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_three = np.mean(interictal_batch_three, axis=0)
del interictal_batch_three

interictal_batch_four = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[3*int(len(files)/18):4*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_four[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_four = np.mean(interictal_batch_four, axis=0)
del interictal_batch_four

interictal_batch_five = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[4*int(len(files)/18):5*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_five[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_five = np.mean(interictal_batch_five, axis=0)
del interictal_batch_five

interictal_batch_six = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[5*int(len(files)/18):6*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_six[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_six = np.mean(interictal_batch_six, axis=0)
del interictal_batch_six

interictal_batch_seven = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[6*int(len(files)/18):7*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_seven[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_seven = np.mean(interictal_batch_seven, axis=0)
del interictal_batch_seven

interictal_batch_eight = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[8*int(len(files)/18):9*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_eight[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_eight = np.mean(interictal_batch_eight, axis=0)
del interictal_batch_eight

interictal_batch_nine = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[9*int(len(files)/18):10*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_nine[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_nine = np.mean(interictal_batch_nine, axis=0)
del interictal_batch_nine

interictal_batch_ten = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[10*int(len(files)/18):11*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_ten[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_ten = np.mean(interictal_batch_ten, axis=0)
del interictal_batch_ten

interictal_batch_eleven = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[11*int(len(files)/18):12*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_eleven[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_eleven = np.mean(interictal_batch_eleven, axis=0)
del interictal_batch_eleven

interictal_batch_twelve = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[12*int(len(files)/18):13*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_twelve[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_twelve = np.mean(interictal_batch_twelve, axis=0)
del interictal_batch_twelve

interictal_batch_thirteen = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[13*int(len(files)/18):14*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_thirteen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_thirteen = np.mean(interictal_batch_thirteen, axis=0)
del interictal_batch_thirteen

interictal_batch_fourteen = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[14*int(len(files)/18):15*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_fourteen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_fourteen = np.mean(interictal_batch_fourteen, axis=0)
del interictal_batch_fourteen

interictal_batch_fifteen = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[15*int(len(files)/18):16*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_fifteen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_fifteen = np.mean(interictal_batch_fifteen, axis=0)
del interictal_batch_fifteen

interictal_batch_sixteen = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[16*int(len(files)/18):17*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_sixteen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_sixteen = np.mean(interictal_batch_sixteen, axis=0)
del interictal_batch_sixteen

interictal_batch_seventeen = np.zeros((int(len(files)/18),16,401,1071))

for idx, name in enumerate(files[17*int(len(files)/18):18*int(len(files)/18)]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_seventeen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_seventeen = np.mean(interictal_batch_seventeen, axis=0)
del interictal_batch_seventeen

interictal_batch_eighteen = np.zeros((int(len(files[18*int(len(files)/18):])),16,401,1071))

for idx, name in enumerate(files[18*int(len(files)/18):]):
    with open(path+name, "rb") as f:
        spectrum = pkl.load(f)

    if np.isnan(spectrum["Sxx"]).any():
        print("nans found")
    else:
        interictal_batch_eighteen[idx,:,:,:] = spectrum["Sxx"]
        print("spectrum added: ", idx, name)

mean_psd_eighteen = np.mean(interictal_batch_eighteen, axis=0)
del interictal_batch_eighteen

psd_mean = np.mean((mean_psd_one,
mean_psd_two,
mean_psd_three,
mean_psd_four,
mean_psd_five,
mean_psd_six,
mean_psd_seven,
mean_psd_eight,
mean_psd_nine,
mean_psd_ten,
mean_psd_eleven,
mean_psd_twelve,
mean_psd_thirteen,
mean_psd_fourteen,
mean_psd_fifteen,
mean_psd_sixteen,
mean_psd_seventeen,
mean_psd_eighteen), axis=0)

with open('*'+"Train_mean_spec.pkl", "wb") as file: # saving path goes here before the '+'
    pkl.dump(psd_mean, file)
