
import os, pickle, json
import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

patients = ["11502", "25302", "59002", "62002", "97002", "109602"]
num_channels = {"11502":48, "25302": 26, "59002": 94, "62002": 38, "97002": 91, "109602": 68}

for idx, patient_id in enumerate(patients):
    with open('/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_'+patient_id+'_extracted_seizures/prediction_models/smote_'+patient_id+"_0.pickle", "rb") as f:
        file = pickle.load(f)

coefficients = np.asarray(file.coef_[0]).reshape((num_channels[patient_id],12))

plt.imshow(coefficients.T, aspect="auto", cmap='RdBu', vmin=-1, vmax=1)
plt.show()
