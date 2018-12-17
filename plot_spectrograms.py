from __future__ import unicode_literals
import os
import scipy.io as sio
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots a spectrogram of preictal and a spectrogram of interictal state.
"""

ident = '11502'
patient_id = '11502'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/spectrograms_baseline'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/spectrograms_preictal'

files_baseline = os.listdir(path_directory+baseline_directory)
files_preictal = os.listdir(path_directory+preictal_directory)

dict_spectrogram_baseline = sio.loadmat(path_directory+baseline_directory+'/'+files_baseline[32])
dict_spectrogram_preictal = sio.loadmat(path_directory+preictal_directory+'/'+files_preictal[4])

spectrogram_baseline = dict_spectrogram_baseline["spectrogram_baseline_1"]
spectrogram_preictal = dict_spectrogram_preictal["spectrogram_preictal"]

kill_IDX = list(np.linspace(195,205,11,dtype=int))
spectrogram_baseline = np.delete(spectrogram_baseline,kill_IDX,axis=2)
spectrogram_preictal = np.delete(spectrogram_preictal,kill_IDX,axis=2)

# plotting spectrograms
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(19.5,10.2))

img1 = ax[0].imshow(np.rot90(spectrogram_baseline[0]),cmap='RdBu_r',aspect='auto',vmin=0.5, vmax=1.5, extent=[0,5,0,510])
ax[0].set_title('Spectrogram of an interictal state (CH: HR1, patient 1)',fontsize=18)
ax[0].set_xlabel('Time (min)',fontsize=18)
ax[0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax[0].set_ylabel('Frequency (Hz)',fontsize=18)
ax[0].set_yticks([0,125,250,375,500])
ax[0].set_yticklabels([0,32,64,96,128],fontsize=16)
fig.text(0.055,0.1,r"$\textbf{A}$",fontsize=18)

img2 = ax[1].imshow(np.rot90(spectrogram_preictal[0]),cmap='RdBu_r',aspect='auto',vmin=0.5, vmax=1.5, extent=[0,5,0,510])
ax[1].set_title('Spectrogram of a preictal state (CH: HR1, patient 1)',fontsize=18)
ax[1].set_xlabel('Time (min)',fontsize=18)
ax[1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig.text(0.47,0.1,r"$\textbf{B}$",fontsize=18)

fig.tight_layout()
fig.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
colorbar = fig.colorbar(img2, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=16)
fig.text(0.92, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig.show()
