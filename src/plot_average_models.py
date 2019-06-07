
import os
import scipy.io as sio
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

"""
This script makes a plot of average models for preictal and interictal states.
"""

ident = '11502'
patient_id = '11502'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/original/AV_models/'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/AV_models/'

files_baseline = os.listdir(path_directory+baseline_directory)
files_preictal = os.listdir(path_directory+preictal_directory)

dict_models_baseline = sio.loadmat(path_directory+baseline_directory+'/'+files_baseline[4])
dict_models_preictal = sio.loadmat(path_directory+preictal_directory+'/'+files_preictal[4])

AV_models_baseline = dict_models_baseline["AV_Models_baseline"]
AV_models_preictal = dict_models_preictal["AV_Models_preictal"]

idxc = 0       # channel's index

AV_models_baseline = AV_models_baseline[idxc,:,:]
AV_models_preictal = AV_models_preictal[idxc,:,:]

vmin_pre = 0.5
vmax_pre = 1.5
vmin_int = 0.5
vmax_int = 1.5

# plotting average models
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16,8))
fig.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)

img1 = ax[0].imshow(np.rot90(AV_models_preictal),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0].set_title('Preictal state (CH: HR1, patient 1)',fontsize=28)
ax[0].set_xlabel('Time (min)',fontsize=26)
ax[0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax[0].set_ylabel('Frequency (Hz)',fontsize=26)
ax[0].set_yticks([0,125,250,375,500])
ax[0].set_yticklabels([0,32,64,96,128],fontsize=22)

img2 = ax[1].imshow(np.rot90(AV_models_baseline),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax[1].set_title('Interictal state (CH: HR1, patient 1)',fontsize=28)
ax[1].set_xlabel('Time (min)',fontsize=26)
ax[1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)

colorbar = fig.colorbar(img2, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=22)

fig.text(0.045,0.1,r"$\textbf{A}$",fontsize=26)
fig.text(0.46,0.1,r"$\textbf{B}$",fontsize=26)
fig.text(0.94, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=26)

# fig.savefig("figures/average_models.pdf", pad_inches=0.4)
plt.show()
