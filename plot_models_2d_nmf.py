
from __future__ import unicode_literals
import os
import scipy.io as sio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, gridspec, transforms, rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots models of time-frequency signatures of preictal and interictal states obtained by 1-dimensional and 2-dimensional NMF.
"""

ident = '11502'
patient_id = '11502'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/'


######################################################################## 1-dimensional NMF #########################################################################################
files_baseline1 = os.listdir(path_directory+baseline_directory+'models_baseline/1/train')
files_preictal1 = os.listdir(path_directory+preictal_directory+'models_preictal/1/train')

dict_models_baseline1 = sio.loadmat(path_directory+baseline_directory+'models_baseline/1/train/'+files_baseline1[0])
dict_models_preictal1 = sio.loadmat(path_directory+preictal_directory+'models_preictal/1/train/'+files_preictal1[0])

W_baseline1 = dict_models_baseline1["W_baseline"]
H_baseline1 = dict_models_baseline1["H_baseline"]
H_model_baseline1 = dict_models_baseline1["H_model_baseline"]
W_model_baseline1 = dict_models_baseline1["W_model_baseline"]

W_preictal1 = dict_models_preictal1["W_preictal"]
H_preictal1 = dict_models_preictal1["H_preictal"]
H_model_preictal1 = dict_models_preictal1["H_model_preictal"]
W_model_preictal1 = dict_models_preictal1["W_model_preictal"]

######################################################################## 2-dimensional NMF #########################################################################################
files_baseline = os.listdir(path_directory+baseline_directory+'2dim/')
files_preictal = os.listdir(path_directory+preictal_directory+'2dim/')

dict_models_baseline = sio.loadmat(path_directory+baseline_directory+'2dim/'+files_baseline[62])
dict_models_preictal = sio.loadmat(path_directory+preictal_directory+'2dim/'+files_preictal[11])

W_baseline = dict_models_baseline["W_baseline"]
H_baseline = dict_models_baseline["H_baseline"]
H_model_baseline = dict_models_baseline["H_model_baseline"]
W_model_baseline = dict_models_baseline["W_model_baseline"]

W_preictal = dict_models_preictal["W_preictal"]
H_preictal = dict_models_preictal["H_preictal"]
H_model_preictal = dict_models_preictal["H_model_preictal"]
W_model_preictal = dict_models_preictal["W_model_preictal"]

idxc = 0
vmin_pre = 0.5
vmax_pre = 1.5              # maximum value for colormap for preictal
vmin_int = 0.5
vmax_int = 1.5              # maximum value for colormap for interictal

# plotting the prototypes
fig,ax = plt.subplots(2, 2,sharex=True,sharey=True,figsize=(19.5,10.2))

img1 = ax[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal1[idxc,:],axis=1)*np.expand_dims(H_model_preictal1[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,0].set_ylabel('Frequency (Hz)',fontsize=18)
ax[0,0].set_yticks([0,125,250,375,500])
ax[0,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax[0,0].set_title('Preictal state (1-dimensional NMF)',fontsize=18)
fig.text(0.025,0.55,r"$\textbf{A}$",fontsize=18)

img2 = ax[0,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc,:,0],axis=1)*np.expand_dims(H_model_preictal[idxc,:,0],axis=1).T)+np.rot90(np.expand_dims(W_model_preictal[idxc,:,1],axis=1)*np.expand_dims(H_model_preictal[idxc,:,1],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,1].set_title('Preictal state (2-dimensional NMF)',fontsize=18)
fig.text(0.46,0.55,r"$\textbf{B}$",fontsize=18)

img3 = ax[1,0].imshow(np.rot90(np.expand_dims(W_model_baseline1[idxc,:],axis=1)*np.expand_dims(H_model_baseline1[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax[1,0].set_xlabel('Time (min)',fontsize=18)
ax[1,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax[1,0].set_ylabel('Frequency (Hz)',fontsize=18)
ax[1,0].set_yticks([0,125,250,375,500])
ax[1,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax[1,0].set_title('Interictal state (1-dimensional NMF)',fontsize=18)
fig.text(0.025,0.1,r"$\textbf{C}$",fontsize=18)

img4 = ax[1,1].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc,:,0],axis=1)*np.expand_dims(H_model_baseline[idxc,:,0],axis=1).T)+np.rot90(np.expand_dims(W_model_baseline[idxc,:,1],axis=1)*np.expand_dims(H_model_baseline[idxc,:,1],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax[1,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax[1,1].set_xlabel('Time (min)',fontsize=18)
ax[1,1].set_title('Interictal state (2-dimensional NMF)',fontsize=18)
fig.text(0.46,0.1,r"$\textbf{D}$",fontsize=18)

fig.suptitle('Models of time-frequency signatures (CH: HR1, patient 1)', ha='center', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(left=0.05, bottom=0.10, right=0.92, top=0.90, wspace=0.14, hspace=0.24)
colorbar = fig.colorbar(img1, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=16)
fig.text(0.93, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig.show()
