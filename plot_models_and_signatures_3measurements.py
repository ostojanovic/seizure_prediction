
import os
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

"""
This script makes two figures:
    * models of time-frequency signatures and signatures of a preictal state
    * models of time-frequency signatures and signatures of an interictal state
"""

ident = '11502'
patient_id = '11502'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'

baseline_train = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/1/train/'
baseline_test = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/1/test/'
baseline_out = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/1/out-of-sample/'

preictal_train = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/1/train/'
preictal_test = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/1/test/'
preictal_out = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/1/out-of-sample/'

files_baseline_train = os.listdir(path_directory+baseline_train)
files_baseline_test = os.listdir(path_directory+baseline_test)
files_baseline_out = os.listdir(path_directory+baseline_out)
files_baseline = np.concatenate((files_baseline_train,files_baseline_test,files_baseline_out))

files_preictal_train = os.listdir(path_directory+preictal_train)
files_preictal_test = os.listdir(path_directory+preictal_test)
files_preictal_out = os.listdir(path_directory+preictal_out)
files_preictal = np.concatenate((files_preictal_train,files_preictal_test,files_preictal_out))

dict_models_baseline = sio.loadmat(path_directory+baseline_train+files_baseline[16])
dict_signatures_baseline1 = sio.loadmat(path_directory+baseline_train+files_baseline[17])

dict_models_preictal = sio.loadmat(path_directory+preictal_train+files_preictal[0])
dict_signatures_preictal1 = sio.loadmat(path_directory+preictal_train+files_preictal[1])

# loading models
H_model_baseline = dict_models_baseline["H_model_baseline"]
W_model_baseline = dict_models_baseline["W_model_baseline"]
H_model_baseline1 = dict_signatures_baseline1["H_model_baseline"]
W_model_baseline1 = dict_signatures_baseline1["W_model_baseline"]

H_model_preictal = dict_models_preictal["H_model_preictal"]
W_model_preictal = dict_models_preictal["W_model_preictal"]
H_model_preictal1 = dict_signatures_preictal1["H_model_preictal"]
W_model_preictal1 = dict_signatures_preictal1["W_model_preictal"]

# loading signatures
W_baseline = dict_models_baseline["W_baseline"]
H_baseline = dict_models_baseline["H_baseline"]
W_baseline1 = dict_signatures_baseline1["W_baseline"]
H_baseline1 = dict_signatures_baseline1["H_baseline"]

W_preictal = dict_models_preictal["W_preictal"]
H_preictal = dict_models_preictal["H_preictal"]
W_preictal1 = dict_signatures_preictal1["W_preictal"]
H_preictal1 = dict_signatures_preictal1["H_preictal"]

idxc = [0,1,3]
electrode_names = ['GB1','GB2','GB3']

# 109602: np.arange(46,55); 'HL8','HL9','HL4','HL2','HL5','HL6','HL3','HL7','HL10'
# 11502: np.arange(0,9); 'HR1','HR2','HR3','HR4','HR5','HR6','HR7','HR8','HR9'
# 25302_2: np.hstack((np.arange(0,8),10)): 'HRA1','HRA2','HRA3','HRA4','HRA5','HRB3','HRB5','HRB4','HRC3'
# 59002_2: np.arange(0,9): 'GB1','GB2','GB3','GB4','GA2','GA3','GA4','GA5','GA6'
# 62002_2: np.arange(0,9): 'TLA4','TLA1','TLA2','TLA3','TLB1','TLB4','TLB2','TLB3','TLC2'
# 97002_2: np.arange(0,9): 'GG5','GG3','GF3','GE7','GF1','GB2','GE4','GE5','GF5'

vmax_pre = 1.3              # maximum value for colormap for preictal
vmax_int = 1.3              # maximum value for colormap for interictal

# plotting the preictal models and signatures
fig,ax = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(19.5,10.2))

img1 = ax[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[0],:],axis=1)*np.expand_dims(H_model_preictal[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,0].set_title('Model (CH: '+electrode_names[0]+', measurement 1)')

img2 = ax[0,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[0],:],axis=1)*np.expand_dims(H_preictal[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,1].set_title('Signature (CH: '+electrode_names[0]+', measurement 1)')

img3 = ax[0,2].imshow(np.rot90(np.expand_dims(W_model_preictal1[idxc[0],:],axis=1)*np.expand_dims(H_model_preictal1[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,2].set_title('Model (CH: '+electrode_names[0]+', measurement 2)')

img4 = ax[0,3].imshow(np.rot90(np.expand_dims(W_preictal1[idxc[0],:],axis=1)*np.expand_dims(H_preictal1[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,3].set_title('Signature (CH: '+electrode_names[0]+', measurement 2)')

img5 = ax[1,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[1],:],axis=1)*np.expand_dims(H_model_preictal[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,0].set_title('Model (CH: '+electrode_names[1]+', measurement 1)')

img6 = ax[1,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[1],:],axis=1)*np.expand_dims(H_preictal[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,1].set_title('Signature (CH: '+electrode_names[1]+', measurement 1)')

img7 = ax[1,2].imshow(np.rot90(np.expand_dims(W_model_preictal1[idxc[1],:],axis=1)*np.expand_dims(H_model_preictal1[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,2].set_title('Model (CH: '+electrode_names[1]+', measurement 2)')

img8 = ax[1,3].imshow(np.rot90(np.expand_dims(W_preictal1[idxc[1],:],axis=1)*np.expand_dims(H_preictal1[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,3].set_title('Signature (CH: '+electrode_names[1]+', measurement 2)')

img9 = ax[2,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[2],:],axis=1)*np.expand_dims(H_model_preictal[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,0].set_title('Model (CH: '+electrode_names[2]+', measurement 1)')

img10 = ax[2,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[2],:],axis=1)*np.expand_dims(H_preictal[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,1].set_title('Signature (CH: '+electrode_names[2]+', measurement 1)')

img11 = ax[2,2].imshow(np.rot90(np.expand_dims(W_model_preictal1[idxc[2],:],axis=1)*np.expand_dims(H_model_preictal1[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,2].set_title('Model (CH: '+electrode_names[2]+', measurement 2)')

img12 = ax[2,3].imshow(np.rot90(np.expand_dims(W_preictal1[idxc[2],:],axis=1)*np.expand_dims(H_preictal1[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,3].set_title('Signature (CH: '+electrode_names[2]+', measurement 2)')

fig.suptitle('Models of time-frequency signatures and signatures of a preictal state (patient 1)', ha='center', fontsize=16)
fig.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=12)
fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=12)
fig.tight_layout()
fig.subplots_adjust(left=0.08, top=0.91, right=0.92, bottom=0.09, wspace=0.13,hspace=0.21)
fig.colorbar(img1, ax=ax.ravel().tolist(),fraction=0.046, pad=0.04)
fig.show()


#plotting the interictal models and signatures
fig2,ax2 = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(19.5,10.2))

img21 = ax2[0,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[0],:],axis=1)*np.expand_dims(H_model_baseline[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,0].set_title('Model (CH: '+electrode_names[0]+', measurement 1)')

img22 = ax2[0,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[0],:],axis=1)*np.expand_dims(H_baseline[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,1].set_title('Signature (CH: '+electrode_names[0]+', measurement 1)')

img23 = ax2[0,2].imshow(np.rot90(np.expand_dims(W_model_baseline1[idxc[0],:],axis=1)*np.expand_dims(H_model_baseline1[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,2].set_title('Model (CH: '+electrode_names[0]+', measurement 2)')

img24 = ax2[0,3].imshow(np.rot90(np.expand_dims(W_baseline1[idxc[0],:],axis=1)*np.expand_dims(H_baseline1[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,3].set_title('Signature (CH: '+electrode_names[0]+', measurement 2)')

img25 = ax2[1,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[1],:],axis=1)*np.expand_dims(H_model_baseline[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,0].set_title('Model (CH: '+electrode_names[1]+', measurement 1)')

img26 = ax2[1,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[1],:],axis=1)*np.expand_dims(H_baseline[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,1].set_title('Signature (CH: '+electrode_names[1]+', measurement 1)')

img27 = ax2[1,2].imshow(np.rot90(np.expand_dims(W_model_baseline1[idxc[1],:],axis=1)*np.expand_dims(H_model_baseline1[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,2].set_title('Model (CH: '+electrode_names[1]+', measurement 2)')

img28 = ax2[1,3].imshow(np.rot90(np.expand_dims(W_baseline1[idxc[1],:],axis=1)*np.expand_dims(H_baseline1[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,3].set_title('Signature (CH: '+electrode_names[1]+', measurement 2)')

img29 = ax2[2,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[2],:],axis=1)*np.expand_dims(H_model_baseline[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,0].set_title('Model (CH: '+electrode_names[2]+', measurement 1)')

img30 = ax2[2,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[2],:],axis=1)*np.expand_dims(H_baseline[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,1].set_title('Signature (CH: '+electrode_names[2]+', measurement 1)')

img31 = ax2[2,2].imshow(np.rot90(np.expand_dims(W_model_baseline1[idxc[2],:],axis=1)*np.expand_dims(H_model_baseline1[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,2].set_title('Model (CH: '+electrode_names[2]+', measurement 2)')

img32 = ax2[2,3].imshow(np.rot90(np.expand_dims(W_baseline1[idxc[2],:],axis=1)*np.expand_dims(H_baseline1[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,3].set_title('Signature (CH: '+electrode_names[2]+', measurement 2)')

fig2.suptitle('Models of time-frequency signatures and signatures of an interictal state (patient 1)', ha='center', fontsize=16)
fig2.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=12)
fig2.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=12)
fig2.tight_layout()
fig2.subplots_adjust(left=0.08, top=0.91, right=0.92, bottom=0.09, wspace=0.13,hspace=0.21)
fig2.colorbar(img21, ax=ax2.ravel().tolist(),fraction=0.046, pad=0.04)
fig2.show()
