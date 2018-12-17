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
This script makes four figures:
    * models of time-frequency signatures of a preictal state
    * models of time-frequency signatures of an interictal state
    * time-frequency signatures of a preictal state
    * time-frequency signatures of an interictal state
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

dict_models_baseline = sio.loadmat(path_directory+baseline_train+files_baseline[0])
dict_models_preictal = sio.loadmat(path_directory+preictal_train+files_preictal[0])

W_baseline = dict_models_baseline["W_baseline"]
H_baseline = dict_models_baseline["H_baseline"]
H_model_baseline = dict_models_baseline["H_model_baseline"]
W_model_baseline = dict_models_baseline["W_model_baseline"]

W_preictal = dict_models_preictal["W_preictal"]
H_preictal = dict_models_preictal["H_preictal"]
H_model_preictal = dict_models_preictal["H_model_preictal"]
W_model_preictal = dict_models_preictal["W_model_preictal"]

idxc = np.arange(0,9)
electrode_names = ['HR1','HR2','HR3','HR4','HR5','HR6','HR7','HR8','HR9']

# 109602: np.arange(46,55); 'HL8','HL9','HL4','HL2','HL5','HL6','HL3','HL7','HL10'
# 11502: np.arange(0,9); 'HR1','HR2','HR3','HR4','HR5','HR6','HR7','HR8','HR9'
# 25302_2: np.hstack((np.arange(0,8),10)): 'HRA1','HRA2','HRA3','HRA4','HRA5','HRB3','HRB5','HRB4','HRC3'
# 59002_2: np.arange(0,9): 'GB1','GB2','GB3','GB4','GA2','GA3','GA4','GA5','GA6'
# 62002_2: np.arange(0,9): 'TLA4','TLA1','TLA2','TLA3','TLB1','TLB4','TLB2','TLB3','TLC2'
# 97002_2: np.arange(0,9): 'GG5','GG3','GF3','GE7','GF1','GB2','GE4','GE5','GF5'

vmin_pre = 0.5
vmax_pre = 1.5              # maximum value for colormap for preictal
vmin_int = 0.5
vmax_int = 1.5              # maximum value for colormap for interictal

# plotting the preictal models
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(19.5,10.2))

img1 = ax[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[0],:],axis=1)*np.expand_dims(H_model_preictal[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,0].set_title('Channel '+electrode_names[0],fontsize=18)
ax[0,0].set_yticks([0,125,250,375,500])
ax[0,0].set_yticklabels([0,32,64,96,128],fontsize=16)
fig.text(0.055,0.66,r"$\textbf{A}$",fontsize=18)

img2 = ax[0,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[1],:],axis=1)*np.expand_dims(H_model_preictal[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,1].set_title('Channel '+electrode_names[1],fontsize=18)
fig.text(0.33,0.66,r"$\textbf{B}$",fontsize=18)

img3 = ax[0,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[2],:],axis=1)*np.expand_dims(H_model_preictal[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[0,2].set_title('Channel '+electrode_names[2],fontsize=18)
fig.text(0.595,0.66,r"$\textbf{C}$",fontsize=18)

img4 = ax[1,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[3],:],axis=1)*np.expand_dims(H_model_preictal[idxc[3],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,0].set_title('Channel '+electrode_names[3],fontsize=18)
ax[1,0].set_yticks([0,125,250,375,500])
ax[1,0].set_yticklabels([0,32,64,96,128],fontsize=16)
fig.text(0.055,0.38,r"$\textbf{D}$",fontsize=18)

img5 = ax[1,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[4],:],axis=1)*np.expand_dims(H_model_preictal[idxc[4],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,1].set_title('Channel '+electrode_names[4],fontsize=18)
fig.text(0.33,0.38,r"$\textbf{E}$",fontsize=18)

img6 = ax[1,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[5],:],axis=1)*np.expand_dims(H_model_preictal[idxc[5],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[1,2].set_title('Channel '+electrode_names[5],fontsize=18)
fig.text(0.595,0.38,r"$\textbf{F}$",fontsize=18)

img7 = ax[2,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[6],:],axis=1)*np.expand_dims(H_model_preictal[idxc[6],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,0].set_title('Channel '+electrode_names[6],fontsize=18)
ax[2,0].set_yticks([0,125,250,375,500])
ax[2,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax[2,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig.text(0.055,0.1,r"$\textbf{G}$",fontsize=18)

img8 = ax[2,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[7],:],axis=1)*np.expand_dims(H_model_preictal[idxc[7],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,1].set_title('Channel '+electrode_names[7],fontsize=18)
ax[2,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig.text(0.33,0.1,r"$\textbf{H}$",fontsize=18)

img9 = ax[2,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[8],:],axis=1)*np.expand_dims(H_model_preictal[idxc[8],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax[2,2].set_title('Channel '+electrode_names[8],fontsize=18)
ax[2,2].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig.text(0.595,0.1,r"$\textbf{I}$",fontsize=18)

fig.suptitle('Models of time-frequency signatures of a preictal state (patient 1)', ha='center', fontsize=20)
fig.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=18)
fig.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=18)
fig.tight_layout()
fig.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
colorbar1 = fig.colorbar(img1, ax=ax.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar1.ax.tick_params(labelsize=16)
fig.text(0.92, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig.show()

# plotting the interictal models
fig2, ax2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(19.5,10.2))

img10 = ax2[0,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[0],:],axis=1)*np.expand_dims(H_model_baseline[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,0].set_title('Channel '+electrode_names[0],fontsize=18)
ax2[0,0].set_yticks([0,125,250,375,500])
ax2[0,0].set_yticklabels([0,32,64,96,128],fontsize=16)
fig2.text(0.055,0.66,r"$\textbf{A}$",fontsize=18)

img11 = ax2[0,1].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[1],:],axis=1)*np.expand_dims(H_model_baseline[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,1].set_title('Channel '+electrode_names[1],fontsize=18)
fig2.text(0.33,0.66,r"$\textbf{B}$",fontsize=18)

img12 = ax2[0,2].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[2],:],axis=1)*np.expand_dims(H_model_baseline[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[0,2].set_title('Channel '+electrode_names[2],fontsize=18)
fig2.text(0.595,0.66,r"$\textbf{C}$",fontsize=18)

img13 = ax2[1,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[3],:],axis=1)*np.expand_dims(H_model_baseline[idxc[3],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,0].set_title('Channel '+electrode_names[3],fontsize=18)
ax2[1,0].set_yticks([0,125,250,375,500])
ax2[1,0].set_yticklabels([0,32,64,96,128],fontsize=16)
fig2.text(0.055,0.38,r"$\textbf{D}$",fontsize=18)

img14 = ax2[1,1].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[4],:],axis=1)*np.expand_dims(H_model_baseline[idxc[4],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,1].set_title('Channel '+electrode_names[4],fontsize=18)
fig2.text(0.33,0.38,r"$\textbf{E}$",fontsize=18)

img15 = ax2[1,2].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[5],:],axis=1)*np.expand_dims(H_model_baseline[idxc[5],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,2].set_title('Channel '+electrode_names[5],fontsize=18)
fig2.text(0.595,0.38,r"$\textbf{F}$",fontsize=18)

img16 = ax2[2,0].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[6],:],axis=1)*np.expand_dims(H_model_baseline[idxc[6],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,0].set_title('Channel '+electrode_names[6],fontsize=18)
ax2[2,0].set_yticks([0,125,250,375,500])
ax2[2,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax2[2,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.055,0.1,r"$\textbf{G}$",fontsize=18)

img17 = ax2[2,1].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[7],:],axis=1)*np.expand_dims(H_model_baseline[idxc[7],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,1].set_title('Channel '+electrode_names[7],fontsize=18)
ax2[2,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.33,0.1,r"$\textbf{H}$",fontsize=18)

img18 = ax2[2,2].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc[8],:],axis=1)*np.expand_dims(H_model_baseline[idxc[8],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[2,2].set_title('Channel '+electrode_names[8],fontsize=18)
ax2[2,2].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.595,0.1,r"$\textbf{I}$",fontsize=18)

fig2.suptitle('Models of time-frequency signatures of an interictal state (patient 1)', ha='center', fontsize=20)
fig2.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=18)
fig2.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=18)
fig2.tight_layout()
fig2.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
colorbar2 = fig2.colorbar(img10, ax=ax2.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar2.ax.tick_params(labelsize=16)
fig2.text(0.92, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig2.show()


# plotting the preictal signatures
fig3, ax3 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(19.5,10.2))

img20 = ax3[0,0].imshow(np.rot90(np.expand_dims(W_preictal[idxc[0],:],axis=1)*np.expand_dims(H_preictal[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[0,0].set_title('Channel: '+electrode_names[0])
ax3[0,0].set_yticks([0,125,250,375,500])
ax3[0,0].set_yticklabels([0,32,64,96,128])

img21 = ax3[0,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[1],:],axis=1)*np.expand_dims(H_preictal[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[0,1].set_title('Channel: '+electrode_names[1])

img22 = ax3[0,2].imshow(np.rot90(np.expand_dims(W_preictal[idxc[2],:],axis=1)*np.expand_dims(H_preictal[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[0,2].set_title('Channel: '+electrode_names[2])

img23 = ax3[1,0].imshow(np.rot90(np.expand_dims(W_preictal[idxc[3],:],axis=1)*np.expand_dims(H_preictal[idxc[3],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[1,0].set_title('Channel: '+electrode_names[3])
ax3[1,0].set_yticks([0,125,250,375,500])
ax3[1,0].set_yticklabels([0,32,64,96,128])

img24 = ax3[1,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[4],:],axis=1)*np.expand_dims(H_preictal[idxc[4],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[1,1].set_title('Channel: '+electrode_names[4])

img25 = ax3[1,2].imshow(np.rot90(np.expand_dims(W_preictal[idxc[5],:],axis=1)*np.expand_dims(H_preictal[idxc[5],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[1,2].set_title('Channel: '+electrode_names[5])

img26 = ax3[2,0].imshow(np.rot90(np.expand_dims(W_preictal[idxc[6],:],axis=1)*np.expand_dims(H_preictal[idxc[6],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[2,0].set_title('Channel: '+electrode_names[6])
ax3[2,0].set_yticks([0,125,250,375,500])
ax3[2,0].set_yticklabels([0,32,64,96,128])

img27 = ax3[2,1].imshow(np.rot90(np.expand_dims(W_preictal[idxc[7],:],axis=1)*np.expand_dims(H_preictal[idxc[7],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[2,1].set_title('Channel: '+electrode_names[7])

img28 = ax3[2,2].imshow(np.rot90(np.expand_dims(W_preictal[idxc[8],:],axis=1)*np.expand_dims(H_preictal[idxc[8],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3[2,2].set_title('Channel: '+electrode_names[8])

fig3.suptitle('Time-frequency signatures of a preictal state (patient 1)', ha='center', fontsize=16)
fig3.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=12)
fig3.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=12)
fig3.tight_layout()
fig3.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
fig3.colorbar(img20, ax=ax3.ravel().tolist(),fraction=0.046, pad=0.04)
fig3.text(0.92, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=12)
fig3.show()


# plotting the interictal signatures
fig4, ax4 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(19.5,10.2))

img30 = ax4[0,0].imshow(np.rot90(np.expand_dims(W_baseline[idxc[0],:],axis=1)*np.expand_dims(H_baseline[idxc[0],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[0,0].set_title('Channel: '+electrode_names[0],fontsize=14)
ax4[0,0].set_yticks([0,125,250,375,500])
ax4[0,0].set_yticklabels([0,32,64,96,128])

img31 = ax4[0,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[1],:],axis=1)*np.expand_dims(H_baseline[idxc[1],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[0,1].set_title('Channel: '+electrode_names[1],fontsize=14)

img32 = ax4[0,2].imshow(np.rot90(np.expand_dims(W_baseline[idxc[2],:],axis=1)*np.expand_dims(H_baseline[idxc[2],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[0,2].set_title('Channel: '+electrode_names[2],fontsize=14)

img33 = ax4[1,0].imshow(np.rot90(np.expand_dims(W_baseline[idxc[3],:],axis=1)*np.expand_dims(H_baseline[idxc[3],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[1,0].set_title('Channel: '+electrode_names[3],fontsize=14)
ax4[1,0].set_yticks([0,125,250,375,500])
ax4[1,0].set_yticklabels([0,32,64,96,128])

img34 = ax4[1,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[4],:],axis=1)*np.expand_dims(H_baseline[idxc[4],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[1,1].set_title('Channel: '+electrode_names[4],fontsize=14)

img35 = ax4[1,2].imshow(np.rot90(np.expand_dims(W_baseline[idxc[5],:],axis=1)*np.expand_dims(H_baseline[idxc[5],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[1,2].set_title('Channel: '+electrode_names[5],fontsize=14)

img36 = ax4[2,0].imshow(np.rot90(np.expand_dims(W_baseline[idxc[6],:],axis=1)*np.expand_dims(H_baseline[idxc[6],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[2,0].set_title('Channel: '+electrode_names[6],fontsize=14)
ax4[2,0].set_yticks([0,125,250,375,500])
ax4[2,0].set_yticklabels([0,32,64,96,128])

img37 = ax4[2,1].imshow(np.rot90(np.expand_dims(W_baseline[idxc[7],:],axis=1)*np.expand_dims(H_baseline[idxc[7],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[2,1].set_title('Channel: '+electrode_names[7],fontsize=14)

img38 = ax4[2,2].imshow(np.rot90(np.expand_dims(W_baseline[idxc[8],:],axis=1)*np.expand_dims(H_baseline[idxc[8],:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=0, vmax=vmax_int, extent=[0,5,0,510])
ax4[2,2].set_title('Channel: '+electrode_names[8],fontsize=14)

fig4.suptitle('Time-frequency signatures of an interictal state (patient 1)', ha='center', fontsize=16)
fig4.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=12)
fig4.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=12)
fig4.tight_layout()
fig4.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
fig4.colorbar(img10, ax=ax4.ravel().tolist(),fraction=0.046, pad=0.04)
fig4.text(0.92, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=12)
fig4.show()
