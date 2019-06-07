
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
This script makes two figures:
    * models of time-frequency signatures of a preictal state
    * models of time-frequency signatures of an interictal state
"""

ident = '11502'
patient_id = '11502'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'

interictal = 'patient_'+ident+'_extracted_seizures/data/models_interictal/'
preictal = 'patient_'+ident+'_extracted_seizures/data/models_preictal/'

files_interictal = os.listdir(path_directory+interictal)
files_preictal = os.listdir(path_directory+preictal)

dict_models_interictal = sio.loadmat(path_directory+interictal+files_interictal[0])
dict_models_preictal = sio.loadmat(path_directory+preictal+files_preictal[0])

W_interictal = dict_models_interictal["W_baseline"]
H_interictal = dict_models_interictal["H_baseline"]
H_model_interictal = dict_models_interictal["H_model_baseline"]
W_model_interictal = dict_models_interictal["W_model_baseline"]

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

############################################################### plotting preictal models ###########################################################################

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(16,8))
fig.subplots_adjust(left=0.09, bottom=0.1, right=0.93, top=0.92, wspace=0.12, hspace=0.3)

img1 = ax[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[0],:],axis=1)*np.expand_dims(H_model_preictal[idxc[0],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[0,0].set_title('Channel '+electrode_names[0],fontsize=22)
ax[0,0].set_yticks([0,125,250,375,500])
ax[0,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax[0,0].tick_params(length=8)

img2 = ax[0,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[1],:],axis=1)*np.expand_dims(H_model_preictal[idxc[1],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[0,1].set_title('Channel '+electrode_names[1],fontsize=22)
ax[0,1].tick_params(length=8)

img3 = ax[0,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[2],:],axis=1)*np.expand_dims(H_model_preictal[idxc[2],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[0,2].set_title('Channel '+electrode_names[2],fontsize=22)
ax[0,2].tick_params(length=8)

img4 = ax[1,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[3],:],axis=1)*np.expand_dims(H_model_preictal[idxc[3],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[1,0].set_title('Channel '+electrode_names[3],fontsize=22)
ax[1,0].set_yticks([0,125,250,375,500])
ax[1,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax[1,0].tick_params(length=8)

img5 = ax[1,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[4],:],axis=1)*np.expand_dims(H_model_preictal[idxc[4],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[1,1].set_title('Channel '+electrode_names[4],fontsize=22)
ax[1,1].tick_params(length=8)

img6 = ax[1,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[5],:],axis=1)*np.expand_dims(H_model_preictal[idxc[5],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[1,2].set_title('Channel '+electrode_names[5],fontsize=22)
ax[1,2].tick_params(length=8)

img7 = ax[2,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[6],:],axis=1)*np.expand_dims(H_model_preictal[idxc[6],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[2,0].set_title('Channel '+electrode_names[6],fontsize=22)
ax[2,0].set_yticks([0,125,250,375,500])
ax[2,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax[2,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax[2,0].tick_params(length=8)

img8 = ax[2,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[7],:],axis=1)*np.expand_dims(H_model_preictal[idxc[7],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[2,1].set_title('Channel '+electrode_names[7],fontsize=22)
ax[2,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax[2,1].tick_params(length=8)

img9 = ax[2,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[8],:],axis=1)*np.expand_dims(H_model_preictal[idxc[8],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax[2,2].set_title('Channel '+electrode_names[8],fontsize=22)
ax[2,2].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax[2,2].tick_params(length=8)

colorbar1 = fig.colorbar(img1, ax=ax.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar1.ax.tick_params(labelsize=22)
colorbar1.ax.tick_params(length=8, labelsize=22)

fig.text(0.5, 0.01, 'Time (min)', ha='center',fontsize=22)
fig.text(0.02, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=22)
fig.text(0.95, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=22)

fig.text(0.055,0.68,r"$\textbf{A}$",fontsize=22)
fig.text(0.335,0.68,r"$\textbf{B}$",fontsize=22)
fig.text(0.6,0.68,r"$\textbf{C}$",fontsize=22)
fig.text(0.055,0.39,r"$\textbf{D}$",fontsize=22)
fig.text(0.335,0.39,r"$\textbf{E}$",fontsize=22)
fig.text(0.6,0.39,r"$\textbf{F}$",fontsize=22)
fig.text(0.055,0.089,r"$\textbf{G}$",fontsize=22)
fig.text(0.335,0.089,r"$\textbf{H}$",fontsize=22)
fig.text(0.6,0.089,r"$\textbf{I}$",fontsize=22)

################################################################## plotting the interictal models ###################################################################################

fig2, ax2 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(16,8))
fig2.subplots_adjust(left=0.09, bottom=0.1, right=0.93, top=0.92, wspace=0.12, hspace=0.3)

img10 = ax2[0,0].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[0],:],axis=1)*np.expand_dims(H_model_interictal[idxc[0],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[0,0].set_title('Channel '+electrode_names[0],fontsize=22)
ax2[0,0].set_yticks([0,125,250,375,500])
ax2[0,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax2[0,0].tick_params(length=8)

img11 = ax2[0,1].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[1],:],axis=1)*np.expand_dims(H_model_interictal[idxc[1],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[0,1].set_title('Channel '+electrode_names[1],fontsize=22)
ax2[0,1].tick_params(length=8)

img12 = ax2[0,2].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[2],:],axis=1)*np.expand_dims(H_model_interictal[idxc[2],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[0,2].set_title('Channel '+electrode_names[2],fontsize=22)
ax2[0,2].tick_params(length=8)

img13 = ax2[1,0].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[3],:],axis=1)*np.expand_dims(H_model_interictal[idxc[3],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[1,0].set_title('Channel '+electrode_names[3],fontsize=22)
ax2[1,0].set_yticks([0,125,250,375,500])
ax2[1,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax2[1,0].tick_params(length=8)

img14 = ax2[1,1].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[4],:],axis=1)*np.expand_dims(H_model_interictal[idxc[4],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[1,1].set_title('Channel '+electrode_names[4],fontsize=22)
ax2[1,1].tick_params(length=8)

img15 = ax2[1,2].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[5],:],axis=1)*np.expand_dims(H_model_interictal[idxc[5],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[1,2].set_title('Channel '+electrode_names[5],fontsize=22)
ax2[1,2].tick_params(length=8)

img16 = ax2[2,0].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[6],:],axis=1)*np.expand_dims(H_model_interictal[idxc[6],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[2,0].set_title('Channel '+electrode_names[6],fontsize=22)
ax2[2,0].set_yticks([0,125,250,375,500])
ax2[2,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax2[2,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax2[2,0].tick_params(length=8)

img17 = ax2[2,1].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[7],:],axis=1)*np.expand_dims(H_model_interictal[idxc[7],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[2,1].set_title('Channel '+electrode_names[7],fontsize=22)
ax2[2,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax2[2,1].tick_params(length=8)

img18 = ax2[2,2].imshow(np.rot90(np.expand_dims(W_model_interictal[idxc[8],:],axis=1)*np.expand_dims(H_model_interictal[idxc[8],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[2,2].set_title('Channel '+electrode_names[8],fontsize=22)
ax2[2,2].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax2[2,2].tick_params(length=8)

colorbar2 = fig2.colorbar(img10, ax=ax2.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar2.ax.tick_params(length=8, labelsize=22)

fig2.text(0.5, 0.01, 'Time (min)', ha='center',fontsize=22)
fig2.text(0.02, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=22)
fig2.text(0.94, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=22)

fig2.text(0.055,0.68,r"$\textbf{A}$",fontsize=22)
fig2.text(0.335,0.68,r"$\textbf{B}$",fontsize=22)
fig2.text(0.6,0.68,r"$\textbf{C}$",fontsize=22)
fig2.text(0.055,0.39,r"$\textbf{D}$",fontsize=22)
fig2.text(0.335,0.39,r"$\textbf{E}$",fontsize=22)
fig2.text(0.6,0.39,r"$\textbf{F}$",fontsize=22)
fig2.text(0.055,0.089,r"$\textbf{G}$",fontsize=22)
fig2.text(0.335,0.089,r"$\textbf{H}$",fontsize=22)
fig2.text(0.6,0.089,r"$\textbf{I}$",fontsize=22)

plt.show()

# fig.savefig("figures/preictal_models.pdf", pad_inches=0.4)
# fig2.savefig("figures/interictal_models.pdf", pad_inches=0.4)
