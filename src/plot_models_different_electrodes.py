
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

patient_id = ''     # patient id goes here
path = ''           # path goes here

files_interictal = os.listdir(path+'/data/models_interictal/')
files_preictal = os.listdir(path+'/data/models_preictal/')

dict_models_interictal = sio.loadmat(path+'/data/models_interictal/'+files_interictal[0])
dict_models_preictal = sio.loadmat(path+'/data/models_preictal/'+files_preictal[0])

W_interictal = dict_models_interictal["W_interictal"]
H_interictal = dict_models_interictal["H_interictal"]
H_model_interictal = dict_models_interictal["H_model_interictal"]
W_model_interictal = dict_models_interictal["W_model_interictal"]

W_preictal = dict_models_preictal["W_preictal"]
H_preictal = dict_models_preictal["H_preictal"]
H_model_preictal = dict_models_preictal["H_model_preictal"]
W_model_preictal = dict_models_preictal["W_model_preictal"]

idxc = np.arange(0,9)
electrode_names = []        # electrode names go here

############################################################### plotting preictal models ###########################################################################

fig1, ax1 = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(16,8))
fig1.subplots_adjust(left=0.09, bottom=0.1, right=0.93, top=0.92, wspace=0.12, hspace=0.3)

img1 = ax1[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[0],:],axis=1)*np.expand_dims(H_model_preictal[idxc[0],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[0,0].set_title('Channel '+electrode_names[0],fontsize=22)
ax1[0,0].set_yticks([0,125,250,375,500])
ax1[0,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax1[0,0].tick_params(length=8)

img2 = ax1[0,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[1],:],axis=1)*np.expand_dims(H_model_preictal[idxc[1],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[0,1].set_title('Channel '+electrode_names[1],fontsize=22)
ax1[0,1].tick_params(length=8)

img3 = ax1[0,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[2],:],axis=1)*np.expand_dims(H_model_preictal[idxc[2],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[0,2].set_title('Channel '+electrode_names[2],fontsize=22)
ax1[0,2].tick_params(length=8)

img4 = ax1[1,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[3],:],axis=1)*np.expand_dims(H_model_preictal[idxc[3],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[1,0].set_title('Channel '+electrode_names[3],fontsize=22)
ax1[1,0].set_yticks([0,125,250,375,500])
ax1[1,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax1[1,0].tick_params(length=8)

img5 = ax1[1,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[4],:],axis=1)*np.expand_dims(H_model_preictal[idxc[4],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[1,1].set_title('Channel '+electrode_names[4],fontsize=22)
ax1[1,1].tick_params(length=8)

img6 = ax1[1,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[5],:],axis=1)*np.expand_dims(H_model_preictal[idxc[5],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[1,2].set_title('Channel '+electrode_names[5],fontsize=22)
ax1[1,2].tick_params(length=8)

img7 = ax1[2,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[6],:],axis=1)*np.expand_dims(H_model_preictal[idxc[6],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[2,0].set_title('Channel '+electrode_names[6],fontsize=22)
ax1[2,0].set_yticks([0,125,250,375,500])
ax1[2,0].set_yticklabels([0,32,64,96,128],fontsize=22)
ax1[2,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax1[2,0].tick_params(length=8)

img8 = ax1[2,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[7],:],axis=1)*np.expand_dims(H_model_preictal[idxc[7],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[2,1].set_title('Channel '+electrode_names[7],fontsize=22)
ax1[2,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax1[2,1].tick_params(length=8)

img9 = ax1[2,2].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc[8],:],axis=1)*np.expand_dims(H_model_preictal[idxc[8],:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax1[2,2].set_title('Channel '+electrode_names[8],fontsize=22)
ax1[2,2].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax1[2,2].tick_params(length=8)

colorbar1 = fig1.colorbar(img1, ax=ax1.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar1.ax1.tick_params(labelsize=22)
colorbar1.ax1.tick_params(length=8, labelsize=22)

fig1.text(0.5, 0.01, 'Time (min)', ha='center',fontsize=22)
fig1.text(0.02, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=22)
fig1.text(0.95, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=22)

fig1.text(0.055,0.68,r"$\textbf{A}$",fontsize=22)
fig1.text(0.335,0.68,r"$\textbf{B}$",fontsize=22)
fig1.text(0.6,0.68,r"$\textbf{C}$",fontsize=22)
fig1.text(0.055,0.39,r"$\textbf{D}$",fontsize=22)
fig1.text(0.335,0.39,r"$\textbf{E}$",fontsize=22)
fig1.text(0.6,0.39,r"$\textbf{F}$",fontsize=22)
fig1.text(0.055,0.089,r"$\textbf{G}$",fontsize=22)
fig1.text(0.335,0.089,r"$\textbf{H}$",fontsize=22)
fig1.text(0.6,0.089,r"$\textbf{I}$",fontsize=22)

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

# fig1.savefig("figures/preictal_models.pdf", pad_inches=0.4)
# fig2.savefig("figures/interictal_models.pdf", pad_inches=0.4)
