
from __future__ import unicode_literals
import os
import scipy.io as sio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt, gridspec, transforms
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script makes three figures:
    * models of time and frequency components of preictal and interictal states
    * models and time-frequency signatures of preictal and interictal states
    * the combined figure of time and frequency components and a model of time-frequency signatures
"""

ident = '11502'
patient_id = '11502'
run_nr = '1'
sample = 'train'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/'+run_nr+'/'+sample+'/'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/'+run_nr+'/'+sample+'/'

files_baseline = os.listdir(path_directory+baseline_directory)
files_preictal = os.listdir(path_directory+preictal_directory)

dict_models_baseline = sio.loadmat(path_directory+baseline_directory+'/'+files_baseline[0])
dict_models_preictal = sio.loadmat(path_directory+preictal_directory+'/'+files_preictal[0])

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

# plotting the models
fig1 = plt.figure(figsize=(19.5,10.2))
gr = gridspec.GridSpec(nrows=2, ncols=2,left=0.06,wspace=0.17, hspace=0.20, width_ratios=[1, 1], height_ratios=[1, 1])

# defining the grids
ax30 = fig1.add_subplot(gr[1,0])
ax10 = fig1.add_subplot(gr[0,0],sharex=ax30)
ax40 = fig1.add_subplot(gr[1,1])
ax20 = fig1.add_subplot(gr[0,1],sharex=ax40)

for tick in ax10.get_xticklabels():
    tick.set_visible(False)

for tick in ax20.get_xticklabels():
    tick.set_visible(False)

ax10.plot(W_preictal[idxc,:])
ax10.plot(W_model_preictal[idxc,:],'--', color='red')
ax10.set_title('Model of the time component of a preictal state (channel: HR1, patient 1)',fontsize=18)
ax10.set_ylabel('Time coefficients',fontsize=18)
ax10.set_yticks([5,15,25,35,45])
ax10.set_yticklabels([5,15,25,35,45],fontsize=16)
ax10.legend(('Time component','Model of a time component'),fontsize=14)
fig1.text(0.035,0.54,r"$\textbf{A}$",fontsize=18)

ax20.plot(H_preictal[idxc,:])
ax20.plot(H_model_preictal[idxc,:],'--', color='red')
ax20.set_title('Model of the frequency component of a preictal state (channel: HR1, patient 1)',fontsize=18)
ax20.set_ylabel('Frequency coefficients',fontsize=18)
ax20.set_yticks([0.0,0.05,0.10,0.15,0.20])
ax20.set_yticklabels([0.0,0.05,0.10,0.15,0.20],fontsize=16)
ax20.legend(('Frequency component','Model of a frequency component'),fontsize=14)
fig1.text(0.5,0.54,r"$\textbf{B}$",fontsize=18)

ax30.plot(W_baseline[idxc,:])
ax30.plot(W_model_baseline[idxc,:],'--', color='red')
ax30.set_title('Model of the time component of an interictal state (channel: HR1, patient 1)',fontsize=18)
ax30.set_xlabel('Time (min)',fontsize=18)
ax30.set_xticks([0,6,12,18,24,28])
ax30.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax30.set_ylabel('Time coefficients',fontsize=18)
ax30.set_yticks([5,15,25,35,45])
ax30.set_yticklabels([5,15,25,35,45],fontsize=16)
ax30.legend(('Time component','Model of the time component'),fontsize=14)
fig1.text(0.035,0.1,r"$\textbf{C}$",fontsize=18)

ax40.plot(H_baseline[idxc,:])
ax40.plot(H_model_baseline[idxc,:],'--', color='red')
ax40.set_title('Model of the frequency component of an interictal state (channel: HR1, patient 1)',fontsize=18)
ax40.set_xlabel('Frequency (Hz)',fontsize=18)
ax40.set_xticks([0,125,250,375,500])
ax40.set_xticklabels([0,32,64,96,128],fontsize=16)
ax40.set_ylabel('Frequency coefficients',fontsize=18)
ax40.set_yticks([0.0,0.05,0.10,0.15,0.20])
ax40.set_yticklabels([0.0,0.05,0.10,0.15,0.20],fontsize=16)
ax40.legend(('Frequency component','Model of the frequency component'),fontsize=14)
fig1.text(0.5,0.11,r"$\textbf{D}$",fontsize=18)
fig1.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
fig1.show()


# plotting the prototypes
fig2,ax2 = plt.subplots(2, 2,sharex=True,sharey=True,figsize=(19.5,10.2))
img1 = ax2[0,0].imshow(np.rot90(np.expand_dims(W_preictal[idxc,:],axis=1)*np.expand_dims(H_preictal[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax2[0,0].set_xlabel('Time (min)',fontsize=18)
ax2[0,0].set_ylabel('Frequency (Hz)',fontsize=18)
ax2[0,0].set_yticks([0,125,250,375,500])
ax2[0,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax2[0,0].set_title('Time-frequency signature of a preictal state (CH: HR1, patient 1)',fontsize=18)
fig2.text(0.025,0.54,r"$\textbf{A}$",fontsize=18)

img2 = ax2[0,1].imshow(np.rot90(np.expand_dims(W_model_preictal[idxc,:],axis=1)*np.expand_dims(H_model_preictal[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax2[0,1].set_xlabel('Time (min)',fontsize=18)
ax2[0,1].set_title('Model of the time-frequency signature of a preictal state (CH: HR1, patient 1)',fontsize=18)
fig2.text(0.455,0.54,r"$\textbf{B}$",fontsize=18)

img3 = ax2[1,0].imshow(np.rot90(np.expand_dims(W_baseline[idxc,:],axis=1)*np.expand_dims(H_baseline[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,0].set_xlabel('Time (min)',fontsize=18)
ax2[1,0].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax2[1,0].set_ylabel('Frequency (Hz)',fontsize=18)
ax2[1,0].set_yticks([0,125,250,375,500])
ax2[1,0].set_yticklabels([0,32,64,96,128],fontsize=16)
ax2[1,0].set_title('Time-frequency signature of an interictal state (CH: HR1, patient 1)',fontsize=18)
fig2.text(0.025,0.1,r"$\textbf{C}$",fontsize=18)

img4 = ax2[1,1].imshow(np.rot90(np.expand_dims(W_model_baseline[idxc,:],axis=1)*np.expand_dims(H_model_baseline[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_int, vmax=vmax_int, extent=[0,5,0,510])
ax2[1,1].set_xlabel('Time (min)',fontsize=18)
ax2[1,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax2[1,1].set_title('Model of the time-frequency signature of an interictal state (CH: HR1, patient 1)',fontsize=18)
fig2.text(0.455,0.1,r"$\textbf{D}$",fontsize=18)

fig2.tight_layout()
fig2.subplots_adjust(left=0.05, bottom=0.10, right=0.92, top=0.90, wspace=0.14, hspace=0.24)
m=fig2.colorbar(img1, ax=ax2.ravel().tolist(), fraction=0.046, pad=0.04)
m.ax.tick_params(labelsize=16)
fig2.text(0.93, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig2.show()


# plotting the combined figure
fig3 = plt.figure(figsize=(19.5,10.2))
gr = gridspec.GridSpec(nrows=2, ncols=2,left=0.06,wspace=0, hspace=0, width_ratios=[1, 5], height_ratios=[5, 1])

# defining the grids
ax3 = fig3.add_subplot(gr[0, 0])
ax4 = fig3.add_subplot(gr[0, 1], sharey=ax3)
ax5 = fig3.add_subplot(gr[1, 1], sharex=ax4)
ax5.set_xlim([0, W_preictal.shape[1]-1])

for tick in ax4.get_xticklabels()+ax4.get_yticklabels():
    tick.set_visible(False)

# actual plotting
y = np.arange(H_preictal.shape[1])
ax3.plot(H_preictal[idxc,:], y)
ax3.plot(H_model_preictal[idxc,:], y, '--', color='red')
ax3.set_ylabel('Frequency (Hz)',fontsize=18)
ax3.set_yticks([0,125,250,375,500])
ax3.set_yticklabels([0,32,64,96,128],fontsize=16)
ax3.set_xlabel('Frequency coefficients',fontsize=18)
ax3.set_xlim([0.15, 0.0])
ax3.set_xticks([0,0.09,0.18])
ax3.set_xticklabels([0,0.09,0.18],fontsize=16)
ax3.xaxis.tick_top()
ax3.xaxis.set_label_position("top")
ax3.legend(('Frequency component','Model of frequency component'), bbox_to_anchor=(1.0,-0.01),fontsize=14)

img5 = ax4.imshow(np.rot90(np.expand_dims(W_model_preictal[idxc,:],axis=1)*np.expand_dims(H_model_preictal[idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,30, 0, 510])
ax4.set_title('Model of the time-frequency signature of a preictal state (CH: HR1, patient 1)',fontsize=18)

ax5.plot(W_preictal[idxc,:], '.-')
ax5.plot(W_model_preictal[idxc,:],'--', color='red')
ax5.set_xlabel('Time (min)',fontsize=18)
ax5.text(29,0,'Time coefficients',rotation=270,fontsize=18)
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position("right")
ax5.set_ylim([50, 0])
ax5.set_yticks([0,20,40])
ax5.set_yticklabels([0,20,40],fontsize=16)
ax5.set_xticks([0,6,12,18,24,28])
ax5.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
ax5.legend(('Time component','Model of time component'), loc="upper right",fontsize=14)

cbaxes = fig3.add_axes([0.95, 0.1, 0.01, 0.8])
cb = plt.colorbar(img5,ax=ax10, cax=cbaxes)
cb.ax.tick_params(labelsize=16)
fig3.text(0.98, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig3.show()
