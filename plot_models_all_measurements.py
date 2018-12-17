
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
This script makes two plots which dipslay models of time-frequency signatures of a preictal state for all measurements for channel 1. 
"""

ident = '11502'
patient_id = '11502'
run_nr = '1'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/'+run_nr+'/'

preictal_train = preictal_directory+'train/'
preictal_test = preictal_directory+'test/'
preictal_out = preictal_directory+'out-of-sample/'

files_preictal_train = os.listdir(path_directory+preictal_train)
files_preictal_test = os.listdir(path_directory+preictal_test)
files_preictal_out = os.listdir(path_directory+preictal_out)

loadings_preictal_train = []
loadings_preictal_test = []
loadings_preictal_out = []

for n in range(len(files_preictal_train)):
    loadings_preictal_train.append(sio.loadmat(path_directory+preictal_train+files_preictal_train[n]))

for n in range(len(files_preictal_test)):
    loadings_preictal_test.append(sio.loadmat(path_directory+preictal_test+files_preictal_test[n]))

for n in range(len(files_preictal_out)):
    loadings_preictal_out.append(sio.loadmat(path_directory+preictal_out+files_preictal_out[n]))


idxc = 0                    # channel HR1
vmin_pre = 0.5
vmax_pre = 1.5              # maximum value for colormap for preictal

# plotting the preictal models
fig1 = plt.figure(figsize=(19.5,10.2))

ax1 = plt.subplot2grid((4, 3), (0, 0))
ax2 = plt.subplot2grid((4, 3), (0, 1))
ax3 = plt.subplot2grid((4, 3), (0, 2))
ax4 = plt.subplot2grid((4, 3), (1, 0))
ax5 = plt.subplot2grid((4, 3), (1, 1))
ax6 = plt.subplot2grid((4, 3), (1, 2))
ax7 = plt.subplot2grid((4, 3), (2, 0))
ax8 = plt.subplot2grid((4, 3), (2, 1))
ax9 = plt.subplot2grid((4, 3), (2, 2))
ax10 = plt.subplot2grid((4, 3), (3, 0))
ax11 = plt.subplot2grid((4, 3), (3, 1))
ax12 = plt.subplot2grid((4, 3), (3, 2))

plt.subplots_adjust(wspace = 0.2, hspace = 0.2) #make the figure look better

for tick in ax1.get_xticklabels():
    tick.set_visible(False)

for tick in ax2.get_xticklabels()+ax2.get_yticklabels():
    tick.set_visible(False)

for tick in ax3.get_xticklabels()+ax3.get_yticklabels():
    tick.set_visible(False)

for tick in ax4.get_xticklabels():
    tick.set_visible(False)

for tick in ax5.get_xticklabels()+ax5.get_yticklabels():
    tick.set_visible(False)

for tick in ax6.get_xticklabels()+ax6.get_yticklabels():
    tick.set_visible(False)

for tick in ax7.get_xticklabels():
    tick.set_visible(False)

for tick in ax8.get_xticklabels()+ax8.get_yticklabels():
    tick.set_visible(False)

for tick in ax9.get_yticklabels()+ax9.get_xticklabels():
    tick.set_visible(False)

for tick in ax11.get_yticklabels():
    tick.set_visible(False)

for tick in ax12.get_yticklabels():
    tick.set_visible(False)

img1=ax1.imshow(np.rot90(np.expand_dims(loadings_preictal_train[0]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[0]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax1.set_title('Measurement 1',fontsize=18)
ax1.set_yticks([0,125,250,375,500])
ax1.set_yticklabels([0,32,64,96,128],fontsize=16)
fig1.text(0.055,0.725,r"$\textbf{A}$",fontsize=18)

ax2.imshow(np.rot90(np.expand_dims(loadings_preictal_train[1]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[1]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax2.set_title('Measurement 2',fontsize=18)
fig1.text(0.35,0.725,r"$\textbf{B}$",fontsize=18)

ax3.imshow(np.rot90(np.expand_dims(loadings_preictal_train[2]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[2]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax3.set_title('Measurement 3',fontsize=18)
fig1.text(0.64,0.725,r"$\textbf{C}$",fontsize=18)

ax4.imshow(np.rot90(np.expand_dims(loadings_preictal_train[3]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[3]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax4.set_title('Measurement 4',fontsize=18)
ax4.set_yticks([0,125,250,375,500])
ax4.set_yticklabels([0,32,64,96,128],fontsize=16)
fig1.text(0.055,0.52,r"$\textbf{D}$",fontsize=18)

ax5.imshow(np.rot90(np.expand_dims(loadings_preictal_train[4]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[4]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax5.set_title('Measurement 5',fontsize=18)
fig1.text(0.35,0.52,r"$\textbf{E}$",fontsize=18)

ax6.imshow(np.rot90(np.expand_dims(loadings_preictal_train[5]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[5]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax6.set_title('Measurement 6',fontsize=18)
fig1.text(0.64,0.52,r"$\textbf{F}$",fontsize=18)

ax7.imshow(np.rot90(np.expand_dims(loadings_preictal_train[6]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[6]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax7.set_title('Measurement 7',fontsize=18)
ax7.set_yticks([0,125,250,375,500])
ax7.set_yticklabels([0,32,64,96,128],fontsize=16)
ax7.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig1.text(0.055,0.31,r"$\textbf{G}$",fontsize=18)

ax8.imshow(np.rot90(np.expand_dims(loadings_preictal_train[7]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[7]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax8.set_title('Measurement 8',fontsize=18)
fig1.text(0.35,0.31,r"$\textbf{H}$",fontsize=18)

ax9.imshow(np.rot90(np.expand_dims(loadings_preictal_train[8]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[8]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax9.set_title('Measurement 9',fontsize=18)
fig1.text(0.64,0.31,r"$\textbf{I}$",fontsize=18)

ax10.imshow(np.rot90(np.expand_dims(loadings_preictal_train[9]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[9]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax10.set_title('Measurement 10',fontsize=18)
ax10.set_yticks([0,125,250,375,500])
ax10.set_yticklabels([0,32,64,96,128],fontsize=16)
ax10.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig1.text(0.055,0.1,r"$\textbf{J}$",fontsize=18)

ax11.imshow(np.rot90(np.expand_dims(loadings_preictal_train[10]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[10]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax11.set_title('Measurement 11',fontsize=18)
ax11.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig1.text(0.35,0.1,r"$\textbf{K}$",fontsize=18)

ax12.imshow(np.rot90(np.expand_dims(loadings_preictal_train[11]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[11]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax12.set_title('Measurement 12',fontsize=18)
ax12.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig1.text(0.64,0.1,r"$\textbf{L}$",fontsize=18)

fig1.suptitle('Models of time-frequency signatures of a preictal state (CH: HR1, patient 1)', ha='center', fontsize=20)
fig1.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=18)
fig1.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=18)
fig1.tight_layout()
fig1.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)

cbaxes1 = fig1.add_axes([0.93, 0.1, 0.01, 0.8])
cb1 = plt.colorbar(img1,ax=ax1, cax=cbaxes1)
cb1.ax.tick_params(labelsize=16)

fig1.text(0.96, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig1.show()


# plotting the preictal models
fig2 = plt.figure(figsize=(19.5,10.2))

ax21 = plt.subplot2grid((4, 3), (0, 0))
ax22 = plt.subplot2grid((4, 3), (0, 1))
ax23 = plt.subplot2grid((4, 3), (0, 2))
ax24 = plt.subplot2grid((4, 3), (1, 0))
ax25 = plt.subplot2grid((4, 3), (1, 1))
ax26 = plt.subplot2grid((4, 3), (1, 2))
ax27 = plt.subplot2grid((4, 3), (2, 0))
ax28 = plt.subplot2grid((4, 3), (2, 1))
ax29 = plt.subplot2grid((4, 3), (2, 2))
ax30 = plt.subplot2grid((4, 3), (3, 0))
ax31 = plt.subplot2grid((4, 3), (3, 1))

for tick in ax21.get_xticklabels():
    tick.set_visible(False)

for tick in ax22.get_xticklabels()+ax22.get_yticklabels():
    tick.set_visible(False)

for tick in ax23.get_xticklabels()+ax23.get_yticklabels():
    tick.set_visible(False)

for tick in ax24.get_xticklabels():
    tick.set_visible(False)

for tick in ax25.get_xticklabels()+ax25.get_yticklabels():
    tick.set_visible(False)

for tick in ax26.get_xticklabels()+ax26.get_yticklabels():
    tick.set_visible(False)

for tick in ax27.get_xticklabels():
    tick.set_visible(False)

for tick in ax28.get_xticklabels()+ax28.get_yticklabels():
    tick.set_visible(False)

for tick in ax29.get_yticklabels():
    tick.set_visible(False)

for tick in ax31.get_yticklabels():
    tick.set_visible(False)

plt.subplots_adjust(wspace = 0.2, hspace = 0.2) #make the figure look better

ax21.imshow(np.rot90(np.expand_dims(loadings_preictal_train[12]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[12]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax21.set_title('Measurement 13',fontsize=18)
ax21.set_yticks([0,125,250,375,500])
ax21.set_yticklabels([0,32,64,96,128],fontsize=16)
fig2.text(0.055,0.725,r"$\textbf{A}$",fontsize=18)

ax22.imshow(np.rot90(np.expand_dims(loadings_preictal_train[13]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[13]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax22.set_title('Measurement 14',fontsize=18)
fig2.text(0.35,0.725,r"$\textbf{B}$",fontsize=18)

ax23.imshow(np.rot90(np.expand_dims(loadings_preictal_train[14]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[14]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax23.set_title('Measurement 15',fontsize=18)
fig2.text(0.64,0.725,r"$\textbf{C}$",fontsize=18)

ax24.imshow(np.rot90(np.expand_dims(loadings_preictal_train[15]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_train[15]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax24.set_title('Measurement 16',fontsize=18)
ax24.set_yticks([0,125,250,375,500])
ax24.set_yticklabels([0,32,64,96,128],fontsize=16)
fig2.text(0.055,0.52,r"$\textbf{D}$",fontsize=18)

ax25.imshow(np.rot90(np.expand_dims(loadings_preictal_test[0]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_test[0]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax25.set_title('Measurement 17',fontsize=18)
fig2.text(0.35,0.52,r"$\textbf{E}$",fontsize=18)

ax26.imshow(np.rot90(np.expand_dims(loadings_preictal_test[1]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_test[1]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax26.set_title('Measurement 18',fontsize=18)
fig2.text(0.64,0.52,r"$\textbf{F}$",fontsize=18)

ax27.imshow(np.rot90(np.expand_dims(loadings_preictal_test[2]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_test[2]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax27.set_title('Measurement 19',fontsize=18)
ax27.set_yticks([0,125,250,375,500])
ax27.set_yticklabels([0,32,64,96,128],fontsize=16)
ax27.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.055,0.31,r"$\textbf{G}$",fontsize=18)

ax28.imshow(np.rot90(np.expand_dims(loadings_preictal_test[3]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_test[3]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax28.set_title('Measurement 20',fontsize=18)
fig2.text(0.35,0.31,r"$\textbf{H}$",fontsize=18)

ax29.imshow(np.rot90(np.expand_dims(loadings_preictal_test[4]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_test[4]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax29.set_title('Measurement 21',fontsize=18)
ax29.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.64,0.31,r"$\textbf{I}$",fontsize=18)

ax30.imshow(np.rot90(np.expand_dims(loadings_preictal_out[0]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_out[0]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax30.set_title('Measurement 22',fontsize=18)
ax30.set_yticks([0,125,250,375,500])
ax30.set_yticklabels([0,32,64,96,128],fontsize=16)
ax30.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.055,0.1,r"$\textbf{J}$",fontsize=18)

ax31.imshow(np.rot90(np.expand_dims(loadings_preictal_out[1]["W_model_preictal"][idxc,:],axis=1)*np.expand_dims(loadings_preictal_out[1]["H_model_preictal"][idxc,:],axis=1).T),cmap='RdBu_r',aspect='auto',vmin=vmin_pre, vmax=vmax_pre, extent=[0,5,0,510])
ax31.set_title('Measurement 23',fontsize=18)
ax31.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=16)
fig2.text(0.35,0.1,r"$\textbf{K}$",fontsize=18)

fig2.suptitle('Models of time-frequency signatures of a preictal state (CH: HR1, patient 1)', ha='center', fontsize=20)
fig2.text(0.5, 0.04, 'Time (min)', ha='center',fontsize=18)
fig2.text(0.04, 0.5, 'Frequency (Hz)', va='center', rotation='vertical',fontsize=18)
fig2.tight_layout()
fig2.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)

cbaxes = fig2.add_axes([0.93, 0.1, 0.01, 0.8])
cb = plt.colorbar(img1,ax=ax1, cax=cbaxes)
cb.ax.tick_params(labelsize=16)

fig2.text(0.96, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=18)
fig2.show()
