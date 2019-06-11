
import pickle
import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt, gridspec, transforms
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

"""
This script makes three figures:
    * models of time and frequency components of preictal and interictal states
    * time-frequency models and corresponding spectrograms of preictal and interictal states
    * the combined figure of time and frequency components and a time-frequency model
"""

##################################################### loading and extracting information #################################################################

patient_id = '11502'
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_'+patient_id+'_extracted_seizures/'

# num_channels = 48
idx_channel = 0

# coefficients = np.zeros((100,num_channels,12))
#
# for idx in range(100):
#     with open(path+'prediction_models/smote_'+patient_id+"_"+str(idx)+".pickle", "rb") as f:
#         file = pickle.load(f)
#     coefficients[idx, :, : ] = file.coef_[0].reshape((num_channels,12))
# coefficients = coefficients.mean(axis=0)[idx_channel,:]
# freq_coeff = coefficients[:10]
# time_coeff = coefficients[9:]

model_interictal = sio.loadmat(path+"data/models_interictal/Model_baseline1_7.mat")
model_preictal = sio.loadmat(path+"data/models_preictal/Model_preictal_14.mat")

spectrogram_interictal = sio.loadmat(path+'data/spectrograms_interictal/spectrogram_baseline1_7.mat')["spectrogram_baseline_1"]
spectrogram_preictal = sio.loadmat(path+'data/spectrograms_preictal/spectrogram_preictal_14.mat')["spectrogram_preictal"]

W_interictal = model_interictal["W_baseline"]
H_interictal = model_interictal["H_baseline"]
H_model_interictal = model_interictal["H_model_baseline"]
W_model_interictal = model_interictal["W_model_baseline"]

W_preictal = model_preictal["W_preictal"]
H_preictal = model_preictal["H_preictal"]
H_model_preictal = model_preictal["H_model_preictal"]
W_model_preictal = model_preictal["W_model_preictal"]

########################################################## extraction finished ######################################################################

# plotting the models
fig1 = plt.figure(figsize=(16,8))
gr = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])
fig1.subplots_adjust(left=0.1, bottom=0.10, right=0.96, top=0.92, wspace=0.17, hspace=0.27)

ax10 = fig1.add_subplot(gr[0,0])
ax10.plot(W_preictal[idx_channel,:], linewidth=2, color="royalblue", alpha=0.6)
ax10.plot(W_model_preictal[idx_channel,:],'--', linewidth=2, color='b', alpha=0.8)

ax10.set_title('Preictal state',fontsize=28)

ax10.set_xlabel('Time (min)',fontsize=22)
ax10.set_ylabel('Time coefficients',fontsize=22)

ax10.set_xticks([0,6,12,18,24,28])
ax10.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax10.set_yticks([5,15,25,35,45])
ax10.set_yticklabels([5,15,25,35,45],fontsize=22)
ax10.set_ylim([-1, 47])
ax10.tick_params(length=8)

ax20 = fig1.add_subplot(gr[0,1], sharey=ax10)
ax20.plot(W_interictal[idx_channel,:], linewidth=2, color="royalblue", alpha=0.6)
ax20.plot(W_model_interictal[idx_channel,:],'--', linewidth=2, color='b', alpha=0.8)
ax20.set_title('Interictal state',fontsize=28)
ax20.set_xlabel('Time (min)',fontsize=22)

ax20.set_xticks([0,6,12,18,24,28])
ax20.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax20.tick_params(axis='y', which='both', top=False, labelsize=22)
ax20.tick_params(length=8)

ax20.legend(('Time \n component', 'Model of a \n time component'), ncol=2, fontsize=18, loc="lower left")

ax30 = fig1.add_subplot(gr[1,0])
ax30.plot(H_preictal[idx_channel,:], color='lightcoral', linewidth=2, alpha=0.8)
ax30.plot(H_model_preictal[idx_channel,:],'--', linewidth=2, color="r", alpha=0.8)
ax30.set_ylabel('Frequency coefficients',fontsize=22)

ax30.set_xticks([0,125,250,375,500])
ax30.set_xticklabels([0,32,64,96,128],fontsize=22)
ax30.set_yticks([0.0,0.05,0.10,0.15,0.20])
ax30.set_yticklabels([0.0,0.05,0.10,0.15,0.20],fontsize=22)
ax30.set_ylim([-0.03, 0.17])
ax30.set_xlabel('Frequency (Hz)',fontsize=22)
ax30.tick_params(length=8)

ax40 = fig1.add_subplot(gr[1,1])
ax40.plot(H_interictal[idx_channel,:], linewidth=2, color='lightcoral', alpha=0.8)
ax40.plot(H_model_interictal[idx_channel,:],'--', linewidth=2, color="r", alpha=0.8)
ax40.set_xlabel('Frequency (Hz)',fontsize=26)

ax40.set_xticks([0,125,250,375,500])
ax40.set_xticklabels([0,32,64,96,128],fontsize=22)
ax40.set_yticks([0.042,0.044,0.046,0.048])
ax40.set_yticklabels([0.042,0.044,0.046,0.048],fontsize=22)
ax40.tick_params(length=8)

ax40.legend(('Frequency \n component','Model of a \n frequency component'), ncol=2, fontsize=18, loc="lower left")

fig1.text(0.08,0.555,r"$\textbf{A}$",fontsize=26)
fig1.text(0.545,0.555,r"$\textbf{B}$",fontsize=26)
fig1.text(0.08,0.1,r"$\textbf{C}$",fontsize=26)
fig1.text(0.545,0.1,r"$\textbf{D}$",fontsize=26)

#####################################################################################################################

# plotting models and corresponding spectrograms

kill_IDX = list(np.linspace(195,205,11,dtype=int))
spectrogram_interictal = np.delete(spectrogram_interictal,kill_IDX,axis=2)
spectrogram_preictal = np.delete(spectrogram_preictal,kill_IDX,axis=2)

fig2, ax2 = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,8))
fig2.subplots_adjust(left=0.08, bottom=0.11, right=0.94, top=0.88, wspace=0.1, hspace=0.24)

img1 = ax2[0,0].imshow(np.rot90(np.expand_dims(W_model_preictal[idx_channel,:],axis=1)*np.expand_dims(H_model_preictal[idx_channel,:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[0,0].set_title('Time-frequency model (preictal state)',fontsize=24)
ax2[0,0].set_ylabel('Frequency (Hz)', fontsize=26)
ax2[0,0].set_yticks([0,125,250,375,500])
ax2[0,0].set_yticklabels([0,32,64,96,128], fontsize=22)
ax2[0,0].tick_params(length=8)

img2 = ax2[0,1].imshow(np.rot90(spectrogram_preictal[0]),cmap='RdBu_r',aspect='auto',vmin=0.5, vmax=1.5, extent=[0,5,0,510])
ax2[0,1].set_title('Spectrogram (preictal state)',fontsize=26)
ax2[0,1].tick_params(length=8)

img3 = ax2[1,0].imshow(np.rot90(np.expand_dims(W_model_interictal[idx_channel,:],axis=1)*np.expand_dims(H_model_interictal[idx_channel,:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,5,0,510])

ax2[1,0].set_title('Time-frequency model (interictal state)',fontsize=24)
ax2[1,0].set_xlabel('Time (min)', fontsize=26)
ax2[1,0].set_xticklabels([-5,-4,-3,-2,-1,0], fontsize=22)
ax2[1,0].set_ylabel('Frequency (Hz)', fontsize=26)
ax2[1,0].set_yticks([0,125,250,375,500])
ax2[1,0].set_yticklabels([0,32,64,96,128], fontsize=22)
ax2[1,0].tick_params(length=8)

img4 = ax2[1,1].imshow(np.rot90(spectrogram_interictal[0]),cmap='RdBu_r',aspect='auto',vmin=0.5, vmax=1.5, extent=[0,5,0,510])
ax2[1,1].set_title('Spectrogram (interictal state)',fontsize=24)
ax2[1,1].set_xlabel('Time (min)',fontsize=26)
ax2[1,1].set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=22)
ax2[1,1].tick_params(length=8)

m = fig2.colorbar(img1, ax=ax2.ravel().tolist(), fraction=0.046, pad=0.03)
m.ax.tick_params(length=8, labelsize=22)

fig2.text(0.04,0.54,r"$\textbf{A}$", fontsize=26)
fig2.text(0.475,0.54,r"$\textbf{B}$", fontsize=26)
fig2.text(0.04,0.1,r"$\textbf{C}$", fontsize=26)
fig2.text(0.475,0.1,r"$\textbf{D}$", fontsize=26)
fig2.text(0.95, 0.5, 'Relative power', va='center', rotation='vertical', fontsize=26)

######################################################################################################################33

# plotting the combined figure
fig3 = plt.figure(figsize=(16,8))
gr = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 5], height_ratios=[5, 1])
fig3.subplots_adjust(left=0.11, bottom=0.16, right=0.8, top=0.91, wspace=0, hspace=0)

# defining the grids
ax3 = fig3.add_subplot(gr[0, 0])
ax4 = fig3.add_subplot(gr[0, 1], sharey=ax3)
ax5 = fig3.add_subplot(gr[1, 1], sharex=ax4)
ax5.set_xlim([0, W_preictal.shape[1]-1])

for tick in ax4.get_xticklabels()+ax4.get_yticklabels():
    tick.set_visible(False)

# actual plotting
y = np.arange(H_preictal.shape[1])
freq_comp = ax3.plot(H_preictal[idx_channel,:], y, linewidth=2, color='lightcoral', alpha=0.8)
freq_model = ax3.plot(H_model_preictal[idx_channel,:], y, '--', linewidth=2, color="r", alpha=0.8)

ax3.set_ylabel('Frequency (Hz)',fontsize=20)
ax3.set_yticks([0,125,250,375,500])
ax3.set_yticklabels([0,32,64,96,128],fontsize=20)

ax3.set_xlabel('Frequency coefficients',fontsize=20)
ax3.set_xlim([0.15, 0.0])
ax3.set_xticks([0,0.09,0.18])
ax3.set_xticklabels([0,0.09,0.18],fontsize=20)
ax3.tick_params(length=8)

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position("top")

img5 = ax4.imshow(np.rot90(np.expand_dims(W_model_preictal[idx_channel,:],axis=1)*np.expand_dims(H_model_preictal[idx_channel,:],axis=1).T),
cmap='RdBu_r',aspect='auto', vmin=0.5, vmax=1.5, extent=[0,30, 0, 510])

time_comp = ax5.plot(W_preictal[idx_channel,:], linewidth=2, color="royalblue", alpha=0.6)
time_model = ax5.plot(W_model_preictal[idx_channel,:],'--', linewidth=2, color="b", alpha=0.8)

ax5.set_xlabel('Time (min)', fontsize=20)
ax5.text(29.5, -10, 'Time coefficients', rotation=270, fontsize=20)
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position("right")

ax5.set_ylim([50, 0])
ax5.set_yticks([0,20,40])
ax5.set_yticklabels([0,20,40],fontsize=20)
ax5.set_xticks([0,6,12,18,24,28])
ax5.set_xticklabels([-5,-4,-3,-2,-1,0],fontsize=20)
ax5.tick_params(length=8)

ax5.legend([freq_comp[0], freq_model[0], time_comp[0], time_model[0]],
['Frequency component','Frequency model','Time component','Time model'],
bbox_to_anchor=(1.01,-0.7), ncol = 4, fontsize=18)

cbaxes = fig3.add_axes([0.88, 0.1, 0.02, 0.8])
cb = plt.colorbar(img5, ax=ax10, cax=cbaxes)
cb.ax.tick_params(length=8, labelsize=20)
fig3.text(0.93, 0.5, 'Relative power', va='center', rotation='vertical',fontsize=26)

##############################################################################################################################

# fig1.savefig("figures/components.pdf", pad_inches=0.4)
# fig2.savefig("figures/models.pdf", pad_inches=0.4)
# fig3.savefig("figures/model_signature_combination.pdf", pad_inches=0.4)

plt.show()
