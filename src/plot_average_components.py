
import os
import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec, transforms
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

"""
This script plots average time and frequency components of preictal and interictal states.
"""

path = ''           # path goes here
patients = []       # patient ids go here
idxc = 0

W_preictal = []
H_preictal = []
W_interictal = []
H_interictal = []

W_preictal_avg = np.zeros((len(patients), 29))
H_preictal_avg = np.zeros((len(patients), 502))
W_interictal_avg = np.zeros((len(patients), 29))
H_interictal_avg = np.zeros((len(patients), 502))

for idx, patient_id in enumerate(patients):

    files_preictal = os.listdir(path+'data/models_preictal/')
    files_interictal = os.listdir(path+'data/models_interictal/')

    for n in range(len(files_preictal)):
        dict_models_preictal = sio.loadmat(path+'data/models_preictal/'+files_preictal[n])
        W_preictal.append(dict_models_preictal["W_model_preictal"][idxc,:])
        H_preictal.append(dict_models_preictal["H_model_preictal"][idxc,:])

    for n in range(len(files_interictal)):
        dict_models_interictal = sio.loadmat(path+'data/models_interictal/'+files_interictal[n])
        H_interictal.append(dict_models_interictal["H_model_interictal"][idxc,:])
        W_interictal.append(dict_models_interictal["W_model_interictal"][idxc,:])

    W_preictal_avg[idx, :] = np.mean(W_preictal,axis=0)
    H_preictal_avg[idx, :] = np.mean(H_preictal,axis=0)

    W_interictal_avg[idx, :] = np.mean(W_interictal,axis=0)
    H_interictal_avg[idx, :] = np.mean(H_interictal,axis=0)

# plotting the models
fig1 = plt.figure(figsize=(16,8))
gr = gridspec.GridSpec(nrows=2, ncols=6)
fig1.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.95, wspace=0.3, hspace=0.23)

ax1 = fig1.add_subplot(gr[0,0])
ax1.plot(W_preictal_avg[0,:],'--', linewidth=2, color='b', alpha=0.8)
ax1.plot(W_interictal_avg[0,:],'+', linewidth=2, color='b', alpha=0.5)
ax1.set_ylabel('Time coefficients',fontsize=20)
ax1.set_ylim([18, 30])
ax1.set_yticks([20,24, 28])
ax1.set_yticklabels([20,24, 28],fontsize=16)
ax1.set_xticks([0,6,12,18,24,28])
ax1.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax1.tick_params(axis="y", labelleft=True, labelsize=16)
ax1.set_xlabel('Time (min)',fontsize=20)
ax1.set_title("Patient 1 (CH: *)", fontsize=20)         # channel name goes instead of *
ax1.tick_params(length=6)

ax2 = fig1.add_subplot(gr[0,1])
ax2.plot(W_preictal_avg[1,:],'--', linewidth=2, color='b', alpha=0.8)
ax2.plot(W_interictal_avg[1,:],'+', linewidth=2, color='b', alpha=0.5)
ax2.set_xticks([0,6,12,18,24,28])
ax2.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax2.set_ylim([18, 30])
ax2.set_yticks([20,24, 28])
ax2.set_yticklabels([20,24, 28],fontsize=16)
ax2.tick_params(axis="y", labelleft=True, labelsize=16)
ax2.set_xlabel('Time (min)',fontsize=20)
ax2.set_title("Patient 2 (CH: *)", fontsize=20)         # channel name goes instead of *
ax2.tick_params(length=6)

ax3 = fig1.add_subplot(gr[0,2])
ax3.plot(W_preictal_avg[2,:],'--', linewidth=2, color='b', alpha=0.8)
ax3.plot(W_interictal_avg[2,:],'+', linewidth=2, color='b', alpha=0.5)
ax3.set_xticks([0,6,12,18,24,28])
ax3.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax3.set_ylim([18, 30])
ax3.set_yticks([20,24, 28])
ax3.set_yticklabels([20,24, 28],fontsize=16)
ax3.tick_params(axis="y", labelleft=True, labelsize=16)
ax3.set_xlabel('Time (min)',fontsize=20)
ax3.set_title("Patient 3 (CH: *)", fontsize=20)         # channel name goes instead of *
ax3.tick_params(length=6)

ax4 = fig1.add_subplot(gr[0,3])
ax4.plot(W_preictal_avg[3,:],'--', linewidth=2, color='b', alpha=0.8)
ax4.plot(W_interictal_avg[3,:],'+', linewidth=2, color='b', alpha=0.5)
ax4.set_xticks([0,6,12,18,24,28])
ax4.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax4.set_ylim([18, 30])
ax4.set_yticks([20,24, 28])
ax4.set_yticklabels([20,24, 28],fontsize=16)
ax4.tick_params(axis="y", labelleft=True, labelsize=16)
ax4.set_xlabel('Time (min)',fontsize=20)
ax4.set_title("Patient 4 (CH: *)", fontsize=20)         # channel name goes instead of *
ax4.tick_params(length=6)

ax5 = fig1.add_subplot(gr[0,4])
ax5.plot(W_preictal_avg[4,:],'--', linewidth=2, color='b', alpha=0.8)
ax5.plot(W_interictal_avg[4,:],'+', linewidth=2, color='b', alpha=0.5)
ax5.set_ylim([10, 70])
ax5.set_yticks([20,40,60])
ax5.set_yticklabels([20,40,60],fontsize=16)
ax5.set_xticks([0,6,12,18,24,28])
ax5.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax5.tick_params(axis="y", labelleft=True, labelsize=16)
ax5.set_xlabel('Time (min)',fontsize=20)
ax5.set_title("Patient 5 (CH: *)", fontsize=20)         # channel name goes instead of *
ax5.tick_params(length=6)

ax6 = fig1.add_subplot(gr[0,5])
ax6.plot(W_preictal_avg[5,:],'--', linewidth=2, color='b', alpha=0.8)
ax6.plot(W_interictal_avg[5,:],'+', linewidth=2, color='b', alpha=0.5)
ax6.set_ylim([5, 80])
ax6.set_yticks([20,45,70])
ax6.set_yticklabels([20,45,70],fontsize=16)
ax6.set_xticks([0,6,12,18,24,28])
ax6.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax6.tick_params(axis="y", labelleft=True, labelsize=16)
ax6.set_xlabel('Time (min)',fontsize=20)
ax6.set_title("Patient 6 (CH: *)", fontsize=20)        # channel name goes instead of *
ax6.legend(["Preictal", "Interictal"], fontsize=17, loc="center right")
ax6.tick_params(length=6)

ax7 = fig1.add_subplot(gr[1,0])
ax7.plot(H_preictal_avg[0,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax7.plot(H_interictal_avg[0,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax7.set_ylabel('Frequency coefficients',fontsize=20)
ax7.set_xlabel('Frequency (Hz)',fontsize=20)
ax7.set_ylim([0.027, 0.053])
ax7.set_yticks([0.03,0.04,0.05])
ax7.set_yticklabels([0.03,0.04,0.05],fontsize=16)
ax7.set_xticks([0,25,50])
ax7.set_xticklabels([0,64,128],fontsize=16)
ax7.tick_params(length=6)

ax8 = fig1.add_subplot(gr[1,1])
ax8.plot(H_preictal_avg[1,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax8.plot(H_interictal_avg[1,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax8.set_xlabel('Frequency (Hz)',fontsize=20)
ax8.set_ylim([0.027, 0.053])
ax8.set_yticks([0.03,0.04,0.05])
ax8.set_yticklabels([0.03,0.04,0.05],fontsize=16)
ax8.set_xticks([0,25,50])
ax8.set_xticklabels([0,64,128],fontsize=16)
ax8.tick_params(length=6)

ax9 = fig1.add_subplot(gr[1,2])
ax9.plot(H_preictal_avg[2,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax9.plot(H_interictal_avg[2,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax9.set_xlabel('Frequency (Hz)',fontsize=20)
ax9.set_ylim([0.03, 0.055])
ax9.set_ylim([0.027, 0.053])
ax9.set_yticks([0.03,0.04,0.05])
ax9.set_yticklabels([0.03,0.04,0.05],fontsize=16)
ax9.set_xticks([0,25,50])
ax9.set_xticklabels([0,64,128],fontsize=16)
ax9.tick_params(length=6)

ax10 = fig1.add_subplot(gr[1,3])
ax10.plot(H_preictal_avg[3,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax10.plot(H_interictal_avg[3,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax10.set_xlabel('Frequency (Hz)',fontsize=20)
ax10.set_ylim([0.03, 0.055])
ax10.set_ylim([0.027, 0.053])
ax10.set_yticks([0.03,0.04,0.05])
ax10.set_yticklabels([0.03,0.04,0.05],fontsize=16)
ax10.set_xticks([0,25,50])
ax10.set_xticklabels([0,64,128],fontsize=16)
ax10.tick_params(length=6)

ax11 = fig1.add_subplot(gr[1,4])
ax11.plot(H_preictal_avg[4,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax11.plot(H_interictal_avg[4,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax11.set_xlabel('Frequency (Hz)',fontsize=20)
ax11.tick_params(axis="y", labelleft=True, labelsize=16)
ax11.tick_params(length=6)
ax11.set_yticks([0.04,0.08,0.12])
ax11.set_yticklabels([0.04,0.08,0.12],fontsize=16)
ax11.set_ylim([0.025, 0.13])
ax11.set_xticks([0,25,50])
ax11.set_xticklabels([0,64,128],fontsize=16)

ax12 = fig1.add_subplot(gr[1,5])
ax12.plot(H_preictal_avg[5,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax12.plot(H_interictal_avg[5,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax12.set_xlabel('Frequency (Hz)',fontsize=20)
ax12.tick_params(axis="y", labelleft=True, labelsize=16)
ax12.tick_params(length=6)
ax12.set_yticks([0.04,0.08,0.12])
ax12.set_yticklabels([0.04,0.08,0.12],fontsize=16)
ax12.set_ylim([0.025, 0.13])
ax12.set_xticks([0,25,50])
ax12.set_xticklabels([0,64,128],fontsize=16)
ax12.legend(["Preictal", "Interictal"], fontsize=17, loc="upper right")

fig1.text(0.043,0.56,r"$\textbf{A}$",fontsize=24)
fig1.text(0.204,0.56,r"$\textbf{B}$",fontsize=24)
fig1.text(0.363,0.56,r"$\textbf{C}$",fontsize=24)
fig1.text(0.52,0.56,r"$\textbf{D}$",fontsize=24)
fig1.text(0.683,0.56,r"$\textbf{E}$",fontsize=24)
fig1.text(0.844,0.56,r"$\textbf{F}$",fontsize=24)
fig1.text(0.043,0.08,r"$\textbf{G}$",fontsize=24)
fig1.text(0.203,0.08,r"$\textbf{H}$",fontsize=24)
fig1.text(0.369,0.08,r"$\textbf{I}$",fontsize=24)
fig1.text(0.527,0.08,r"$\textbf{J}$",fontsize=24)
fig1.text(0.681,0.08,r"$\textbf{K}$",fontsize=24)
fig1.text(0.844,0.08,r"$\textbf{L}$",fontsize=24)

# fig1.savefig("figures/average_components.pdf", pad_inches=0.4)
plt.show()
