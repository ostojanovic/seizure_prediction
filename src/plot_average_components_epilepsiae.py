
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
This script plots average time and frequency components of preictal and interictal states of the EPILEPSIAE dataset.
"""

path = "*"         # path goes here
patients = []       # patient ids go here as strings in a list

W_preictal_avg = np.zeros((len(patients), 29))
H_preictal_avg = np.zeros((len(patients), 502))
W_interictal_avg = np.zeros((len(patients), 29))
H_interictal_avg = np.zeros((len(patients), 502))

for idx, patient_id in enumerate(patients):

    files_preictal = os.listdir("*") # path to a folder with preictal models goes here
    files_interictal = os.listdir("*") # path to a folder with interictal models goes here

    W_preictal = []
    H_preictal = []
    W_interictal = []
    H_interictal = []

    for n in range(len(files_preictal)):
        dict_models_preictal = sio.loadmat('*'+files_preictal[n]) # path to a folder with preictal models goes here before the '+'
        W_preictal.append(dict_models_preictal["W_model_preictal"])
        H_preictal.append(dict_models_preictal["H_model_preictal"])

    W_preictal_measurements = np.mean(np.asarray(W_preictal), axis=1)
    H_preictal_measurements = np.mean(np.asarray(H_preictal), axis=1)

    for n in range(len(files_interictal)):
        dict_models_interictal = sio.loadmat('*'+files_interictal[n]) # path to a folder with interictal models goes here before the '+'
        H_interictal.append(dict_models_interictal["H_model_interictal"])
        W_interictal.append(dict_models_interictal["W_model_interictal"])

    W_interictal_measurements = np.mean(np.asarray(W_interictal), axis=1)
    H_interictal_measurements = np.mean(np.asarray(H_interictal), axis=1)

    W_preictal_avg[idx, :] = np.mean(W_preictal_measurements,axis=0)
    H_preictal_avg[idx, :] = np.mean(H_preictal_measurements,axis=0)

    W_interictal_avg[idx, :] = np.mean(W_interictal_measurements,axis=0)
    H_interictal_avg[idx, :] = np.mean(H_interictal_measurements,axis=0)

# plotting the models
fig1 = plt.figure(figsize=(16,8))
gr = gridspec.GridSpec(nrows=2, ncols=5)
fig1.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.95, wspace=0.3, hspace=0.23)

ax1 = fig1.add_subplot(gr[0,0])
ax1.plot(W_preictal_avg[0,:],'--', linewidth=2, color='b', alpha=0.8)
ax1.plot(W_interictal_avg[0,:],'+', linewidth=2, color='b', alpha=0.5)
ax1.set_ylabel('Time coefficients',fontsize=20)
ax1.set_ylim([16, 32])
ax1.set_yticks([18, 22, 26, 30])
ax1.set_yticklabels([18, 22, 26, 30],fontsize=16)
ax1.set_xticks([0,6,12,18,24,28])
ax1.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax1.tick_params(axis="y", labelleft=True, labelsize=16)
ax1.set_xlabel('Time (min)',fontsize=20)
ax1.set_title("Patient 1", fontsize=20)
ax1.tick_params(length=6)

ax2 = fig1.add_subplot(gr[0,1], sharey=ax1)
ax2.plot(W_preictal_avg[1,:],'--', linewidth=2, color='b', alpha=0.8)
ax2.plot(W_interictal_avg[1,:],'+', linewidth=2, color='b', alpha=0.5)
ax2.set_xticks([0,6,12,18,24,28])
ax2.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax2.tick_params(axis="y", labelleft=False, labelsize=16)
ax2.set_xlabel('Time (min)',fontsize=20)
ax2.set_title("Patient 2", fontsize=20)
ax2.tick_params(length=6)

ax3 = fig1.add_subplot(gr[0,2], sharey=ax1)
ax3.plot(W_preictal_avg[2,:],'--', linewidth=2, color='b', alpha=0.8)
ax3.plot(W_interictal_avg[2,:],'+', linewidth=2, color='b', alpha=0.5)
ax3.set_xticks([0,6,12,18,24,28])
ax3.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax3.tick_params(axis="y", labelleft=False, labelsize=16)
ax3.set_xlabel('Time (min)',fontsize=20)
ax3.set_title("Patient 3", fontsize=20)
ax3.tick_params(length=6)

ax4 = fig1.add_subplot(gr[0,3], sharey=ax1)
ax4.plot(W_preictal_avg[3,:],'--', linewidth=2, color='b', alpha=0.8)
ax4.plot(W_interictal_avg[3,:],'+', linewidth=2, color='b', alpha=0.5)
ax4.set_xticks([0,6,12,18,24,28])
ax4.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax4.tick_params(axis="y", labelleft=False, labelsize=16)
ax4.set_xlabel('Time (min)',fontsize=20)
ax4.set_title("Patient 4", fontsize=20)
ax4.tick_params(length=6)

ax5 = fig1.add_subplot(gr[0,4])
ax5.plot(W_preictal_avg[4,:],'--', linewidth=2, color='b', alpha=0.8)
ax5.plot(W_interictal_avg[4,:],'+', linewidth=2, color='b', alpha=0.5)
ax5.set_ylim([0, 220])
ax5.set_yticks([50,100,150, 200])
ax5.set_yticklabels([50,100,150, 200],fontsize=16)
ax5.set_xticks([0,6,12,18,24,28])
ax5.set_xticklabels([0,1,2,3,4,5],fontsize=16)
ax5.tick_params(axis="y", labelleft=True, labelsize=16)
ax5.set_xlabel('Time (min)',fontsize=20)
ax5.set_title("Patient 5", fontsize=20)
ax5.tick_params(length=6)
ax5.legend(["Preictal", "Interictal"], fontsize=17, loc="center right")

ax6 = fig1.add_subplot(gr[1,0])
ax6.plot(H_preictal_avg[0,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax6.plot(H_interictal_avg[0,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax6.set_ylabel('Frequency coefficients',fontsize=20)
ax6.set_xlabel('Frequency (Hz)',fontsize=20)
ax6.set_ylim([0.033, 0.061])
ax6.set_yticks([0.04, 0.05, 0.06])
ax6.set_yticklabels([0.04, 0.05, 0.06],fontsize=16)
ax6.set_xticks([0,25,50])
ax6.set_xticklabels([0,64,128],fontsize=16)
ax6.tick_params(length=6)

ax7 = fig1.add_subplot(gr[1,1])
ax7.plot(H_preictal_avg[1,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax7.plot(H_interictal_avg[1,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax7.set_xlabel('Frequency (Hz)',fontsize=20)
ax7.set_ylim([0.01, 0.065])
ax7.set_yticks([0.02, 0.04, 0.06])
ax7.set_yticklabels([0.02, 0.04, 0.06],fontsize=16)
ax7.set_xticks([0,25,50])
ax7.set_xticklabels([0,128, 256],fontsize=16)
ax7.tick_params(length=6)

ax8 = fig1.add_subplot(gr[1,2])
ax8.plot(H_preictal_avg[2,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax8.plot(H_interictal_avg[2,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax8.set_xlabel('Frequency (Hz)',fontsize=20)
ax8.set_ylim([0.02, 0.1])
ax8.set_yticks([0.03, 0.06, 0.09])
ax8.set_yticklabels([0.03, 0.06, 0.09],fontsize=16)
ax8.set_xticks([0,25,50])
ax8.set_xticklabels([0,256,512],fontsize=16)
ax8.tick_params(length=6)

ax9 = fig1.add_subplot(gr[1,3])
ax9.plot(H_preictal_avg[3,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax9.plot(H_interictal_avg[3,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax9.set_xlabel('Frequency (Hz)',fontsize=20)
ax9.set_ylim([-0.38, 0.13])
ax9.set_yticks([-0.3, -0.2, -0.1, 0, 0.1])
ax9.set_yticklabels([-0.3, -0.2, -0.1, 0, 0.1],fontsize=16)
ax9.set_xticks([0,25,50])
ax9.set_xticklabels([0,256,512],fontsize=16)
ax9.tick_params(length=6)

ax10 = fig1.add_subplot(gr[1,4])
ax10.plot(H_preictal_avg[4,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax10.plot(H_interictal_avg[4,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax10.set_xlabel('Frequency (Hz)',fontsize=20)
ax10.tick_params(axis="y", labelleft=True, labelsize=16)
ax10.tick_params(length=6)
ax10.set_ylim([0.01, 0.065])
ax10.set_yticks([0.02, 0.04, 0.06])
ax10.set_yticklabels([0.02, 0.04, 0.06],fontsize=16)
ax10.set_xticks([0,25,50])
ax10.set_xticklabels([0,64,128],fontsize=16)
ax10.legend(["Preictal", "Interictal"], fontsize=17, loc="upper right")

fig1.text(0.043,0.56,r"$\textbf{A}$",fontsize=24)
fig1.text(0.235,0.56,r"$\textbf{B}$",fontsize=24)
fig1.text(0.43,0.56,r"$\textbf{C}$",fontsize=24)
fig1.text(0.62,0.56,r"$\textbf{D}$",fontsize=24)
fig1.text(0.815,0.56,r"$\textbf{E}$",fontsize=24)
fig1.text(0.043,0.08,r"$\textbf{G}$",fontsize=24)
fig1.text(0.235,0.08,r"$\textbf{H}$",fontsize=24)
fig1.text(0.435,0.08,r"$\textbf{I}$",fontsize=24)
fig1.text(0.627,0.08,r"$\textbf{J}$",fontsize=24)
fig1.text(0.815,0.08,r"$\textbf{K}$",fontsize=24)

fig1.savefig("../figures/average_components.pdf", pad_inches=0.4)
