
import os
import scipy.io as sio
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec, transforms
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

"""
This script plots average time and frequency components of preictal and interictal states of the Epilepyecosystem dataset.
"""

path = "*"         # path goes here
patients = []       # patient ids go here as strings in a list

W_preictal_avg = np.zeros((len(patients), 401))
H_preictal_avg = np.zeros((len(patients), 1071))
W_interictal_avg = np.zeros((len(patients), 401))
H_interictal_avg = np.zeros((len(patients), 1071))

for idx, patient_id in enumerate(patients):

    files_preictal = os.listdir("*") # path to a folder with preictal models goes here
    files_interictal = os.listdir("*") # path to a folder with interictal models goes here

    W_preictal = []
    H_preictal = []
    W_interictal = []
    H_interictal = []

    for n in range(len(files_preictal)):

        with open("*"+files_preictal[n], "rb") as f: # path to a folder with preictal models goes here before the +
            dict_models_preictal = pkl.load(f)

        W_preictal.append(np.squeeze(dict_models_preictal["W_predict"]))
        H_preictal.append(np.squeeze(dict_models_preictal["H_predict"]))

    W_preictal_measurements = np.mean(np.asarray(W_preictal), axis=1)
    H_preictal_measurements = np.mean(np.asarray(H_preictal), axis=1)

    for n in range(len(files_interictal)):

        with open("*"+files_interictal[n], "rb") as f: # path to a folder with interictal models goes here before the +
            dict_models_interictal = pkl.load(f)

        H_interictal.append(np.squeeze(dict_models_interictal["H_predict"]))
        W_interictal.append(np.squeeze(dict_models_interictal["W_predict"]))

    W_interictal_measurements = np.mean(np.asarray(W_interictal), axis=1)
    H_interictal_measurements = np.mean(np.asarray(H_interictal), axis=1)

    W_preictal_avg[idx, :] = np.mean(W_preictal_measurements,axis=0)
    H_preictal_avg[idx, :] = np.mean(H_preictal_measurements,axis=0)

    W_interictal_avg[idx, :] = np.mean(W_interictal_measurements,axis=0)
    H_interictal_avg[idx, :] = np.mean(H_interictal_measurements,axis=0)

# plotting the models
fig1 = plt.figure(figsize=(16,8))
gr = gridspec.GridSpec(nrows=2, ncols=3)
fig1.subplots_adjust(left=0.06, bottom=0.09, right=0.98, top=0.95, wspace=0.25, hspace=0.23)

ax1 = fig1.add_subplot(gr[0,0])
ax1.plot(W_preictal_avg[0,0::10],'--', linewidth=2, color='b', alpha=0.8)
ax1.plot(W_interictal_avg[0,0::10],'+', linewidth=2, color='b', alpha=0.5)
ax1.set_ylabel('Time coefficients',fontsize=20)
ax1.set_ylim([0.2, 1.8])
ax1.set_yticks([0.4, 0.8, 1.2, 1.6])
ax1.set_yticklabels([0.4, 0.8, 1.2, 1.6],fontsize=16)
ax1.set_xticks([0,8,16,24,32,40])
ax1.set_xticklabels([0,2,4,6,8,10],fontsize=16)
ax1.tick_params(axis="y", labelleft=True, labelsize=16)
ax1.set_xlabel('Time (min)',fontsize=20)
ax1.set_title("Patient 1", fontsize=20)
ax1.tick_params(length=6)

ax2 = fig1.add_subplot(gr[0,1], sharey=ax1)
ax2.plot(W_preictal_avg[1,0::10],'--', linewidth=2, color='b', alpha=0.8)
ax2.plot(W_interictal_avg[1,0::10],'+', linewidth=2, color='b', alpha=0.5)
ax2.set_xticks([0,8,16,24,32,40])
ax2.set_xticklabels([0,2,4,6,8,10],fontsize=16)
ax2.set_ylim([0.2, 1.8])
ax2.set_yticks([0.4, 0.8, 1.2, 1.6])
ax2.tick_params(axis="y", labelleft=False, labelsize=16)
ax2.set_xlabel('Time (min)',fontsize=20)
ax2.set_title("Patient 2", fontsize=20)
ax2.tick_params(length=6)

ax3 = fig1.add_subplot(gr[0,2], sharey=ax1)
ax3.plot(W_preictal_avg[2,0::10],'--', linewidth=2, color='b', alpha=0.8)
ax3.plot(W_interictal_avg[2,0::10],'+', linewidth=2, color='b', alpha=0.5)
ax3.set_xticks([0,8,16,24,32,40])
ax3.set_xticklabels([0,2,4,6,8,10],fontsize=16)
ax3.set_ylim([0.2, 1.8])
ax3.set_yticks([0.4, 0.8, 1.2, 1.6])
ax3.tick_params(axis="y", labelleft=False, labelsize=16)
ax3.set_xlabel('Time (min)',fontsize=20)
ax3.set_title("Patient 3", fontsize=20)
ax3.tick_params(length=6)
ax3.legend(["Preictal", "Interictal"], fontsize=17, loc="lower center")

ax4 = fig1.add_subplot(gr[1,0])
ax4.plot(H_preictal_avg[0,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax4.plot(H_interictal_avg[0,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax4.set_ylabel('Frequency coefficients',fontsize=20)
ax4.set_xlabel('Frequency (Hz)',fontsize=20)
ax4.set_ylim([0.35, 0.62])
ax4.set_yticks([0.4, 0.5, 0.6])
ax4.set_yticklabels([0.4, 0.5, 0.6],fontsize=16)
ax4.set_xticks([0,54,108])
ax4.set_xticklabels([0,100,200],fontsize=16)
ax4.tick_params(length=6)

ax5 = fig1.add_subplot(gr[1,1])
ax5.plot(H_preictal_avg[1,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax5.plot(H_interictal_avg[1,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax5.set_xlabel('Frequency (Hz)',fontsize=20)
ax5.set_ylim([0.51, 0.64])
ax5.set_yticks([0.55, 0.6])
ax5.set_yticklabels([0.55, 0.6, 0.65],fontsize=16)
ax5.set_xticks([0,54,108])
ax5.set_xticklabels([0,100,200],fontsize=16)
ax5.tick_params(length=6)

ax6 = fig1.add_subplot(gr[1,2])
ax6.plot(H_preictal_avg[2,0::10],'--', linewidth=2, color='r', alpha=0.8)
ax6.plot(H_interictal_avg[2,0::10],'+', linewidth=2, color='r', alpha=0.5)
ax6.set_xlabel('Frequency (Hz)',fontsize=20)
ax6.set_ylim([0.51, 0.64])
ax6.set_yticks([0.55, 0.6])
ax6.set_xticks([0,54,108])
ax6.set_xticklabels([0,100,200],fontsize=16)
ax6.tick_params(length=6, labelleft=False)
ax6.legend(["Preictal", "Interictal"], fontsize=17, loc="upper center")

fig1.text(0.043,0.56,r"$\textbf{A}$",fontsize=24)
fig1.text(0.37,0.56,r"$\textbf{B}$",fontsize=24)
fig1.text(0.7,0.56,r"$\textbf{C}$",fontsize=24)
fig1.text(0.043,0.09,r"$\textbf{D}$",fontsize=24)
fig1.text(0.37,0.09,r"$\textbf{E}$",fontsize=24)
fig1.text(0.7,0.09,r"$\textbf{F}$",fontsize=24)

fig1.savefig("../figures/average_components_ecosystem.pdf", pad_inches=0.4)
