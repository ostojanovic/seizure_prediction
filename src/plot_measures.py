
import json, pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

patients_epilepsiae = []   # patient ids go here as strings in a list
patients_ecosystem = []   # patient ids go here as strings in a list

accuracy_epilepsiae = []
accuracy_ecosystem = []
accuracy_benchmark = []

specificity_epilepsiae = []
specificity_ecosystem = []
specificity_benchmark = []

sensitivity_epilepsiae = []
sensitivity_ecosystem = []
sensitivity_benchmark = []

precision_epilepsiae = []
precision_ecosystem = []
precision_benchmark = []

ppv_epilepsiae = []
ppv_ecosystem = []
ppv_benchmark = []

npv_epilepsiae = []
npv_ecosystem = []
npv_benchmark = []

####################################################### loading and extracting information #####################################################################

for idx, patient_id in enumerate(patients_epilepsiae):
    with open('*.txt') as f:            # add path with text file with all results for the epilepsiae dataset
        file = json.load(f)

    accuracy_epilepsiae.append(np.round(file[0]["accuracy_smote"],decimals=3))
    specificity_epilepsiae.append(np.round(file[0]["specificity_smote"],decimals=3))
    sensitivity_epilepsiae.append(np.round(file[0]["sensitivity_smote"],decimals=3))
    ppv_epilepsiae.append(np.round(file[0]["positive_predictive_smote"],decimals=3))
    npv_epilepsiae.append(np.round(file[0]["negative_predictive_smote"],decimals=3))

for idx, patient_id in enumerate(patients_ecosystem):
    with open('*.txt') as f:            # add path with text file with all results for the epilepsyecosystem dataset
        file = json.load(f)

    accuracy_ecosystem.append(np.round(file[0]["accuracy_svmsmote"],decimals=3))
    specificity_ecosystem.append(np.round(file[0]["specificity_svmsmote"],decimals=3))
    sensitivity_ecosystem.append(np.round(file[0]["sensitivity_svmsmote"],decimals=3))
    ppv_ecosystem.append(np.round(file[0]["positive_predictive_svmsmote"],decimals=3))
    npv_ecosystem.append(np.round(file[0]["negative_predictive_svmsmote"],decimals=3))

for idx, patient_id in enumerate(patients_ecosystem):
    with open('*.txt') as f:            # add path with text file with all results for the benchmark dataset
        file = json.load(f)

    accuracy_benchmark.append(np.round(file["accuracy"],decimals=3))
    specificity_benchmark.append(np.round(file["specificity"],decimals=3))
    sensitivity_benchmark.append(np.round(file["sensitivity"],decimals=3))
    ppv_benchmark.append(np.round(file["positive_predictive_value"],decimals=3))
    npv_benchmark.append(np.round(file["negative_predictive_value"],decimals=3))

#################################################################### extraction finished ######################################################################

fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3, 3, left=0.07, right = 0.95, wspace=0.05, hspace=0.22, top=0.92, bottom = 0.09)

ax1 = plt.subplot(gs[0])

for point, patient, color in zip([n*100 for n in accuracy_epilepsiae], np.linspace(1,6,6), cm.Dark2(np.linspace(0, 1, len(accuracy_epilepsiae)))):
    plt.hlines(patient, 0, 101, color="grey")
    ax1.plot(point, patient, marker="o", markersize=12, color=color, alpha=0.5)

ax1.set_xlim([50, 101])
ax1.set_xticks([50,60,70, 80, 90, 100])
ax1.set_ylim([0,7])
ax1.set_yticks([1, 2, 3, 4, 5])
ax1.set_yticklabels([1, 2, 3, 4, 5])
ax1.tick_params(axis='both', which='both', labelbottom=False, top=False, labelsize=18, length=8)

ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_ylabel("Patient's number", fontsize=22, rotation="vertical")


ax2 = plt.subplot(gs[1])

for sensitivity, specificity, color in zip(sensitivity_epilepsiae, specificity_epilepsiae, cm.Dark2(np.linspace(0, 1, len(accuracy_epilepsiae)))):
    plt.plot(sensitivity, specificity, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax2.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both', labelbottom=False, labelsize=18, length=8)

plt.ylabel('Specificity',fontsize=22)
fig.axes[1].yaxis.set_label_position("right")
ax2.set_aspect(1)

ax3 = plt.subplot(gs[2])

for ppv, npv, color in zip(ppv_epilepsiae, npv_epilepsiae, cm.Dark2(np.linspace(0, 1, len(accuracy_epilepsiae)))):
    plt.plot(ppv, npv, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax3.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both',labelsize=18, labelbottom=False, labelleft=False, length=8)

plt.ylabel('Negative \n predictive value',fontsize=22)
fig.axes[2].yaxis.set_label_position("right")
ax3.set_aspect(1)

ax4 = plt.subplot(gs[3])

for point, patient, color in zip([n*100 for n in accuracy_ecosystem], np.linspace(1,3,3), cm.Dark2(np.linspace(0, 1, len(accuracy_ecosystem)))):
    plt.hlines(patient, 0, 101, color="grey")
    ax4.plot(point, patient, marker="o", markersize=12, color=color, alpha=0.5)

ax4.set_xlim([60,101])
ax4.set_xticks([60,70, 80, 90, 100])
ax4.set_ylim([0,4])
ax4.set_yticks([1, 2, 3])
ax4.set_yticklabels([1, 2, 3])
ax4.tick_params(axis='both', which='both', top=False, labelsize=18, length=8, labelbottom=False)

ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax4.set_ylabel("Patient's number", fontsize=22, rotation="vertical")


ax5 = plt.subplot(gs[4])

for sensitivity, specificity, color in zip(sensitivity_ecosystem, specificity_ecosystem, cm.Dark2(np.linspace(0, 1, len(accuracy_ecosystem)))):
    plt.plot(sensitivity, specificity, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax5.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both',labelsize=18, length=8, labelbottom=False)

plt.ylabel('Specificity',fontsize=22)
fig.axes[4].yaxis.set_label_position("right")
ax5.set_aspect(1)

ax6 = plt.subplot(gs[5])

for ppv, npv, color in zip(ppv_ecosystem, npv_ecosystem, cm.Dark2(np.linspace(0, 1, len(accuracy_ecosystem)))):
    plt.plot(ppv, npv, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax6.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both',labelsize=18, labelleft=False, length=8, labelbottom=False)

plt.ylabel('Negative \n predictive value',fontsize=22)
fig.axes[5].yaxis.set_label_position("right")
ax6.set_aspect(1)

ax7 = plt.subplot(gs[6])

for point, patient, color in zip([n*100 for n in accuracy_benchmark], np.linspace(1,3,3), cm.Dark2(np.linspace(0, 1, len(accuracy_benchmark)))):
    plt.hlines(patient, 0, 101, color="grey")
    ax7.plot(point, patient, marker="o", markersize=12, color=color, alpha=0.5)

ax7.set_xlim([60,101])
ax7.set_xticks([60,70, 80, 90, 100])
ax7.set_xticklabels([60,70, 80, 90, 100])
ax7.set_ylim([0,4])
ax7.set_yticks([1, 2, 3])
ax7.set_yticklabels([1, 2, 3])
ax7.tick_params(axis='both', which='both', top=False, labelsize=18, length=8)

ax7.spines['right'].set_visible(False)
ax7.spines['left'].set_visible(False)
ax7.spines['top'].set_visible(False)

ax7.set_xlabel('Accuracy', fontsize=22)
ax7.set_ylabel("Patient's number", fontsize=22, rotation="vertical")


ax8 = plt.subplot(gs[7])

for sensitivity, specificity, color in zip(sensitivity_benchmark, specificity_benchmark, cm.Dark2(np.linspace(0, 1, len(accuracy_benchmark)))):
    plt.plot(sensitivity, specificity, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax8.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both',labelsize=18, length=8)

plt.xlabel('Sensitivity',fontsize=22)
plt.ylabel('Specificity',fontsize=22)
fig.axes[7].yaxis.set_label_position("right")
ax8.set_aspect(1)

ax9 = plt.subplot(gs[8])

for ppv, npv, color in zip(ppv_benchmark, npv_benchmark, cm.Dark2(np.linspace(0, 1, len(accuracy_benchmark)))):
    plt.plot(ppv, npv, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
ax9.set_xticks([0, 0.5, 1])
plt.tick_params(axis='both',labelsize=18, labelleft=False, length=8)

plt.xlabel('Positive predictive value',fontsize=22)
plt.ylabel('Negative \n predictive value',fontsize=22)
fig.axes[8].yaxis.set_label_position("right")
ax9.set_aspect(1)


fig.text(0.05,0.925,r"$\textbf{A}$",fontsize=26)
fig.text(0.43,0.925,r"$\textbf{B}$",fontsize=26)
fig.text(0.73,0.925,r"$\textbf{C}$",fontsize=26)
fig.text(0.05,0.63,r"$\textbf{D}$",fontsize=26)
fig.text(0.43,0.63,r"$\textbf{E}$",fontsize=26)
fig.text(0.73,0.63,r"$\textbf{F}$",fontsize=26)
fig.text(0.05,0.335,r"$\textbf{G}$",fontsize=26)
fig.text(0.43,0.335,r"$\textbf{H}$",fontsize=26)
fig.text(0.73,0.335,r"$\textbf{I}$",fontsize=26)

##############################################################################################################################

fig.savefig("../figures/measures.pdf", pad_inches=0.4)
