
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

patients = []       # patient ids go here

accuracy_svm = []
accuracy_smote = []

specificity_svm = []
specificity_smote = []

sensitivity_svm = []
sensitivity_smote = []

precision_svm = []
precision_smote = []

ppv_svm = []
ppv_smote = []

npv_svm = []
npv_smote = []

####################################################### loading and extracting information #####################################################################

for idx, patient_id in enumerate(patients):
    with open(''.txt") as f:            # add path with text file with all results
        file = json.load(f)

    accuracy_smote.append(np.round(file[0]["accuracy_smote"],decimals=3))
    specificity_smote.append(np.round(file[0]["specificity_smote"],decimals=3))
    sensitivity_smote.append(np.round(file[0]["sensitivity_smote"],decimals=3))
    ppv_smote.append(np.round(file[0]["positive_predictive_smote"],decimals=3))
    npv_smote.append(np.round(file[0]["negative_predictive_smote"],decimals=3))

#################################################################### extraction finished ######################################################################

fig = plt.figure(figsize=(12,4))
gs = gridspec.GridSpec(1, 3, left=0.07, right = 0.94, wspace=0.15, hspace=0.15, top=0.84, bottom = 0.2)

ax1 = plt.subplot(gs[0])

for point, patient, color in zip([n*100 for n in accuracy_smote], np.linspace(1,6,6), cm.Dark2(np.linspace(0, 1, len(accuracy_smote)))):
    plt.hlines(patient, 0, 101, color="grey")
    ax1.plot(point, patient, marker="o", markersize=12, color=color, alpha=0.5)

ax1.set_xlim([70, 101])
ax1.set_xticks([70, 80, 90, 100])
ax1.set_xticklabels([70, 80, 90, 100])
ax1.set_ylim([0,7])
ax1.set_yticks([1, 2, 3, 4, 5, 6])
ax1.set_yticklabels([1, 2, 3, 4, 5, 6])
ax1.tick_params(axis='both', which='both', top=False, labelsize=18, length=8)

ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax1.set_xlabel('Accuracy', fontsize=22)
ax1.set_ylabel("Patient's number", fontsize=22, rotation="vertical")


ax2 = plt.subplot(gs[1])

handles = []
for sensitivity, specificity, color in zip(sensitivity_smote, specificity_smote, cm.Dark2(np.linspace(0, 1, len(accuracy_smote)))):
    handle = plt.plot(sensitivity, specificity, "o", markersize=12, color=color, alpha=0.5)
    handles.append(handle[0])

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
plt.tick_params(axis='both',labelsize=18, length=8)

plt.xlabel('Sensitivity',fontsize=22)
plt.ylabel('Specificity',fontsize=22)
fig.axes[1].yaxis.set_label_position("right")
ax2.set_aspect(1)

ax3 = plt.subplot(gs[2])

for ppv, npv, color in zip(ppv_smote, npv_smote, cm.Dark2(np.linspace(0, 1, len(accuracy_smote)))):
    plt.plot(ppv, npv, "o", markersize=12, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
plt.tick_params(axis='both',labelsize=18, labelleft=False, length=8)

plt.xlabel('Positive predictive value',fontsize=22)
plt.ylabel('Negative \n predictive value',fontsize=22)
fig.axes[2].yaxis.set_label_position("right")
ax3.set_aspect(1)

fig.text(0.05,0.85,r"$\textbf{A}$",fontsize=26)
fig.text(0.4,0.85,r"$\textbf{B}$",fontsize=26)
fig.text(0.7,0.85,r"$\textbf{C}$",fontsize=26)

plt.show()

##############################################################################################################################

# fig.savefig("figures/measures.pdf", pad_inches=0.4)
