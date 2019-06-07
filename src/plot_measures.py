
import json, pickle
import numpy as np
import seaborn as sns
from collections import OrderedDict
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

patients = ["11502", "25302", "59002", "62002", "97002", "109602"]

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
    with open('/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_'+patient_id+'_extracted_seizures/results/all_results_'+patient_id+".txt") as f:
        file = json.load(f)

    accuracy_smote.append(np.round(file[0]["accuracy_smote"],decimals=3))
    specificity_smote.append(np.round(file[0]["specificity_smote"],decimals=3))
    sensitivity_smote.append(np.round(file[0]["sensitivity_smote"],decimals=3))
    ppv_smote.append(np.round(file[0]["positive_predictive_smote"],decimals=3))
    npv_smote.append(np.round(file[0]["negative_predictive_smote"],decimals=3))

#################################################################### extraction finished ######################################################################

fig1 = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(1, 1, left=0.14, right = 0.94, wspace=0.23, hspace=0.14, top=0.9, bottom = 0.25)

ax1 = plt.subplot(gs[0])

for point, patient in zip([n*100 for n in accuracy_smote], np.linspace(1,6,6)):
    plt.hlines(patient, 0, 101, color="grey")
    ax1.plot(point, patient, marker="o", markersize=12, color="cornflowerblue")

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

#####################################################################################################################

fig2 = plt.figure(figsize=(10,5))
fig2.subplots(sharey=True)
plt.subplots_adjust(left=0.1, right = 0.94, wspace=0.1, hspace=0.48, top=0.85, bottom = 0.16)

plt.subplot(121, adjustable='box', aspect=1)

handles = []
for sensitivity, specificity, color in zip(sensitivity_smote, specificity_smote, cm.Dark2(np.linspace(0, 1, len(accuracy_smote)))):
    handle = plt.plot(sensitivity, specificity, "o", markersize=18, color=color, alpha=0.5)
    handles.append(handle[0])

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
plt.tick_params(axis='both',labelsize=18, length=8)

plt.legend(handles, ["1", "2", "3", "4", "5", "6"], fontsize=14, ncol=2, loc="lower left")

plt.xlabel('Sensitivity',fontsize=26)
plt.ylabel('Specificity',fontsize=26)

plt.subplot(122, adjustable='box', aspect=1)

for ppv, npv, color in zip(ppv_smote, npv_smote, cm.Dark2(np.linspace(0, 1, len(accuracy_smote)))):
    plt.plot(ppv, npv, "o", markersize=18, color=color, alpha=0.5)

plt.plot([0,1], [1,0], color="black", alpha=0.8)
plt.fill_between([0,1],[1,0], facecolor="none", hatch="X", edgecolor="black", linewidth=0.0, alpha=0.5)

plt.xlim([-0.01, 1.05])
plt.ylim([-0.01, 1.05])
plt.tick_params(axis='both',labelsize=18, labelleft=False, length=8)

plt.xlabel('Positive predictive value',fontsize=22)
plt.ylabel('Negative predictive value',fontsize=22)
fig2.axes[1].yaxis.set_label_position("right")

fig2.text(0.05,0.16,r"$\textbf{A}$",fontsize=26)
fig2.text(0.53,0.16,r"$\textbf{B}$",fontsize=26)

plt.show()

##############################################################################################################################

# fig1.savefig("figures/accuracy.pdf", pad_inches=0.4)
# fig2.savefig("figures/measures.pdf", pad_inches=0.4)
