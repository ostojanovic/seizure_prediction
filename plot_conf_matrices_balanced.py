
from __future__ import unicode_literals
import json
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import gridspec
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots confusion matrices for balanced set (RDF and SVM).
"""

################################################################# loading and extraction of information ##################################################################################

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/11502/all_results_11502.txt") as f1:
    balanced_11502 = json.load(f1)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/25302_2/all_results_25302_2.txt") as f2:
    balanced_25302 = json.load(f2)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/59002_2/all_results_59002_2.txt") as f3:
    balanced_59002 = json.load(f3)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/62002_2/all_results_62002_2.txt") as f4:
    balanced_62002 = json.load(f4)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/97002_3/all_results_97002_3.txt") as f5:
    balanced_97002 = json.load(f5)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/50-50-merged/109602/all_results_109602.txt") as f6:
    balanced_109602 = json.load(f6)

# 50-50 merged set
# confusion matrix svm: 11502
tp_svm_11502 = np.round(balanced_11502[1]["avg_true_positives_svm"],decimals=2)
fp_svm_11502 = np.round(balanced_11502[1]["avg_false_positives_svm"],decimals=2)
fn_svm_11502 = np.round(balanced_11502[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_11502 = np.round(balanced_11502[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_11502 = [[tp_svm_11502,fp_svm_11502],[fn_svm_11502,tn_svm_11502]]

# confusion matrix rdf: 11502
tp_rdf_11502 = np.round(balanced_11502[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_11502 = np.round(balanced_11502[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_11502 = np.round(balanced_11502[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_11502 = np.round(balanced_11502[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_11502 = [[tp_rdf_11502,fp_rdf_11502],[fn_rdf_11502,tn_rdf_11502]]

# confusion matrix svm: 25302
tp_svm_25302 = np.round(balanced_25302[1]["avg_true_positives_svm"],decimals=2)
fp_svm_25302 = np.round(balanced_25302[1]["avg_false_positives_svm"],decimals=2)
fn_svm_25302 = np.round(balanced_25302[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_25302 = np.round(balanced_25302[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_25302 = [[tp_svm_25302,fp_svm_25302],[fn_svm_25302,tn_svm_25302]]

# confusion matrix rdf: 25302
tp_rdf_25302 = np.round(balanced_25302[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_25302 = np.round(balanced_25302[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_25302 = np.round(balanced_25302[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_25302 = np.round(balanced_25302[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_25302 = [[tp_rdf_25302,fp_rdf_25302],[fn_rdf_25302,tn_rdf_25302]]

# confusion matrix svm: 59002
tp_svm_59002 = np.round(balanced_59002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_59002 = np.round(balanced_59002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_59002 = np.round(balanced_59002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_59002 = np.round(balanced_59002[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_59002 = [[tp_svm_59002,fp_svm_59002],[fn_svm_59002,tn_svm_59002]]

# confusion matrix rdf: 59002
tp_rdf_59002 = np.round(balanced_59002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_59002 = np.round(balanced_59002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_59002 = np.round(balanced_59002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_59002 = np.round(balanced_59002[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_59002 = [[tp_rdf_59002,fp_rdf_59002],[fn_rdf_59002,tn_rdf_59002]]

# confusion matrix svm: 62002
tp_svm_62002 = np.round(balanced_62002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_62002 = np.round(balanced_62002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_62002 = np.round(balanced_62002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_62002 = np.round(balanced_62002[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_62002 = [[tp_svm_62002,fp_svm_62002],[fn_svm_62002,tn_svm_62002]]

# confusion matrix rdf: 62002
tp_rdf_62002 = np.round(balanced_62002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_62002 = np.round(balanced_62002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_62002 = np.round(balanced_62002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_62002 = np.round(balanced_62002[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_62002 = [[tp_rdf_62002,fp_rdf_62002],[fn_rdf_62002,tn_rdf_62002]]

# confusion matrix svm: 97002
tp_svm_97002 = np.round(balanced_97002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_97002 = np.round(balanced_97002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_97002 = np.round(balanced_97002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_97002 = np.round(balanced_97002[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_97002 = [[tp_svm_97002,fp_svm_97002],[fn_svm_97002,tn_svm_97002]]

# confusion matrix rdf: 97002
tp_rdf_97002 = np.round(balanced_97002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_97002 = np.round(balanced_97002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_97002 = np.round(balanced_97002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_97002 = np.round(balanced_97002[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_97002 = [[tp_rdf_97002,fp_rdf_97002],[fn_rdf_97002,tn_rdf_97002]]

# confusion matrix svm: 109602
tp_svm_109602 = np.round(balanced_109602[1]["avg_true_positives_svm"],decimals=2)
fp_svm_109602 = np.round(balanced_109602[1]["avg_false_positives_svm"],decimals=2)
fn_svm_109602 = np.round(balanced_109602[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_109602 = np.round(balanced_109602[1]["avg_true_negatives_svm"],decimals=2)
confusion_matrix_svm_109602 = [[tp_svm_109602,fp_svm_109602],[fn_svm_109602,tn_svm_109602]]

# confusion matrix rdf: 109602
tp_rdf_109602 = np.round(balanced_109602[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_109602 = np.round(balanced_109602[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_109602 = np.round(balanced_109602[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_109602 = np.round(balanced_109602[1]["avg_true_negatives_rdf"],decimals=2)
confusion_matrix_rdf_109602 = [[tp_rdf_109602,fp_rdf_109602],[fn_rdf_109602,tn_rdf_109602]]

################################################################# loading, extraction and organization finished ############################################################################

# plotting all of the confusion matrices
fig = plt.figure(figsize=(19.5,10.2))
gr = gridspec.GridSpec(nrows=2, ncols=6, wspace=0.15, hspace=0.05, width_ratios=[1,1,1,1,1,1], height_ratios=[1,1])

# defining axes
ax7 = fig.add_subplot(gr[1, 0], adjustable='box-forced', aspect="equal")
ax8 = fig.add_subplot(gr[1, 1], sharey=ax7,adjustable='box-forced',aspect="equal")
ax9 = fig.add_subplot(gr[1, 2], sharey=ax7,adjustable='box-forced',aspect="equal")
ax10 = fig.add_subplot(gr[1, 3], sharey=ax7,adjustable='box-forced',aspect="equal")
ax11 = fig.add_subplot(gr[1, 4], sharey=ax7,adjustable='box-forced',aspect="equal")
ax12 = fig.add_subplot(gr[1, 5], sharey=ax7,adjustable='box-forced',aspect="equal")

ax1 = fig.add_subplot(gr[0, 0], sharex=ax7, adjustable='box-forced', aspect="equal")
ax2 = fig.add_subplot(gr[0, 1], sharey=ax1, sharex=ax8,adjustable='box-forced',aspect="equal")
ax3 = fig.add_subplot(gr[0, 2], sharey=ax1, sharex=ax9,adjustable='box-forced',aspect="equal")
ax4 = fig.add_subplot(gr[0, 3], sharey=ax1, sharex=ax10,adjustable='box-forced',aspect="equal")
ax5 = fig.add_subplot(gr[0, 4], sharey=ax1, sharex=ax11,adjustable='box-forced',aspect="equal")
ax6 = fig.add_subplot(gr[0, 5], sharey=ax1, sharex=ax12,adjustable='box-forced',aspect="equal")

for tick in ax1.get_xticklabels():
    tick.set_visible(False)
for tick in ax2.get_xticklabels():
    tick.set_visible(False)
for tick in ax2.get_yticklabels():
    tick.set_visible(False)
for tick in ax3.get_xticklabels():
    tick.set_visible(False)
for tick in ax3.get_yticklabels():
    tick.set_visible(False)
for tick in ax4.get_xticklabels():
    tick.set_visible(False)
for tick in ax4.get_yticklabels():
    tick.set_visible(False)
for tick in ax5.get_xticklabels():
    tick.set_visible(False)
for tick in ax5.get_yticklabels():
    tick.set_visible(False)
for tick in ax6.get_xticklabels():
    tick.set_visible(False)
for tick in ax6.get_yticklabels():
    tick.set_visible(False)

for tick in ax8.get_yticklabels():
    tick.set_visible(False)
for tick in ax9.get_yticklabels():
    tick.set_visible(False)
for tick in ax10.get_yticklabels():
    tick.set_visible(False)
for tick in ax11.get_yticklabels():
    tick.set_visible(False)
for tick in ax12.get_yticklabels():
    tick.set_visible(False)

# plotting matrices
# upper row
img1 = ax1.imshow(confusion_matrix_svm_11502, cmap='RdBu_r',vmin=0,vmax=1)
ax1.text(0,0,np.char.mod("%s",confusion_matrix_svm_11502[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax1.text(1,0,np.char.mod("%s",confusion_matrix_svm_11502[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax1.text(0,1,np.char.mod("%s",confusion_matrix_svm_11502[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax1.text(1,1,np.char.mod("%s",confusion_matrix_svm_11502[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax1.set_title("Patient 1 (SVM)",fontsize=18)
ax1.set_ylabel("True label",fontsize=18)
ax1.set_yticks([0,1])
ax1.set_yticklabels([1,0],fontsize=16)
fig.text(0.035,0.62,r"$\textbf{A}$",fontsize=18)

ax2.imshow(confusion_matrix_svm_25302, cmap='RdBu_r',vmin=0,vmax=1)
ax2.text(0,0,np.char.mod("%s",confusion_matrix_svm_25302[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax2.text(1,0,np.char.mod("%s",confusion_matrix_svm_25302[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax2.text(0,1,np.char.mod("%s",confusion_matrix_svm_25302[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax2.text(1,1,np.char.mod("%s",confusion_matrix_svm_25302[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax2.set_title("Patient 2 (SVM)",fontsize=18)
fig.text(0.185,0.62,r"$\textbf{B}$",fontsize=18)

ax3.imshow(confusion_matrix_svm_59002, cmap='RdBu_r',vmin=0,vmax=1)
ax3.text(0,0,np.char.mod("%s",confusion_matrix_svm_59002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax3.text(1,0,np.char.mod("%s",confusion_matrix_svm_59002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax3.text(0,1,np.char.mod("%s",confusion_matrix_svm_59002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax3.text(1,1,np.char.mod("%s",confusion_matrix_svm_59002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax3.set_title("Patient 3 (SVM)",fontsize=18)
fig.text(0.335,0.62,r"$\textbf{C}$",fontsize=18)

ax4.imshow(confusion_matrix_svm_62002, cmap='RdBu_r',vmin=0,vmax=1)
ax4.text(0,0,np.char.mod("%s",confusion_matrix_svm_62002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax4.text(1,0,np.char.mod("%s",confusion_matrix_svm_62002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax4.text(0,1,np.char.mod("%s",confusion_matrix_svm_62002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax4.text(1,1,np.char.mod("%s",confusion_matrix_svm_62002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax4.set_title("Patient 4 (SVM)",fontsize=18)
fig.text(0.48,0.62,r"$\textbf{D}$",fontsize=18)

ax5.imshow(confusion_matrix_svm_97002, cmap='RdBu_r',vmin=0,vmax=1)
ax5.text(0,0,np.char.mod("%s",confusion_matrix_svm_97002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax5.text(1,0,np.char.mod("%s",confusion_matrix_svm_97002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax5.text(0,1,np.char.mod("%s",confusion_matrix_svm_97002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax5.text(1,1,np.char.mod("%s",confusion_matrix_svm_97002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax5.set_title("Patient 5 (SVM)",fontsize=18)
fig.text(0.63,0.62,r"$\textbf{E}$",fontsize=18)

ax6.imshow(confusion_matrix_svm_109602, cmap='RdBu_r',vmin=0,vmax=1)
ax6.text(0,0,np.char.mod("%s",confusion_matrix_svm_109602[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax6.text(1,0,np.char.mod("%s",confusion_matrix_svm_109602[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax6.text(0,1,np.char.mod("%s",confusion_matrix_svm_109602[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax6.text(1,1,np.char.mod("%s",confusion_matrix_svm_109602[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax6.set_title("Patient 6 (SVM)",fontsize=18)
fig.text(0.78,0.62,r"$\textbf{F}$",fontsize=18)

# lower row
ax7.imshow(confusion_matrix_rdf_11502, cmap='RdBu_r',vmin=0,vmax=1)
ax7.text(0,0,np.char.mod("%s",confusion_matrix_rdf_11502[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax7.text(1,0,np.char.mod("%s",confusion_matrix_rdf_11502[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax7.text(0,1,np.char.mod("%s",confusion_matrix_rdf_11502[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax7.text(1,1,np.char.mod("%s",confusion_matrix_rdf_11502[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax7.set_title("Patient 1 (RDF)",fontsize=18)
ax7.set_xlabel("Predicted label",fontsize=18)
ax7.set_xticks([1,0])
ax7.set_xticklabels([0,1],fontsize=16)
ax7.set_ylabel("True label",fontsize=18)
ax7.set_yticks([0,1])
ax7.set_yticklabels([1,0],fontsize=16)
fig.text(0.035,0.185,r"$\textbf{G}$",fontsize=18)

ax8.imshow(confusion_matrix_rdf_25302, cmap='RdBu_r',vmin=0,vmax=1)
ax8.text(0,0,np.char.mod("%s",confusion_matrix_rdf_25302[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax8.text(1,0,np.char.mod("%s",confusion_matrix_rdf_25302[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax8.text(0,1,np.char.mod("%s",confusion_matrix_rdf_25302[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax8.text(1,1,np.char.mod("%s",confusion_matrix_rdf_25302[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax8.set_title("Patient 2 (RDF)",fontsize=18)
ax8.set_xlabel("Predicted label",fontsize=18)
ax8.set_xticks([1,0])
ax8.set_xticklabels([0,1],fontsize=16)
fig.text(0.185,0.185,r"$\textbf{H}$",fontsize=18)

ax9.imshow(confusion_matrix_rdf_59002, cmap='RdBu_r',vmin=0,vmax=1)
ax9.text(0,0,np.char.mod("%s",confusion_matrix_rdf_59002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax9.text(1,0,np.char.mod("%s",confusion_matrix_rdf_59002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax9.text(0,1,np.char.mod("%s",confusion_matrix_rdf_59002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax9.text(1,1,np.char.mod("%s",confusion_matrix_rdf_59002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax9.set_title("Patient 3 (RDF)",fontsize=18)
ax9.set_xlabel("Predicted label",fontsize=18)
ax9.set_xticks([1,0])
ax9.set_xticklabels([0,1],fontsize=16)
fig.text(0.34,0.185,r"$\textbf{I}$",fontsize=18)

ax10.imshow(confusion_matrix_rdf_62002, cmap='RdBu_r',vmin=0,vmax=1)
ax10.text(0,0,np.char.mod("%s",confusion_matrix_rdf_62002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax10.text(1,0,np.char.mod("%s",confusion_matrix_rdf_62002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax10.text(0,1,np.char.mod("%s",confusion_matrix_rdf_62002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax10.text(1,1,np.char.mod("%s",confusion_matrix_rdf_62002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax10.set_title("Patient 4 (RDF)",fontsize=18)
ax10.set_xlabel("Predicted label",fontsize=18)
ax10.set_xticks([1,0])
ax10.set_xticklabels([0,1],fontsize=16)
fig.text(0.485,0.185,r"$\textbf{J}$",fontsize=18)

ax11.imshow(confusion_matrix_rdf_97002, cmap='RdBu_r',vmin=0,vmax=1)
ax11.text(0,0,np.char.mod("%s",confusion_matrix_rdf_97002[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax11.text(1,0,np.char.mod("%s",confusion_matrix_rdf_97002[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax11.text(0,1,np.char.mod("%s",confusion_matrix_rdf_97002[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax11.text(1,1,np.char.mod("%s",confusion_matrix_rdf_97002[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax11.set_title("Patient 5 (RDF)",fontsize=18)
ax11.set_xlabel("Predicted label",fontsize=18)
ax11.set_xticks([1,0])
ax11.set_xticklabels([0,1],fontsize=16)
fig.text(0.63,0.185,r"$\textbf{K}$",fontsize=18)

ax12.imshow(confusion_matrix_rdf_109602, cmap='RdBu_r',vmin=0,vmax=1)
ax12.text(0,0,np.char.mod("%s",confusion_matrix_rdf_109602[0][0]),horizontalalignment="center",color="black",fontsize=16)
ax12.text(1,0,np.char.mod("%s",confusion_matrix_rdf_109602[0][1]),horizontalalignment="center",color="black",fontsize=16)
ax12.text(0,1,np.char.mod("%s",confusion_matrix_rdf_109602[1][0]),horizontalalignment="center",color="black",fontsize=16)
ax12.text(1,1,np.char.mod("%s",confusion_matrix_rdf_109602[1][1]),horizontalalignment="center",color="black",fontsize=16)
ax12.set_title("Patient 6 (RDF)",fontsize=14)
ax12.set_xlabel("Predicted label",fontsize=12)
ax12.set_xticks([1,0])
ax12.set_xticklabels([0,1],fontsize=16)
fig.text(0.78,0.185,r"$\textbf{L}$",fontsize=18)

fig.suptitle('Confusion matrices for all patients and both models', ha='center', fontsize=20)
fig.subplots_adjust(left=0.05, bottom=0.10, right=0.92, top=0.95, wspace=0.15, hspace=0.05)
cbaxes = fig.add_axes([0.94, 0.1, 0.01, 0.8])
cb = plt.colorbar(img1,ax=ax1, cax=cbaxes)
cb.ax.tick_params(labelsize=16)
fig.text(0.98, 0.5, 'p(assigned label $|$ true label)', va='center', rotation='vertical',fontsize=18)
fig.show()
