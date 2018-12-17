from __future__ import unicode_literals
import json, pickle
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots 2 figures:
    * accuracy, specificity and sensitivity for balanced and imbalaned set, and for both classifiers (RDF and SVM)
    * accuracy for balanced set (used for a presentation)."""

path = "/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/"


####################################################### loading and extracting onformation ################################################################################################################

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

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/11502/all_results_11502.txt") as f7:
    imbalanced_11502 = json.load(f7)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/25302_2/all_results_25302_2.txt") as f8:
    imbalanced_25302 = json.load(f8)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/59002_2/all_results_59002_2.txt") as f9:
    imbalanced_59002 = json.load(f9)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/62002_2/all_results_62002_2.txt") as f10:
    imbalanced_62002 = json.load(f10)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/97002_3/all_results_97002_3.txt") as f11:
    imbalanced_97002 = json.load(f11)

with open("/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/60-40-merged/109602/all_results_109602.txt") as f12:
    imbalanced_109602 = json.load(f12)

variables = pickle.load(open(path+"variables_additional_calculations.pickle","rb"))

x = np.array([0.0, 1.15, 2.65, 3.8, 5.3, 6.45, 7.95, 9.1, 10.6, 11.75, 13.25, 14.45])
patients = ['1-balanced','1-imbalanced','2-balanced','2-imbalanced','3-balanced','3-imbalanced','4-balanced','4-imbalanced','5-balanced','5-imbalanced','6-balanced','6-imbalanced']
width = 0.17       # the width of bars

acc_rdf1_b = np.round(balanced_11502[0]["accuracy_rdf"],decimals=2)
acc_svm1_b = np.round(balanced_11502[0]["accuracy_svm"],decimals=2)
acc_rdf2_b = np.round(balanced_25302[0]["accuracy_rdf"],decimals=2)
acc_svm2_b = np.round(balanced_25302[0]["accuracy_svm"],decimals=2)
acc_rdf3_b = np.round(balanced_59002[0]["accuracy_rdf"],decimals=2)
acc_svm3_b = np.round(balanced_59002[0]["accuracy_svm"],decimals=2)
acc_rdf4_b = np.round(balanced_62002[0]["accuracy_rdf"],decimals=2)
acc_svm4_b = np.round(balanced_62002[0]["accuracy_svm"],decimals=2)
acc_rdf5_b = np.round(balanced_97002[0]["accuracy_rdf"],decimals=2)
acc_svm5_b = np.round(balanced_97002[0]["accuracy_svm"],decimals=2)
acc_rdf6_b = np.round(balanced_109602[0]["accuracy_rdf"],decimals=2)
acc_svm6_b = np.round(balanced_109602[0]["accuracy_svm"],decimals=2)

acc_rdf1_imb = np.round(imbalanced_11502[0]["accuracy_rdf"],decimals=2)
acc_svm1_imb = np.round(imbalanced_11502[0]["accuracy_svm"],decimals=2)
acc_rdf2_imb = np.round(imbalanced_25302[0]["accuracy_rdf"],decimals=2)
acc_svm2_imb = np.round(imbalanced_25302[0]["accuracy_svm"],decimals=2)
acc_rdf3_imb = np.round(imbalanced_59002[0]["accuracy_rdf"],decimals=2)
acc_svm3_imb = np.round(imbalanced_59002[0]["accuracy_svm"],decimals=2)
acc_rdf4_imb = np.round(imbalanced_62002[0]["accuracy_rdf"],decimals=2)
acc_svm4_imb = np.round(imbalanced_62002[0]["accuracy_svm"],decimals=2)
acc_rdf5_imb = np.round(imbalanced_97002[0]["accuracy_rdf"],decimals=2)
acc_svm5_imb = np.round(imbalanced_97002[0]["accuracy_svm"],decimals=2)
acc_rdf6_imb = np.round(imbalanced_109602[0]["accuracy_rdf"],decimals=2)
acc_svm6_imb = np.round(imbalanced_109602[0]["accuracy_svm"],decimals=2)

specificity_rdf_b = variables["specificities_rdf_balanced"]
specificity_svm_b = variables["specificities_svm_balanced"]
specificity_rdf_imb = variables["specificities_rdf_imbalanced"]
specificity_svm_imb = variables["specificities_svm_imbalanced"]

spec1_rdf_b = np.round(specificity_rdf_b[0],decimals=2)
spec2_rdf_b = np.round(specificity_rdf_b[1],decimals=2)
spec3_rdf_b = np.round(specificity_rdf_b[2],decimals=2)
spec4_rdf_b = np.round(specificity_rdf_b[3],decimals=2)
spec5_rdf_b = np.round(specificity_rdf_b[4],decimals=2)
spec6_rdf_b = np.round(specificity_rdf_b[5],decimals=2)

spec1_svm_b = np.round(specificity_svm_b[0],decimals=2)
spec2_svm_b = np.round(specificity_svm_b[1],decimals=2)
spec3_svm_b = np.round(specificity_svm_b[2],decimals=2)
spec4_svm_b = np.round(specificity_svm_b[3],decimals=2)
spec5_svm_b = np.round(specificity_svm_b[4],decimals=2)
spec6_svm_b = np.round(specificity_svm_b[5],decimals=2)

spec1_rdf_imb = np.round(specificity_rdf_imb[0],decimals=2)
spec2_rdf_imb = np.round(specificity_rdf_imb[1],decimals=2)
spec3_rdf_imb = np.round(specificity_rdf_imb[2],decimals=2)
spec4_rdf_imb = np.round(specificity_rdf_imb[3],decimals=2)
spec5_rdf_imb = np.round(specificity_rdf_imb[4],decimals=2)
spec6_rdf_imb = np.round(specificity_rdf_imb[5],decimals=2)

spec1_svm_imb = np.round(specificity_svm_imb[0],decimals=2)
spec2_svm_imb = np.round(specificity_svm_imb[1],decimals=2)
spec3_svm_imb = np.round(specificity_svm_imb[2],decimals=2)
spec4_svm_imb = np.round(specificity_svm_imb[3],decimals=2)
spec5_svm_imb = np.round(specificity_svm_imb[4],decimals=2)
spec6_svm_imb = np.round(specificity_svm_imb[5],decimals=2)

sens_rdf1_b = np.round(balanced_11502[0]["recall_rdf"],decimals=2)
sens_svm1_b = np.round(balanced_11502[0]["recall_svm"],decimals=2)
sens_rdf2_b = np.round(balanced_25302[0]["recall_rdf"],decimals=2)
sens_svm2_b = np.round(balanced_25302[0]["recall_svm"],decimals=2)
sens_rdf3_b = np.round(balanced_59002[0]["recall_rdf"],decimals=2)
sens_svm3_b = np.round(balanced_59002[0]["recall_svm"],decimals=2)
sens_rdf4_b = np.round(balanced_62002[0]["recall_rdf"],decimals=2)
sens_svm4_b = np.round(balanced_62002[0]["recall_svm"],decimals=2)
sens_rdf5_b = np.round(balanced_97002[0]["recall_rdf"],decimals=2)
sens_svm5_b = np.round(balanced_97002[0]["recall_svm"],decimals=2)
sens_rdf6_b = np.round(balanced_109602[0]["recall_rdf"],decimals=2)
sens_svm6_b = np.round(balanced_109602[0]["recall_svm"],decimals=2)

sens_rdf1_imb = np.round(imbalanced_11502[0]["recall_rdf"],decimals=2)
sens_svm1_imb = np.round(imbalanced_11502[0]["recall_svm"],decimals=2)
sens_rdf2_imb = np.round(imbalanced_25302[0]["recall_rdf"],decimals=2)
sens_svm2_imb = np.round(imbalanced_25302[0]["recall_svm"],decimals=2)
sens_rdf3_imb = np.round(imbalanced_59002[0]["recall_rdf"],decimals=2)
sens_svm3_imb = np.round(imbalanced_59002[0]["recall_svm"],decimals=2)
sens_rdf4_imb = np.round(imbalanced_62002[0]["recall_rdf"],decimals=2)
sens_svm4_imb = np.round(imbalanced_62002[0]["recall_svm"],decimals=2)
sens_rdf5_imb = np.round(imbalanced_97002[0]["recall_rdf"],decimals=2)
sens_svm5_imb = np.round(imbalanced_97002[0]["recall_svm"],decimals=2)
sens_rdf6_imb = np.round(imbalanced_109602[0]["recall_rdf"],decimals=2)
sens_svm6_imb = np.round(imbalanced_109602[0]["recall_svm"],decimals=2)

#################################################################### extraction finished ##############################################################################################################

# making arrays with a structure: [50-50, 60-40, 50-50, etc]
rdf_accuracy = np.array([acc_rdf1_b,acc_rdf1_imb,acc_rdf2_b,acc_rdf2_imb,acc_rdf3_b,acc_rdf3_imb,acc_rdf4_b,acc_rdf4_imb,acc_rdf5_b,acc_rdf5_imb,acc_rdf6_b,acc_rdf6_imb])
svm_accuracy = np.array([acc_svm1_b,acc_svm1_imb,acc_svm2_b,acc_svm2_imb,acc_svm3_b,acc_svm3_imb,acc_svm4_b,acc_svm4_imb,acc_svm5_b,acc_svm5_imb,acc_svm6_b,acc_svm6_imb])
thresholds   = np.array([50.0,  58.82, 50.0, 60.0, 50.0,  57.14, 50.0, 60.0, 50.0,  57.14, 50.0,  60.0])

rdf_specificity = np.array([spec1_rdf_b,spec1_rdf_imb,spec2_rdf_b,spec2_rdf_imb,spec3_rdf_b,spec3_rdf_imb,spec4_rdf_b,spec4_rdf_imb,spec5_rdf_b,spec5_rdf_imb,spec6_rdf_b,spec6_rdf_imb])
svm_specificity = np.array([spec1_svm_b,spec1_svm_imb,spec2_svm_b,spec2_svm_imb,spec3_svm_b,spec3_svm_imb,spec4_svm_b,spec4_svm_imb,spec5_svm_b,spec5_svm_imb,spec6_svm_b,spec6_svm_imb])

rdf_sensitivity = np.array([sens_rdf1_b,sens_rdf1_imb,sens_rdf2_b,sens_rdf2_imb,sens_rdf3_b,sens_rdf3_imb,sens_rdf4_b,sens_rdf4_imb,sens_rdf5_b,sens_rdf5_imb,sens_rdf6_b,sens_rdf6_imb])
svm_sensitivity = np.array([sens_svm1_b,sens_svm1_imb,sens_svm2_b,sens_svm2_imb,sens_svm3_b,sens_svm3_imb,sens_svm4_b,sens_svm4_imb,sens_svm5_b,sens_svm5_imb,sens_svm6_b,sens_svm6_imb])

# plotting accuracy, sensitivity and specidificty for both sets and classifiers
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(19.5,10.2))
rdf1 = ax[0].bar(x, rdf_accuracy, width, color='firebrick')
svm1 = ax[0].bar(x+width, svm_accuracy, width, color='royalblue')
thres1 = ax[0].bar(x+width*2, thresholds, width, color='green')
ax[0].set_yticks([0,20,40,60,80,100])
ax[0].set_yticklabels([0,20,40,60,80,100],fontsize=16)
ax[0].set_ylabel('Accuracy (in percentages)',fontsize=18)
ax[0].set_title('Accuracy',fontsize=18)
ax[0].legend((rdf1, svm1, thres1), ('RDF','SVM','Threshold'),fontsize=14)
fig.text(0.025,0.55,r"$\textbf{A}$",fontsize=18)

rdf2 = ax[1].bar(x, rdf_specificity, width, color='firebrick')
svm2 = ax[1].bar(x+width, svm_specificity, width, color='royalblue')
rdf3 = ax[1].bar(x+0.34, rdf_sensitivity, width, color='darkorchid')
svm3 = ax[1].bar(x+0.34+width, svm_sensitivity, width, color='peru')
ax[1].set_xticks(x+width/0.7)
ax[1].set_xticklabels(patients,fontsize=16)
ax[1].set_xlabel('Patient\'s number and set',fontsize=18)
ax[1].set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax[1].set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
ax[1].set_ylabel('Specificity and sensitivity',fontsize=18)
ax[1].set_title('Specificity and sensitivity',fontsize=18)
ax[1].legend((rdf2, svm2, rdf3, svm3), ('Specificity (RDF)','Specificity (SVM)','Sensitivity (RDF)','Sensitivity (SVM)'),loc=3,fontsize=14)
fig.text(0.025,0.08,r"$\textbf{B}$",fontsize=18)

fig.tight_layout()
fig.show()

# for the presentation; 50-50 set only
x_new = np.array([0.0, 2.65, 5.3, 7.95, 10.6, 13.25])
patients = ['1','2','3','4','5','6']
width = 0.4

rdf_accuracy1 = np.array([acc_rdf1_b,acc_rdf2_b,acc_rdf3_b,acc_rdf4_b,acc_rdf5_b,acc_rdf6_b])
svm_accuracy1 = np.array([acc_svm1_b,acc_svm2_b,acc_svm3_b,acc_svm4_b,acc_svm5_b,acc_svm6_b])

fig2 = plt.figure(figsize=(19.5,10.2))
rdf2 = plt.bar(x_new, rdf_accuracy1, width, color='firebrick')
svm2 = plt.bar(x_new+width, svm_accuracy1, width, color='royalblue')
plt.yticks([0,20,40,60,80,100],fontsize=26)
plt.ylabel('Accuracy (in percentages)',fontsize=26)
plt.title('Accuracy',fontsize=26)
plt.legend(('RDF','SVM'),loc='upper right',fontsize=22)

plt.xticks(x_new+width/2,patients,fontsize=26)
plt.xlabel('Patient\'s number',fontsize=26)

fig2.tight_layout()
fig2.show()
