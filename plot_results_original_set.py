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
This script plots accuracy, specificity and sensitivity for the original set, for both classifiers (RDF and SVM).
"""

path = "/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/"

with open(path+"original-merged/11502/all_results_11502.txt") as f1:
    balanced_11502 = json.load(f1)

with open(path+"original-merged/25302_2/all_results_25302_2.txt") as f2:
    balanced_25302 = json.load(f2)

with open(path+"original-merged/59002_2/all_results_59002_2.txt") as f3:
    balanced_59002 = json.load(f3)

with open(path+"original-merged/62002_2/all_results_62002_2.txt") as f4:
    balanced_62002 = json.load(f4)

with open(path+"original-merged/97002_3/all_results_97002_3.txt") as f5:
    balanced_97002 = json.load(f5)

with open(path+"original-merged/109602/all_results_109602.txt") as f6:
    balanced_109602 = json.load(f6)

variables = pickle.load(open(path+"variables_additional_calculations_original_set.pickle","rb"))

x = np.array([0.0, 2.65, 5.3, 7.95, 10.6, 13.25])
patients = ['1','2','3','4','5','6']
width = 0.17       # width of bars

acc_rdf1 = np.round(balanced_11502[0]["accuracy_rdf"],decimals=2)
acc_rdf2 = np.round(balanced_25302[0]["accuracy_rdf"],decimals=2)
acc_rdf3 = np.round(balanced_59002[0]["accuracy_rdf"],decimals=2)
acc_rdf4 = np.round(balanced_62002[0]["accuracy_rdf"],decimals=2)
acc_rdf5 = np.round(balanced_97002[0]["accuracy_rdf"],decimals=2)
acc_rdf6 = np.round(balanced_109602[0]["accuracy_rdf"],decimals=2)

acc_svm1 = np.round(balanced_11502[0]["accuracy_svm"],decimals=2)
acc_svm2 = np.round(balanced_25302[0]["accuracy_svm"],decimals=2)
acc_svm3 = np.round(balanced_59002[0]["accuracy_svm"],decimals=2)
acc_svm4 = np.round(balanced_62002[0]["accuracy_svm"],decimals=2)
acc_svm5 = np.round(balanced_97002[0]["accuracy_svm"],decimals=2)
acc_svm6 = np.round(balanced_109602[0]["accuracy_svm"],decimals=2)

specificity_rdf = variables["specificities_rdf_original"]
specificity_svm = variables["specificities_svm_original"]

spec1_rdf = np.round(specificity_rdf[0],decimals=2)
spec2_rdf = np.round(specificity_rdf[1],decimals=2)
spec3_rdf = np.round(specificity_rdf[2],decimals=2)
spec4_rdf = np.round(specificity_rdf[3],decimals=2)
spec5_rdf = np.round(specificity_rdf[4],decimals=2)
spec6_rdf = np.round(specificity_rdf[5],decimals=2)

spec1_svm = np.round(specificity_svm[0],decimals=2)
spec2_svm = np.round(specificity_svm[1],decimals=2)
spec3_svm = np.round(specificity_svm[2],decimals=2)
spec4_svm = np.round(specificity_svm[3],decimals=2)
spec5_svm = np.round(specificity_svm[4],decimals=2)
spec6_svm = np.round(specificity_svm[5],decimals=2)

sens_rdf1 = np.round(balanced_11502[0]["recall_rdf"],decimals=2)
sens_rdf2 = np.round(balanced_25302[0]["recall_rdf"],decimals=2)
sens_rdf3 = np.round(balanced_59002[0]["recall_rdf"],decimals=2)
sens_rdf4 = np.round(balanced_62002[0]["recall_rdf"],decimals=2)
sens_rdf5 = np.round(balanced_97002[0]["recall_rdf"],decimals=2)
sens_rdf6 = np.round(balanced_109602[0]["recall_rdf"],decimals=2)

sens_svm1 = np.round(balanced_11502[0]["recall_svm"],decimals=2)
sens_svm2 = np.round(balanced_25302[0]["recall_svm"],decimals=2)
sens_svm3 = np.round(balanced_59002[0]["recall_svm"],decimals=2)
sens_svm4 = np.round(balanced_62002[0]["recall_svm"],decimals=2)
sens_svm5 = np.round(balanced_97002[0]["recall_svm"],decimals=2)
sens_svm6 = np.round(balanced_109602[0]["recall_svm"],decimals=2)

# [50-50, 60-40, 50-50, etc]
rdf_accuracy = np.array([acc_rdf1,acc_rdf2,acc_rdf3,acc_rdf4,acc_rdf5,acc_rdf6])
svm_accuracy = np.array([acc_svm1,acc_svm2,acc_svm3,acc_svm4,acc_svm5,acc_svm6])
thresholds   = np.array([79.4,87.5,88.89,94.29,62.5,66.67])

rdf_specificity = np.array([spec1_rdf,spec2_rdf,spec3_rdf,spec4_rdf,spec5_rdf,spec6_rdf])
svm_specificity = np.array([spec1_svm,spec2_svm,spec3_svm,spec4_svm,spec5_svm,spec6_svm])

rdf_sensitivity = np.array([sens_rdf1,sens_rdf2,sens_rdf3,sens_rdf4,sens_rdf5,sens_rdf6])
svm_sensitivity = np.array([sens_svm1,sens_svm2,sens_svm3,sens_svm4,sens_svm5,sens_svm6])
fig,ax = plt.subplots(2, 1, sharex=True, figsize=(19.5,10.2))

rdf1 = ax[0].bar(x, rdf_accuracy, width, color='firebrick')
svm1 = ax[0].bar(x+width, svm_accuracy, width, color='royalblue')
thres1 = ax[0].bar(x+width*2, thresholds, width, color='green')
ax[0].set_yticks([0,20,40,60,80,100])
ax[0].set_yticklabels([0,20,40,60,80,100],fontsize=16)
ax[0].set_ylabel('Accuracy (in percentages)',fontsize=18)
ax[0].set_title('Accuracy on the original set',fontsize=18)
ax[0].legend((rdf1, svm1, thres1), ('RDF','SVM','Threshold'),fontsize=14)

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
ax[1].set_title('Specificity and sensitivity on the original set',fontsize=18)
ax[1].legend((rdf2, svm2, rdf3, svm3), ('Specificity (RDF)','Specificity (SVM)','Sensitivity (RDF)','Sensitivity (SVM)'),loc=3,fontsize=14)

fig.tight_layout()
fig.show()
