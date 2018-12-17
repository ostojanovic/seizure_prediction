
from __future__ import unicode_literals
import pickle, matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots positive and negative predictive values for balanced and imbalanced sets.
"""

path = "/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results/"

patients = ['1','2','3','4','5','6']

variables = pickle.load(open(path+"variables_additional_calculations.pickle","rb"))

positive_predictive_value_svm_balanced = variables["positive_predictive_value_svm_balanced"]
positive_predictive_value_rdf_balanced = variables["positive_predictive_value_rdf_balanced"]
positive_predictive_value_svm_imbalanced = variables["positive_predictive_value_svm_imbalanced"]
positive_predictive_value_rdf_imbalanced = variables["positive_predictive_value_rdf_imbalanced"]

negative_predictive_value_svm_balanced = variables["negative_predictive_value_svm_balanced"]
negative_predictive_value_rdf_balanced = variables["negative_predictive_value_rdf_balanced"]
negative_predictive_value_svm_imbalanced = variables["negative_predictive_value_svm_imbalanced"]
negative_predictive_value_rdf_imbalanced = variables["negative_predictive_value_rdf_imbalanced"]

# plotting the positive and the negative predictive values
fig,ax = plt.subplots(2, 2, sharex=True, figsize=(19.5,10.2))
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
width = 0.17       # the width of bars

ppv_rdf_balanced = ax[0,0].bar(x, positive_predictive_value_rdf_balanced, width, color='firebrick')
ppv_svm_balanced = ax[0,0].bar(x+width, positive_predictive_value_svm_balanced, width, color='royalblue')
ax[0,0].set_yticks(y)
ax[0,0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=16)
ax[0,0].set_ylabel('Positive predictive value',fontsize=18)
ax[0,0].set_title('Positive predictive value for balanced set',fontsize=18)
ax[0,0].legend((ppv_rdf_balanced, ppv_svm_balanced), ('RDF','SVM'),fontsize=14)
fig.text(0.08,0.55,r"$\textbf{A}$",fontsize=18)

npv_rdf_balanced = ax[0,1].bar(x, negative_predictive_value_rdf_balanced, width, color='firebrick')
npv_svm_balanced = ax[0,1].bar(x+width, negative_predictive_value_svm_balanced, width, color='royalblue')
ax[0,1].set_yticks(y)
ax[0,1].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=16)
ax[0,1].set_ylabel('Negative predictive value',fontsize=18)
ax[0,1].set_title('Negative predictive value for balanced set',fontsize=18)
ax[0,1].legend((npv_rdf_balanced, npv_svm_balanced), ('RDF','SVM'),fontsize=14)
fig.text(0.51,0.55,r"$\textbf{B}$",fontsize=18)

ppv_rdf_imbalanced = ax[1,0].bar(x, positive_predictive_value_rdf_imbalanced, width, color='firebrick')
ppv_svm_imbalanced = ax[1,0].bar(x+width, positive_predictive_value_svm_imbalanced, width, color='royalblue')
ax[1,0].set_xticks(x+width/2)
ax[1,0].set_xticklabels(patients,fontsize=16)
ax[1,0].set_yticks(y)
ax[1,0].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=16)
ax[1,0].set_xlabel('Patient\'s number',fontsize=18)
ax[1,0].set_ylabel('Positive predictive value',fontsize=18)
ax[1,0].set_title('Positive predictive value for imbalanced set',fontsize=18)
ax[1,0].legend((ppv_rdf_imbalanced, ppv_svm_imbalanced), ('RDF','SVM'),fontsize=14)
fig.text(0.08,0.1,r"$\textbf{C}$",fontsize=18)

npv_rdf_imbalanced = ax[1,1].bar(x, negative_predictive_value_rdf_imbalanced, width, color='firebrick')
npv_svm_imbalanced = ax[1,1].bar(x+width, negative_predictive_value_svm_imbalanced, width, color='royalblue')
ax[1,1].set_xticks(x+width/2)
ax[1,1].set_xticklabels(patients,fontsize=16)
ax[1,1].set_yticks(y)
ax[1,1].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=16)
ax[1,1].set_xlabel('Patient\'s number',fontsize=18)
ax[1,1].set_ylabel('Negative predictive value',fontsize=18)
ax[1,1].set_title('Negative predictive value for imbalanced set',fontsize=18)
ax[1,1].legend((npv_rdf_imbalanced, npv_svm_imbalanced), ('RDF','SVM'),fontsize=14)
fig.text(0.51,0.1,r"$\textbf{D}$",fontsize=18)

fig.subplots_adjust(left=0.12,top=0.9,right=0.9,bottom=0.10,wspace=0.20,hspace=0.25)
fig.show()
