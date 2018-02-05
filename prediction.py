# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:43:22 2017

@author: ostojanovic
"""
import os, pickle
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
ident = '109602'
patient_id = '109602'
base_directory = 'patient_'+ident+'_extracted_seizures/'
run_numbers = list(map(str,range(101)))[1:]     # the number of runs (100); where every numbers is a string; which is needed for saving

accuracies_rdf = []
accuracies_svm = []

for run_nr in run_numbers:

    ### out-of-sample set - data preparation ###
    files_out = []
    for root_out, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/random/'+run_nr+'/out-of-sample',topdown=True):
        files_out.append(files)

    corr_dict_AV_baseline_baseline_out = sio.loadmat(os.path.join(root_out,files_out[0][0]))
    corr_dict_AV_baseline_preictal_out = sio.loadmat(os.path.join(root_out,files_out[0][1]))
    corr_dict_AV_preictal_baseline_out = sio.loadmat(os.path.join(root_out,files_out[0][2]))
    corr_dict_AV_preictal_preictal_out = sio.loadmat(os.path.join(root_out,files_out[0][3]))

    corr_matrix_AV_baseline_baseline_out = corr_dict_AV_baseline_baseline_out["corr_matrix_AV_baseline_baseline"]
    corr_matrix_AV_baseline_preictal_out = corr_dict_AV_baseline_preictal_out["corr_matrix_AV_baseline_preictal"]
    corr_matrix_AV_preictal_baseline_out = corr_dict_AV_preictal_baseline_out["corr_matrix_AV_preictal_baseline"]
    corr_matrix_AV_preictal_preictal_out = corr_dict_AV_preictal_preictal_out["corr_matrix_AV_preictal_preictal"]

    class_vec_baseline_out = corr_dict_AV_baseline_baseline_out["class_labels_baseline"]
    class_vec_preictal_out = corr_dict_AV_baseline_preictal_out["class_labels_preictal"]

    data_baseline_out = np.concatenate((corr_matrix_AV_baseline_baseline_out,corr_matrix_AV_preictal_baseline_out),axis=1)
    data_preictal_out = np.concatenate((corr_matrix_AV_baseline_preictal_out,corr_matrix_AV_preictal_preictal_out),axis=1)

    data_out = np.concatenate((data_baseline_out,data_preictal_out))
    class_vec_out = np.concatenate((class_vec_baseline_out,class_vec_preictal_out))

    s_out = np.arange(class_vec_out.shape[0])
    np.random.shuffle(s_out)
    class_vec_out = class_vec_out[s_out]
    data_out = data_out[s_out]

    # rdf
    # reading rdf data
    with open(path+'/classification_and_prediction/models/random/'+patient_id+'/forests/'+'forest_'+patient_id+'_'+run_nr+'.pickle', 'rb') as f:
        forest = pickle.load(f)

    predict_rdf_out = forest.predict(data_out)
    resulting_probabilities_rdf = forest.predict_proba(data_out)

    out_rdf_err = 1-(predict_rdf_out == class_vec_out.T[0]).mean()
    acc_rdf = (1-out_rdf_err)*100
    accuracies_rdf.append(acc_rdf)

    # svm_
    # reading svm data
    with open(path+'/classification_and_prediction/models/random/'+patient_id+'/svm_models/'+'svm_'+patient_id+'_'+run_nr+'.pickle', 'rb') as s:
        svm_model = pickle.load(s)

    predict_svm_out = svm_model.predict(data_out)
    resulting_probabilities_svm = svm_model.predict_proba(data_out)

    out_svm_err = 1-(predict_svm_out == class_vec_out.T[0]).mean()
    acc_svm = (1-out_svm_err)*100
    accuracies_svm.append(acc_svm)

    with open(path+'/classification_and_prediction/results/'+patient_id+'/results_preidiction_'+patient_id+'_'+run_nr+'.txt', 'w') as text_file:
        text_file.write("patient id: "+str(patient_id)+"\n")
        text_file.write("the number of run: "+str(run_nr)+"\n")
        text_file.write("accuracy (rdf): "+str(acc_rdf)+"\n")
        text_file.write("accuracy (svm): "+str(acc_svm)+"\n")
        text_file.write("out-of-sample classes:   "+str(class_vec_out.T[0])+"\n")
        text_file.write("predicted classes (rdf): "+str(predict_rdf_out)+"\n")
        text_file.write("predicted classes (svm): "+str(predict_svm_out)+"\n")
        text_file.write("predicted probabilities (rdf): "+str(resulting_probabilities_rdf)+"\n")
        text_file.write("predicted probabilities (svm): "+str(resulting_probabilities_svm)+"\n")
        text_file.write("error on an out-of-sample set (rdf):  "+str(out_rdf_err)+"\n")
        text_file.write("error on an out-of-sample set (svm):  "+str(out_svm_err))

with open(path+'/classification_and_prediction/results/'+patient_id+'/all_results_prediction_'+patient_id+'.txt', 'w') as result_file:
    result_file.write("patient id: "+str(patient_id)+"\n")
    result_file.write("accuracies (rdf): "+str(accuracies_rdf)+"\n")
    result_file.write("accuracies (svm): "+str(accuracies_svm)+"\n")
    result_file.write("average accuracy (rdf): "+str(sum(accuracies_rdf)/len(accuracies_rdf))+"\n")
    result_file.write("average accuracy (svm): "+str(sum(accuracies_svm)/len(accuracies_svm)))
