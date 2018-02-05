# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:43:22 2017

@author: ostojanovic
"""
import os, pickle, time
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
ident = '109602'
patient_id = '109602'
base_directory = 'patient_'+ident+'_extracted_seizures/'
run_numbers = list(map(str,range(101)))[1:]     # the number of runs (100); where every numbers is a string; which is needed for saving

accuracies_rdf = []
accuracies_svm = []

for run_nr in run_numbers:

    ### training set - data preparation ###
    files_train = []
    for root_train, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/random/'+run_nr+'/train',topdown=True):
        files_train.append(files)

    corr_dict_AV_baseline_baseline_train = sio.loadmat(os.path.join(root_train,files_train[0][0]))
    corr_dict_AV_baseline_preictal_train = sio.loadmat(os.path.join(root_train,files_train[0][1]))
    corr_dict_AV_preictal_baseline_train = sio.loadmat(os.path.join(root_train,files_train[0][2]))
    corr_dict_AV_preictal_preictal_train = sio.loadmat(os.path.join(root_train,files_train[0][3]))

    corr_matrix_AV_baseline_baseline_train = corr_dict_AV_baseline_baseline_train["corr_matrix_AV_baseline_baseline"]
    corr_matrix_AV_baseline_preictal_train = corr_dict_AV_baseline_preictal_train["corr_matrix_AV_baseline_preictal"]
    corr_matrix_AV_preictal_baseline_train = corr_dict_AV_preictal_baseline_train["corr_matrix_AV_preictal_baseline"]
    corr_matrix_AV_preictal_preictal_train = corr_dict_AV_preictal_preictal_train["corr_matrix_AV_preictal_preictal"]

    class_vec_baseline_train = corr_dict_AV_baseline_baseline_train["class_labels_baseline"]
    class_vec_preictal_train = corr_dict_AV_baseline_preictal_train["class_labels_preictal"]

    data_baseline_train = np.concatenate((corr_matrix_AV_baseline_baseline_train,corr_matrix_AV_preictal_baseline_train),axis=1)
    data_preictal_train = np.concatenate((corr_matrix_AV_baseline_preictal_train,corr_matrix_AV_preictal_preictal_train),axis=1)

    data_train = np.concatenate((data_baseline_train,data_preictal_train))
    class_vec_train = np.concatenate((class_vec_baseline_train,class_vec_preictal_train))

    s = np.arange(class_vec_train.shape[0])
    np.random.shuffle(s)
    class_vec_train = class_vec_train[s]
    data_train = data_train[s]

    ### testing set - data preparation ###
    files_test = []
    for root_test, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/random/'+run_nr+'/test',topdown=True):
        files_test.append(files)

    corr_dict_AV_baseline_baseline_test = sio.loadmat(os.path.join(root_test,files_test[0][0]))
    corr_dict_AV_baseline_preictal_test = sio.loadmat(os.path.join(root_test,files_test[0][1]))
    corr_dict_AV_preictal_baseline_test = sio.loadmat(os.path.join(root_test,files_test[0][2]))
    corr_dict_AV_preictal_preictal_test = sio.loadmat(os.path.join(root_test,files_test[0][3]))

    corr_matrix_AV_baseline_baseline_test = corr_dict_AV_baseline_baseline_test["corr_matrix_AV_baseline_baseline"]
    corr_matrix_AV_baseline_preictal_test = corr_dict_AV_baseline_preictal_test["corr_matrix_AV_baseline_preictal"]
    corr_matrix_AV_preictal_baseline_test = corr_dict_AV_preictal_baseline_test["corr_matrix_AV_preictal_baseline"]
    corr_matrix_AV_preictal_preictal_test = corr_dict_AV_preictal_preictal_test["corr_matrix_AV_preictal_preictal"]

    class_vec_baseline_test = corr_dict_AV_baseline_baseline_test["class_labels_baseline"]
    class_vec_preictal_test = corr_dict_AV_baseline_preictal_test["class_labels_preictal"]

    data_baseline_test = np.concatenate((corr_matrix_AV_baseline_baseline_test,corr_matrix_AV_preictal_baseline_test),axis=1)
    data_preictal_test = np.concatenate((corr_matrix_AV_baseline_preictal_test,corr_matrix_AV_preictal_preictal_test),axis=1)

    data_test = np.concatenate((data_baseline_test,data_preictal_test))
    class_vec_test = np.concatenate((class_vec_baseline_test,class_vec_preictal_test))

    s_test = np.arange(class_vec_test.shape[0])
    np.random.shuffle(s_test)
    class_vec_test = class_vec_test[s_test]
    data_test = data_test[s_test]

    # rdf
    # construct the set of hyperparameters to tune
    params_rdf = {"n_estimators": [1, 10, 50, 100, 200],
                  "max_depth": np.hstack([np.arange(1, 100, 5), None])}

    # rdf on training and testing set on the time component
    model_rdf = RandomForestClassifier(n_jobs=-1,criterion="entropy")
    grid_rdf = RandomizedSearchCV(model_rdf, params_rdf)
    start_rdf = time.time()

    # fitting a model and making predictions
    grid_rdf.fit(data_train, class_vec_train.ravel())
    print("grid search for rdf took {:.2f} seconds".format(time.time() - start_rdf))

    acc_rdf = grid_rdf.score(data_test,class_vec_test)
    accuracies_rdf.append(acc_rdf*100)

    print("grid_rdf search accuracy: {:.2f}%".format(acc_rdf * 100))
    print("grid_rdf search best parameters: {}".format(grid_rdf.best_params_))

    # store coeffed values in yy
    predict_rdf_train = grid_rdf.predict(data_train)
    predict_rdf_test = grid_rdf.predict(data_test)

    # evaluate the error on a training and a testing dataset and store it for each tree depth
    train_rdf_err = 1-(predict_rdf_train == class_vec_train.T[0]).mean()
    test_rdf_err = 1-(predict_rdf_test == class_vec_test.T[0]).mean()

    # svm
    # construct the set of hyperparameters to tune
    params_svm = {"kernel": ['linear','rbf'],
                  "class_weight": [None,'balanced']}

    # svm on training and testing set on the time component
    model_svm = svm.SVC(degree=3, gamma='auto', probability=True)
    grid_svm = GridSearchCV(model_svm, params_svm)
    start_svm = time.time()

    # fitting a model and making predictions
    grid_svm.fit(data_train, class_vec_train.ravel())
    print("grid search for svm took {:.2f} seconds".format(time.time() - start_svm))
    acc_svm = grid_svm.score(data_test,class_vec_test)

    accuracies_svm.append(acc_svm*100)

    print("grid_svm search accuracy: {:.2f}%".format(acc_svm * 100))
    print("grid_svm search best parameters: {}".format(grid_svm.best_params_))

    # store coeffed values in yy
    predict_svm_train = grid_svm.predict(data_train)
    predict_svm_test = grid_svm.predict(data_test)

    # evaluate the error on a training and a testing dataset and store it for each tree depth
    train_svm_err = 1-(predict_svm_train == class_vec_train.T[0]).mean()
    test_svm_err = 1-(predict_svm_test == class_vec_test.T[0]).mean()

    with open(path+'/classification_and_prediction/models/random/'+patient_id+'/forests/'+'forest_'+patient_id+'_'+run_nr+'.pickle', 'wb') as f:
        pickle.dump(grid_rdf, f)

    with open(path+'/classification_and_prediction/models/random/'+patient_id+'/svm_models/'+'svm_'+patient_id+'_'+run_nr+'.pickle', 'wb') as s:
        pickle.dump(grid_svm, s)

    with open(path+'/classification_and_prediction/results/'+patient_id+'/results_classification_'+patient_id+'_'+run_nr+'.txt', 'w') as text_file:
        text_file.write("patient id: "+str(patient_id)+"\n")
        text_file.write("number of the run: "+str(run_nr)+"\n")
        text_file.write("forest parameters: \n")
        text_file.write("   number of trees: "+str(grid_rdf.best_params_['n_estimators'])+ "\n")
        text_file.write("   maximal depth:   "+str(grid_rdf.best_params_['max_depth'])+ "\n")
        text_file.write("svm parameters: \n")
        text_file.write("   kernel: "+str(grid_svm.best_params_['kernel'])+ "\n")
        text_file.write("   class weight:   "+str(grid_svm.best_params_['class_weight'])+ "\n")
        text_file.write("rdf accuracy: "+str(acc_rdf*100)+"\n")
        text_file.write("svm accuracy: "+str(acc_svm*100)+"\n")
        text_file.write("test classes:            "+str(class_vec_test.T[0])+"\n")
        text_file.write("predicted classes (rdf): "+str(predict_rdf_test)+"\n")
        text_file.write("predicted classes (svm): "+str(predict_svm_test)+"\n")
        text_file.write("training error (rdf): "+str(train_rdf_err)+"\n")
        text_file.write("training error (svm): "+str(train_svm_err)+"\n")
        text_file.write("testing error (rdf):  "+str(test_rdf_err)+"\n")
        text_file.write("testing error (svm):  "+str(test_svm_err))

with open(path+'/classification_and_prediction/results/'+patient_id+'/all_results_classification_'+patient_id+'.txt', 'w') as result_file:
    result_file.write("patient id: "+str(patient_id)+"\n")
    result_file.write("accuracies (rdf): "+str(accuracies_rdf)+"\n")
    result_file.write("accuracies (svm): "+str(accuracies_svm)+"\n")
    result_file.write("average accuracy (rdf): "+str(sum(accuracies_rdf)/len(accuracies_rdf))+"\n")
    result_file.write("average accuracy (svm): "+str(sum(accuracies_svm)/len(accuracies_svm)))
