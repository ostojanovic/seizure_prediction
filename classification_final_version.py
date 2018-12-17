
import os, pickle, time, json
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
ident = '97002'
patient_id = '97002_3'
percent = 'original'
base_directory = 'patient_'+ident+'_extracted_seizures/97002102'
run_numbers = list(map(str,range(101)))[1:]     # the number of runs (100); where every numbers is a string; which is needed for saving

accuracies_rdf = []
accuracies_svm = []

precisions_rdf = []
precisions_svm = []

recalls_rdf = []
recalls_svm = []

fscores_rdf = []
fscores_svm = []

true_negatives_rdf = []
false_positives_rdf = []
false_negatives_rdf = []
true_positives_rdf = []

true_negatives_svm = []
false_positives_svm = []
false_negatives_svm = []
true_positives_svm = []

for run_nr in run_numbers:

    ### training set - data preparation ###
    files_train = []
    for root_train, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/'+percent+'/'+run_nr+'/train',topdown=True):
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

    ### evaluation set - data preparation ###
    files_test = []
    for root_test, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/'+percent+'/'+run_nr+'/test',topdown=True):
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

    files_out = []
    for root_out, dirs, files in os.walk(path+base_directory+'/corr_coeff_'+patient_id+'/'+percent+'/'+run_nr+'/out-of-sample',topdown=True):
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

    data_eval = np.concatenate((data_test,data_out))
    class_vec_eval = np.concatenate((class_vec_test,class_vec_out))

    s_eval = np.arange(class_vec_eval.shape[0])
    np.random.shuffle(s_eval)
    class_vec_eval = class_vec_eval[s_eval]
    data_eval = data_eval[s_eval]

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
    print("grid_rdf search best parameters: {}".format(grid_rdf.best_params_))

    predict_rdf_train = grid_rdf.predict(data_train)
    predict_rdf_eval = grid_rdf.predict(data_eval)

    # evaluate the error on a training and an evaluation dataset and store it
    train_rdf_err = 1-(predict_rdf_train == class_vec_train.T[0]).mean()
    eval_rdf_err = 1-(predict_rdf_eval == class_vec_eval.T[0]).mean()

    acc_rdf = grid_rdf.score(data_eval,class_vec_eval)
    accuracies_rdf.append(acc_rdf*100)
    print("grid_rdf accuracy: {:.2f}%".format(acc_rdf * 100))

    conf_rdf = confusion_matrix(class_vec_eval, predict_rdf_eval)
    conf_rdf_norm = conf_rdf/conf_rdf.sum(axis=1)[:, np.newaxis]
    tn, fp, fn, tp = conf_rdf_norm.ravel()

    true_negatives_rdf.append(tn)
    false_positives_rdf.append(fp)
    false_negatives_rdf.append(fn)
    true_positives_rdf.append(tp)

    precision_rdf, recall_rdf, fscore_rdf, support_rdf = precision_recall_fscore_support(class_vec_eval, predict_rdf_eval, average='binary')

    precisions_rdf.append(precision_rdf)
    recalls_rdf.append(recall_rdf)
    fscores_rdf.append(fscore_rdf)

    parameters_rdf = grid_rdf.best_params_
    evaluation_rdf = dict([('conf_rdf',conf_rdf.tolist()),('conf_rdf_norm',conf_rdf_norm.tolist()),('accuracy_rdf',acc_rdf),('precision_rdf',precision_rdf),('recall_rdf',recall_rdf),('fscore_rdf',fscore_rdf),('support_rdf',support_rdf)])

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
    print("grid_svm search best parameters: {}".format(grid_svm.best_params_))

    predict_svm_train = grid_svm.predict(data_train)
    predict_svm_eval = grid_svm.predict(data_eval)

    # evaluate the error on a training and an evaluation dataset and store it
    train_svm_err = 1-(predict_svm_train == class_vec_train.T[0]).mean()
    eval_svm_err = 1-(predict_svm_eval == class_vec_eval.T[0]).mean()

    acc_svm = grid_svm.score(data_eval,class_vec_eval)
    accuracies_svm.append(acc_svm*100)
    print("grid_svm accuracy: {:.2f}%".format(acc_svm * 100))

    conf_svm = confusion_matrix(class_vec_eval, predict_svm_eval)
    conf_svm_norm = conf_svm/conf_svm.sum(axis=1)[:, np.newaxis]
    tn, fp, fn, tp = conf_svm_norm.ravel()

    true_negatives_svm.append(tn)
    false_positives_svm.append(fp)
    false_negatives_svm.append(fn)
    true_positives_svm.append(tp)

    precision_svm, recall_svm, fscore_svm, support_svm = precision_recall_fscore_support(class_vec_eval, predict_svm_eval, average='binary')

    precisions_svm.append(precision_svm)
    recalls_svm.append(recall_svm)
    fscores_svm.append(fscore_svm)

    parameters_svm = grid_svm.best_params_
    evaluation_svm = dict([('conf_svm',conf_svm.tolist()),('conf_svm_norm',conf_svm_norm.tolist()),('accuracy_svm',acc_svm),('precision_svm',precision_svm),('recall_svm',recall_svm),('fscore_svm',fscore_svm),('support_svm',support_svm)])

    with open(path+'classification_and_prediction/results/'+percent+'-merged/'+patient_id+'/'+patient_id+'_'+run_nr+'.txt', 'w') as data_file:
        json.dump((parameters_svm,evaluation_svm,parameters_rdf,evaluation_rdf),data_file)

    with open(path+'classification_and_prediction/models/'+percent+'-merged/'+patient_id+'/forests/'+'forest_'+patient_id+'_'+run_nr+'.pickle', 'wb') as f:
        pickle.dump(grid_rdf, f)

    with open(path+'classification_and_prediction/models/'+percent+'-merged/'+patient_id+'/svm_models/'+'svm_'+patient_id+'_'+run_nr+'.pickle', 'wb') as s:
        pickle.dump(grid_svm, s)


accuracy_rdf = np.mean(accuracies_rdf)
accuracy_svm = np.mean(accuracies_svm)

precision_rdf = np.mean(precisions_rdf)
precision_svm = np.mean(precisions_svm)

recall_rdf = np.mean(recalls_rdf)
recall_svm = np.mean(recalls_svm)

fscore_rdf = np.mean(fscores_rdf)
fscore_svm = np.mean(fscores_svm)

final_evaluation = dict([('accuracy_rdf',accuracy_rdf),('accuracy_svm',accuracy_svm),('precision_rdf',precision_rdf),('precision_svm',precision_svm),('recall_rdf',recall_rdf),('recall_svm',recall_svm)])
final_evaluation["fscore_rdf"] = fscore_rdf
final_evaluation["fscore_svm"] = fscore_svm

avg_true_negatives_rdf = np.mean(true_negatives_rdf)
avg_false_positives_rdf = np.mean(false_positives_rdf)
avg_false_negatives_rdf = np.mean(false_negatives_rdf)
avg_true_positives_rdf = np.mean(true_positives_rdf)

avg_true_negatives_svm = np.mean(true_negatives_svm)
avg_false_positives_svm = np.mean(false_positives_svm)
avg_false_negatives_svm = np.mean(false_negatives_svm)
avg_true_positives_svm = np.mean(true_positives_svm)

conf_matrices_final = dict([('avg_true_negatives_rdf',avg_true_negatives_rdf),('avg_false_positives_rdf',avg_false_positives_rdf),('avg_false_negatives_rdf',avg_false_negatives_rdf),('avg_true_positives_rdf',avg_true_positives_rdf)])
conf_matrices_final["avg_true_negatives_svm"] = avg_true_negatives_svm
conf_matrices_final["avg_false_positives_svm"] = avg_false_positives_svm
conf_matrices_final["avg_false_negatives_svm"] = avg_false_negatives_svm
conf_matrices_final["avg_true_positives_svm"] = avg_true_positives_svm

with open(path+'classification_and_prediction/results/'+percent+'-merged/'+patient_id+'/all_results_'+patient_id+'.txt', 'w') as final_file:
    json.dump((final_evaluation,conf_matrices_final),final_file,indent=True)
