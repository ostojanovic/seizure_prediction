
import os, json, csv
import numpy as np, pickle as pkl
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

patient_id = 'Pat1'  # patient id goes here
set = "Train"
benchmark_set = "Test"
path = "*"   # path goes here
benchmark_path = "*"   # path goes here
num_channels = 16   # number of channels for each patient goes here
size_freq_parameters = 24
size_time_parameters = 3
num_folds = 100

####################################################################################################################################################################

coefficients_svm = []
coefficients_smote = []

accuracy_svm = np.zeros((num_folds,1))
accuracy_smote = np.zeros((num_folds,1))

sensitivity_svm = np.zeros((num_folds,1))
sensitivity_smote = np.zeros((num_folds,1))

specificity_svm = np.zeros((num_folds,1))
specificity_smote = np.zeros((num_folds,1))

precision_svm = np.zeros((num_folds,1))
precision_smote = np.zeros((num_folds,1))

fscore_svm = np.zeros((num_folds,1))
fscore_smote = np.zeros((num_folds,1))

ppv_svm = np.zeros((num_folds,1))
ppv_smote = np.zeros((num_folds,1))

npv_svm = np.zeros((num_folds,1))
npv_smote = np.zeros((num_folds,1))

tn_svm = np.zeros((num_folds,1))
tn_smote = np.zeros((num_folds,1))

fp_svm = np.zeros((num_folds,1))
fp_smote = np.zeros((num_folds,1))

fn_svm = np.zeros((num_folds,1))
fn_smote = np.zeros((num_folds,1))

tp_svm = np.zeros((num_folds,1))
tp_smote = np.zeros((num_folds,1))

fpr_smote = np.zeros((num_folds,3))
fpr_svm = np.zeros((num_folds,3))

tpr_smote = np.zeros((num_folds,3))
tpr_svm = np.zeros((num_folds,3))

thresholds_smote = np.zeros((num_folds,3))
thresholds_svm = np.zeros((num_folds,3))

auc_smote = np.zeros((num_folds,1))
auc_svm = np.zeros((num_folds,1))

num_smote_samples_preictal = np.zeros((num_folds,1))
num_smote_samples_interictal = np.zeros((num_folds,1))

### obtaining data ###
files_interictal = []
for root_interictal, dirs, files in os.walk('*',topdown=True): # path to a folder with interictal models goes here
    files_interictal.append(files)
interictal = np.zeros((num_channels, size_freq_parameters+size_time_parameters, len(files_interictal[0])))

for idx, val in enumerate(files_interictal[0]):

    with open("*"+val, "rb") as f: # path to a folder with interictal models goes here before '+'
        current = pkl.load(f)

    interictal[:,:,idx] = np.hstack((current['H_coeff'], current['W_coeff']))

files_preictal = []
for root_preictal, dirs, files in os.walk('*',topdown=True): # path to a folder with preictal models goes here
    files_preictal.append(files)
preictal = np.zeros((num_channels, size_freq_parameters+size_time_parameters, len(files_preictal[0])))

for idx, val in enumerate(files_preictal[0]):

    with open("*"+val, "rb") as f: # path to a folder with preictal models goes here before '+'
        current = pkl.load(f)

    preictal[:,:,idx] = np.hstack((current['H_coeff'], current['W_coeff']))

data = np.concatenate((interictal, preictal), axis=2)
labels = np.concatenate((np.zeros(len(files_interictal[0])) , np.ones(len(files_preictal[0]))))

### benchmark set ###
benchmark_filenames = []

benchmark_files = os.listdir('*') # path to a folder with benchmark models goes here
benchmark_files.sort()

benchmark_data = np.zeros((num_channels, size_freq_parameters+size_time_parameters, len(benchmark_files)))

for idx_benchmark_file, benchmark_file in enumerate(benchmark_files):
    with open("*"+benchmark_file, "rb") as f: # path to a folder with benchmark models goes here before '+'
        current = pkl.load(f)
    benchmark_data[:,:,idx_benchmark_file] = np.hstack((current['H_coeff'], current['W_coeff']))
    benchmark_filenames.append(benchmark_file.split("_model.pkl")[0]+".csv")

benchmark_data = benchmark_data.reshape((num_channels*(size_freq_parameters+size_time_parameters), -1)).T
predict_smote_benchmark = np.zeros((num_folds, benchmark_data.shape[0]))
predict_svm_benchmark = np.zeros((num_folds, benchmark_data.shape[0]))

for idx in range(num_folds):
    print(idx)

    ### svm classifier ###
    data_train, data_test, label_train, label_test = train_test_split(
    data.reshape((num_channels*(size_freq_parameters+size_time_parameters), -1)).T, labels, test_size=0.3, shuffle=True)

    model_svm = LinearSVC(penalty="l1", dual=False, max_iter=5000)
    model_svm.fit(data_train, label_train)
    parameters_svm = model_svm.get_params()
    coefficients_svm.append(model_svm.coef_.tolist())

    predict_svm = model_svm.predict(data_test)
    accuracy_svm[idx,:] = (predict_svm == label_test).mean()

    conf_svm = confusion_matrix(label_test, predict_svm, labels=[0, 1]).ravel()
    tn_svm[idx, :], fp_svm[idx, :], fn_svm[idx, :], tp_svm[idx, :] = conf_svm/conf_svm.sum()

    specificity_svm[idx, :] = tn_svm[idx, :] / (tn_svm[idx, :] + fp_svm[idx, :])
    ppv_svm[idx, :]  = tp_svm[idx, :] / (tp_svm[idx, :] + fp_svm[idx, :])
    npv_svm[idx, :] = tn_svm[idx, :] / (tn_svm[idx, :] + fn_svm[idx, :])

    precision_svm[idx, :], sensitivity_svm[idx, :], fscore_svm[idx, :], _ = precision_recall_fscore_support(label_test, predict_svm, average='binary')

    fpr_svm[idx,:], tpr_svm[idx,:], thresholds_svm[idx,:] = roc_curve(predict_svm, label_test, pos_label=1)
    auc_svm[idx,:] = auc(fpr_svm[idx,:], tpr_svm[idx,:])

    ### benchmark set ###
    predict_svm_benchmark[idx,:] = model_svm.predict(benchmark_data)

#############################################################################################################################################################

    ### smote+svm classifier ###
    smote = SMOTE(k_neighbors=5, sampling_strategy="minority")
    X_smote, y_smote = smote.fit_resample(data.reshape((num_channels*(size_freq_parameters+size_time_parameters), -1)).T, labels)

    num_smote_samples_preictal[idx, :] = np.count_nonzero(y_smote)  # need this only to report it in a paper
    num_smote_samples_interictal[idx, :] = np.shape(y_smote)[0] - np.count_nonzero(y_smote)

    smote_data_train, smote_data_test, smote_label_train, smote_label_test = train_test_split(X_smote, y_smote, test_size=0.3, shuffle=True)

    model_smote = LinearSVC(penalty="l1", dual=False, max_iter=5000)
    model_smote.fit(smote_data_train, smote_label_train)
    parameters_smote = model_smote.get_params()
    coefficients_smote.append(model_smote.coef_.tolist())

    predict_smote = model_smote.predict(smote_data_test)
    accuracy_smote[idx, :] = (predict_smote == smote_label_test).mean()

    conf_smote = confusion_matrix(smote_label_test, predict_smote, labels=[0, 1]).ravel()
    tn_smote[idx, :], fp_smote[idx, :], fn_smote[idx, :], tp_smote[idx, :] = conf_smote/conf_smote.sum()

    specificity_smote[idx, :] = tn_smote[idx, :] / (tn_smote[idx, :] + fp_smote[idx, :])
    ppv_smote[idx, :] = tp_smote[idx, :] / (tp_smote[idx, :] + fp_smote[idx, :])
    npv_smote[idx, :] = tn_smote[idx, :] / (tn_smote[idx, :] + fn_smote[idx, :])

    precision_smote[idx, :], sensitivity_smote[idx, :], fscore_smote[idx, :], _ = precision_recall_fscore_support(smote_label_test, predict_smote, average='binary')

    fpr_smote[idx, :], tpr_smote[idx, :], thresholds_smote[idx, :] = roc_curve(predict_smote, smote_label_test, pos_label=1)
    auc_smote[idx,:] = auc(fpr_smote[idx,:], tpr_smote[idx,:])

    ### benchmark set ###
    predict_smote_benchmark[idx, :] = model_smote.predict(benchmark_data)

#############################################################################################################################################################

    evaluation_svm = dict([('accuracy_svm', accuracy_svm[idx, :].tolist()),
                           ('sensitivity_svm', sensitivity_svm[idx, :].tolist()),
                           ('specificity_svm', specificity_svm[idx, :].tolist()),
                           ('precision_svm', precision_svm[idx, :].tolist()),
                           ('fscore_svm', fscore_svm[idx, :].tolist()),
                           ("positive_predictive_value_svm", ppv_svm[idx, :].tolist()),
                           ("negative_predictive_value_svm", npv_svm[idx, :].tolist()),
                           ('conf_svm', conf_svm.tolist()),
                           ('auc_svm', auc_svm[idx,:].tolist())])

    evaluation_smote = dict([('accuracy_smote', accuracy_smote[idx, :].tolist()),
                             ('sensitivity_smote', sensitivity_smote[idx, :].tolist()),
                             ('specificity_smote', specificity_smote[idx, :].tolist()),
                             ('precision_smote', precision_smote[idx, :].tolist()),
                             ('fscore_smote', fscore_smote[idx, :].tolist()),
                             ("positive_predictive_value_smote", ppv_smote[idx, :].tolist()),
                             ("negative_predictive_value_smote", npv_smote[idx, :].tolist()),
                             ('conf_smote', conf_smote.tolist()),
                             ('auc_smote', auc_smote[idx,:].tolist())])

#############################################################################################################################################################

    with open('*.txt' , 'w') as results_file:    # new path and name go here instead of *
        json.dump((parameters_svm, evaluation_svm, coefficients_svm, parameters_smote, coefficients_smote, evaluation_smote), results_file)

    with open('*.pkl', 'wb') as svm_file:    # new path and name go here instead of *
        pkl.dump(model_svm, svm_file)

    with open('*.pkl', 'wb') as smote_file:  # new path and name go here instead of *
        pkl.dump(model_smote, smote_file)

#############################################################################################################################################################

evaluation_final = dict([('accuracy_svm', np.mean(accuracy_svm)),
                         ('precision_svm', np.mean(precision_svm)),
                         ('sensitivity_svm', np.mean(sensitivity_svm)),
                         ('specificity_svm', np.mean(specificity_svm)),
                         ("fscore_svm", np.mean(fscore_svm)),
                         ("positive_predictive_svm", np.mean(ppv_svm)),
                         ("negative_predictive_svm", np.mean(npv_svm)),
                         ("auc_svm", np.mean(auc_svm)),
                         ("num_files_preictal", len(files_preictal[0])),
                         ("num_files_interictal", len(files_interictal[0])),

                         ('accuracy_smote', np.mean(accuracy_smote)),
                         ('precision_smote', np.mean(precision_smote)),
                         ('sensitivity_smote', np.mean(sensitivity_smote)),
                         ('specificity_smote', np.mean(specificity_smote)),
                         ("fscore_smote", np.mean(fscore_smote)),
                         ("positive_predictive_smote", np.mean(ppv_smote)),
                         ("negative_predictive_smote", np.mean(npv_smote)),
                         ("auc_smote", np.mean(auc_smote)),
                         ("num_smote_samples_preictal", np.mean(num_smote_samples_preictal)),
                         ("num_smote_samples_interictal", np.mean(num_smote_samples_interictal))])

conf_matrices_final = dict([("true_negative_svm", np.mean(tn_svm)),
                            ("false_positive_svm", np.mean(fp_svm)),
                            ("false_negative_svm", np.mean(fn_svm)),
                            ("true_positive_svm", np.mean(tp_svm)),

                            ('true_negative_smote', np.mean(tn_smote)),
                            ('false_positive_smote', np.mean(fp_smote)),
                            ('false_negative_smote', np.mean(fn_smote)),
                            ('true_positive_smote', np.mean(tp_smote))])

with open('*.txt', 'w') as final_file:    # new path and name go here instead of *
    json.dump((evaluation_final, conf_matrices_final), final_file, indent=True)

### benchmark predictions ###
min_auc_svm = np.min(auc_svm)
argmin_auc_svm = np.argmin(auc_svm)
max_auc_svm = np.max(auc_svm)
argmax_auc_svm = np.argmax(auc_svm)

min_auc_smote = np.min(auc_smote)
argmin_auc_smote = np.argmin(auc_smote)
max_auc_smote = np.max(auc_smote)
argmax_auc_smote = np.argmax(auc_smote)

benchmark_final_svm_min_auc = predict_svm_benchmark[argmin_auc_svm, :].astype(int)
benchmark_final_smote_min_auc = predict_smote_benchmark[argmin_auc_smote, :].astype(int)

benchmark_final_svm_max_auc = predict_svm_benchmark[argmax_auc_svm, :].astype(int)
benchmark_final_smote_max_auc = predict_smote_benchmark[argmax_auc_smote, :].astype(int)

final_svm_avg = np.mean(predict_svm_benchmark, axis=0).astype(int)
final_smote_avg = np.mean(predict_smote_benchmark, axis=0).astype(int)

final_svm_weighted = np.heaviside(np.average(predict_svm_benchmark, weights=accuracy_svm.reshape(-1), axis=0), 0).astype(int)
final_smote_weighted = np.heaviside(np.average(predict_smote_benchmark, weights=accuracy_smote.reshape(-1), axis=0), 0).astype(int)

benchmark_params = dict([('min_auc_svm', min_auc_svm.tolist()),
                         ('argmin_auc_svm', argmin_auc_svm.tolist()),
                         ('max_auc_svm', max_auc_svm.tolist()),
                         ('argmax_auc_svm', argmax_auc_svm.tolist()),
                         ('min_auc_smote', min_auc_smote.tolist()),
                         ("argmin_auc_smote", argmin_auc_smote.tolist()),
                         ("max_auc_smote", max_auc_smote.tolist()),
                         ('argmax_auc_smote', argmax_auc_smote.tolist())])

with open('*.txt' , 'w') as benchmark_params_file: # new path and name go here instead of *
    json.dump((benchmark_params), benchmark_params_file)

csv_smote_weighted = np.stack((benchmark_filenames, final_smote_weighted), axis=1)
write_file_smote_weighted = open("*.csv", "w") # new path and name go here instead of *
with write_file_smote_weighted:
    writer = csv.writer(write_file_smote_weighted)
    writer.writerows(csv_smote_weighted)

csv_smote_avg = np.stack((benchmark_filenames, final_smote_avg), axis=1)
write_file_smote_avg = open("*.csv", "w") # new path and name go here instead of *
with write_file_smote_avg:
    writer = csv.writer(write_file_smote_avg)
    writer.writerows(csv_smote_avg)

csv_smote_max_auc = np.stack((benchmark_filenames, benchmark_final_smote_max_auc), axis=1)
write_file_smote_max_auc = open("*.csv", "w") # new path and name go here instead of *
with write_file_smote_max_auc:
    writer = csv.writer(write_file_smote_max_auc)
    writer.writerows(csv_smote_max_auc)
