
import os, pickle, json
import numpy as np
import scipy.io as sio
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

patient_id = '109602'  #[11502, 25302, 59002, 62002, 97002, 109602]
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_'+patient_id+'_extracted_seizures/'

num_channels = {"11502":48, "25302": 26, "59002": 94, "62002": 38, "97002": 91, "109602": 68}
size_freq_component = 502
size_time_component = 29
num_folds = 100

####################################################################################################################################################################

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

### obtaining data ###
files_interictal = []
for root_interictal, dirs, files in os.walk(path+'data/models_interictal/',topdown=True):
    files_interictal.append(files)
interictal = np.zeros((num_channels[patient_id], size_freq_component+size_time_component, len(files_interictal[0])))

for idx, val in enumerate(files_interictal[0]):
    current = sio.loadmat(os.path.join(root_interictal,val))
    interictal[:,:,idx] = np.hstack((current['H_model_baseline'], current['W_model_baseline']))

files_preictal = []
for root_preictal, dirs, files in os.walk(path+'data/models_preictal/',topdown=True):
    files_preictal.append(files)
preictal = np.zeros((num_channels[patient_id], size_freq_component+size_time_component, len(files_preictal[0])))

for idx, val in enumerate(files_preictal[0]):
    current = sio.loadmat(os.path.join(root_preictal,val))
    preictal[:,:,idx] = np.hstack((current['H_model_preictal'], current['W_model_preictal']))

data = np.concatenate((interictal, preictal), axis=2)
labels = np.concatenate((np.zeros(len(files_interictal[0])) , np.ones(len(files_preictal[0]))))

s = np.arange(labels.shape[0])
np.random.shuffle(s)
labels = labels[s]
data = data[:,:,s]

for idx in range(num_folds):
    print(idx)

    ### svm classifier ###
    data_train, data_test, label_train, label_test = train_test_split(
    data.reshape((num_channels[patient_id]*(size_freq_component+size_time_component), -1)).T, labels, test_size=0.3, shuffle=True)

    model_svm = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, max_iter=5000)
    model_svm.fit(data_train, label_train)
    parameters_svm = model_svm.get_params()

    predict_svm = model_svm.predict(data_test)
    accuracy_svm[idx,:] = (predict_svm == label_test).mean()

    conf_svm = confusion_matrix(label_test, predict_svm, labels=[0, 1]).ravel()
    tn_svm[idx, :], fp_svm[idx, :], fn_svm[idx, :], tp_svm[idx, :] = conf_svm/conf_svm.sum()

    specificity_svm[idx, :] = tn_svm[idx, :] / (tn_svm[idx, :] + fp_svm[idx, :])
    ppv_svm[idx, :]  = tp_svm[idx, :] / (tp_svm[idx, :] + fp_svm[idx, :])
    npv_svm[idx, :] = tn_svm[idx, :] / (tn_svm[idx, :] + fn_svm[idx, :])

    precision_svm[idx, :], sensitivity_svm[idx, :], fscore_svm[idx, :], _ = precision_recall_fscore_support(label_test, predict_svm, average='binary')

#############################################################################################################################################################

    ### smote+svm classifier ###
    smote = SMOTE(k_neighbors=5, sampling_strategy="minority")
    X_smote, y_smote = smote.fit_resample(data.reshape((num_channels[patient_id]*(size_freq_component+size_time_component), -1)).T, labels)
    smote_data_train, smote_data_test, smote_label_train, smote_label_test = train_test_split(X_smote, y_smote, test_size=0.3, shuffle=True)

    model_smote = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, max_iter=5000)
    model_smote.fit(smote_data_train, smote_label_train)
    parameters_smote = model_smote.get_params()

    predict_smote = model_smote.predict(smote_data_test)
    accuracy_smote[idx, :] = (predict_smote == smote_label_test).mean()

    conf_smote = confusion_matrix(smote_label_test, predict_smote, labels=[0, 1]).ravel()
    tn_smote[idx, :], fp_smote[idx, :], fn_smote[idx, :], tp_smote[idx, :] = conf_smote/conf_smote.sum()

    specificity_smote[idx, :] = tn_smote[idx, :] / (tn_smote[idx, :] + fp_smote[idx, :])
    ppv_smote[idx, :] = tp_smote[idx, :] / (tp_smote[idx, :] + fp_smote[idx, :])
    npv_smote[idx, :] = tn_smote[idx, :] / (tn_smote[idx, :] + fn_smote[idx, :])

    precision_smote[idx, :], sensitivity_smote[idx, :], fscore_smote[idx, :], _ = precision_recall_fscore_support(smote_label_test, predict_smote, average='binary')

#############################################################################################################################################################

    evaluation_svm = dict([('accuracy_svm', accuracy_svm[idx, :].tolist()),
                           ('sensitivity_svm', sensitivity_svm[idx, :].tolist()),
                           ('specificity_svm', specificity_svm[idx, :].tolist()),
                           ('precision_svm', precision_svm[idx, :].tolist()),
                           ('fscore_svm', fscore_svm[idx, :].tolist()),
                           ("positive_predictive_value_svm", ppv_svm[idx, :].tolist()),
                           ("negative_predictive_value_svm", npv_svm[idx, :].tolist()),
                           ('conf_svm', conf_svm.tolist())])

    evaluation_smote = dict([('accuracy_smote', accuracy_smote[idx, :].tolist()),
                             ('sensitivity_smote', sensitivity_smote[idx, :].tolist()),
                             ('specificity_smote', specificity_smote[idx, :].tolist()),
                             ('precision_smote', precision_smote[idx, :].tolist()),
                             ('fscore_smote', fscore_smote[idx, :].tolist()),
                             ("positive_predictive_value_smote", ppv_smote[idx, :].tolist()),
                             ("negative_predictive_value_smote", npv_smote[idx, :].tolist()),
                             ('conf_smote', conf_smote.tolist())])

#############################################################################################################################################################

    with open(path+'results/results_'+patient_id+'_'+str(idx)+'.txt', 'w') as results_file:
        json.dump((parameters_svm, evaluation_svm, parameters_smote, evaluation_smote), results_file)

    with open(path+'prediction_models/svm_'+patient_id+'_'+str(idx)+'.pickle', 'wb') as svm_file:
        pickle.dump(model_svm, svm_file)

    with open(path+'prediction_models/smote_'+patient_id+'_'+str(idx)+'.pickle', 'wb') as smote_file:
        pickle.dump(model_smote, smote_file)

#############################################################################################################################################################

evaluation_final = dict([('accuracy_svm', np.mean(accuracy_svm)),
                         ('precision_svm', np.mean(precision_svm)),
                         ('sensitivity_svm', np.mean(sensitivity_svm)),
                         ('specificity_svm', np.mean(specificity_svm)),
                         ("fscore_svm", np.mean(fscore_svm)),
                         ("positive_predictive_svm", np.mean(ppv_svm)),
                         ("negative_predictive_svm", np.mean(npv_svm)),

                         ('accuracy_smote', np.mean(accuracy_smote)),
                         ('precision_smote', np.mean(precision_smote)),
                         ('sensitivity_smote', np.mean(sensitivity_smote)),
                         ('specificity_smote', np.mean(specificity_smote)),
                         ("fscore_smote", np.mean(fscore_smote)),
                         ("positive_predictive_smote", np.mean(ppv_smote)),
                         ("negative_predictive_smote", np.mean(npv_smote))])

conf_matrices_final = dict([("true_negative_svm", np.mean(tn_svm)),
                            ("false_positive_svm", np.mean(fp_svm)),
                            ("false_negative_svm", np.mean(fn_svm)),
                            ("true_positive_svm", np.mean(tp_svm)),

                            ('true_negative_smote', np.mean(tn_smote)),
                            ('false_positive_smote', np.mean(fp_smote)),
                            ('false_negative_smote', np.mean(fn_smote)),
                            ('true_positive_smote', np.mean(tp_smote))])

with open(path+'results/all_results_'+patient_id+'.txt', 'w') as final_file:
    json.dump((evaluation_final, conf_matrices_final), final_file, indent=True)
