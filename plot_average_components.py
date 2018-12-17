
from __future__ import unicode_literals
import os
import scipy.io as sio
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec, transforms
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots average time and frequency components of preictal and interictal states.
"""

ident = '109602'
patient_id = '109602'
run_nr = '1'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/'+run_nr+'/'
baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/'+run_nr+'/'

preictal_train = preictal_directory+'train/'
preictal_test = preictal_directory+'test/'
preictal_out = preictal_directory+'out-of-sample/'

baseline_train = baseline_directory+'train/'
baseline_test = baseline_directory+'test/'
baseline_out = baseline_directory+'out-of-sample/'

files_preictal_train = os.listdir(path_directory+preictal_train)
files_preictal_test = os.listdir(path_directory+preictal_test)
files_preictal_out = os.listdir(path_directory+preictal_out)

files_baseline_train = os.listdir(path_directory+baseline_train)
files_baseline_test = os.listdir(path_directory+baseline_test)
files_baseline_out = os.listdir(path_directory+baseline_out)

idxc = 0

# plotting the models
fig1 = plt.figure(figsize=(19.5,10.2))
gr = gridspec.GridSpec(nrows=2, ncols=2,left=0.06,wspace=0.17, hspace=0.20, width_ratios=[1, 1], height_ratios=[1, 1])

# defining the grids
ax30 = fig1.add_subplot(gr[1,0])
ax10 = fig1.add_subplot(gr[0,0],sharex=ax30)
ax40 = fig1.add_subplot(gr[1,1])
ax20 = fig1.add_subplot(gr[0,1],sharex=ax40)

for tick in ax10.get_xticklabels():
    tick.set_visible(False)

for tick in ax20.get_xticklabels():
    tick.set_visible(False)

W_preictal_train = []
W_preictal_test = []
W_preictal_out = []

H_preictal_train = []
H_preictal_test = []
H_preictal_out = []

W_baseline_train = []
W_baseline_test = []
W_baseline_out = []

H_baseline_train = []
H_baseline_test = []
H_baseline_out = []

for n in range(len(files_preictal_train)):

    dict_models_preictal = sio.loadmat(path_directory+preictal_train+files_preictal_train[n])
    W_preictal = dict_models_preictal["W_preictal"]
    H_preictal = dict_models_preictal["H_preictal"]
    H_model_preictal = dict_models_preictal["H_model_preictal"]
    W_model_preictal = dict_models_preictal["W_model_preictal"]

    #ax10.plot(W_preictal[idxc,:])
    #ax10.plot(W_model_preictal[idxc,:],'--')
    ax10.set_title('Model of a time component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax10.set_ylabel('Time coefficients',fontsize=12)
    ax10.legend(('Time component','Model of a time component'))

    #ax20.plot(H_preictal[idxc,:])
    #ax20.plot(H_model_preictal[idxc,:],'--')
    ax20.set_title('Model of a frequency component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax20.set_ylabel('Frequency coefficients',fontsize=12)
    ax20.legend(('Frequency component','Model of a frequency component'))

    W_preictal_train.append(W_model_preictal[idxc,:])
    H_preictal_train.append(H_model_preictal[idxc,:])

for n in range(len(files_preictal_test)):

    dict_models_preictal = sio.loadmat(path_directory+preictal_test+files_preictal_test[n])
    W_preictal = dict_models_preictal["W_preictal"]
    H_preictal = dict_models_preictal["H_preictal"]
    H_model_preictal = dict_models_preictal["H_model_preictal"]
    W_model_preictal = dict_models_preictal["W_model_preictal"]

    #ax10.plot(W_preictal[idxc,:])
    #ax10.plot(W_model_preictal[idxc,:],'--')
    ax10.set_title('Model of a time component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax10.set_ylabel('Time coefficients',fontsize=12)
    ax10.legend(('Time component','Model of a time component'))

    #ax20.plot(H_preictal[idxc,:])
    #ax20.plot(H_model_preictal[idxc,:],'--')
    ax20.set_title('Model of a frequency component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax20.set_ylabel('Frequency coefficients',fontsize=12)
    ax20.legend(('Frequency component','Model of a frequency component'))

    W_preictal_test.append(W_model_preictal[idxc,:])
    H_preictal_test.append(H_model_preictal[idxc,:])

for n in range(len(files_preictal_out)):

    dict_models_preictal = sio.loadmat(path_directory+preictal_out+files_preictal_out[n])
    W_preictal = dict_models_preictal["W_preictal"]
    H_preictal = dict_models_preictal["H_preictal"]
    H_model_preictal = dict_models_preictal["H_model_preictal"]
    W_model_preictal = dict_models_preictal["W_model_preictal"]

    #ax10.plot(W_preictal[idxc,:])
    #ax10.plot(W_model_preictal[idxc,:],'--')
    ax10.set_title('Model of a time component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax10.set_ylabel('Time coefficients',fontsize=12)
    ax10.legend(('Time component','Model of a time component'))

    #ax20.plot(H_preictal[idxc,:])
    #ax20.plot(H_model_preictal[idxc,:],'--')
    ax20.set_title('Model of a frequency component of a preictal state (channel: HR1, patient 1)',fontsize=14)
    ax20.set_ylabel('Frequency coefficients',fontsize=12)
    ax20.legend(('Frequency component','Model of a frequency component'))

    W_preictal_out.append(W_model_preictal[idxc,:])
    H_preictal_out.append(H_model_preictal[idxc,:])

for n in range(len(files_baseline_train)):

    dict_models_baseline = sio.loadmat(path_directory+baseline_train+files_baseline_train[n])
    W_baseline = dict_models_baseline["W_baseline"]
    H_baseline = dict_models_baseline["H_baseline"]
    H_model_baseline = dict_models_baseline["H_model_baseline"]
    W_model_baseline = dict_models_baseline["W_model_baseline"]

    #ax30.plot(W_baseline[idxc,:])
    #ax30.plot(W_model_baseline[idxc,:],'--')
    ax30.set_title('Model of a time component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax30.set_xlabel('Time (min)',fontsize=12)
    ax30.set_xticks([0,6,12,18,24,28])
    ax30.set_xticklabels([0,1,2,3,4,5])
    ax30.set_ylabel('Time coefficients',fontsize=12)
    ax30.legend(('Time component','Model of a time component'))

    #ax40.plot(H_baseline[idxc,:])
    #ax40.plot(H_model_baseline[idxc,:],'--')
    ax40.set_title('Model of a frequency component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax40.set_xlabel('Frequency (Hz)',fontsize=12)
    ax40.set_xticks([0,125,250,375,500])
    ax40.set_xticklabels([0,32,64,96,128],fontsize=10)
    ax40.set_ylabel('Frequency coefficients',fontsize=12)
    ax40.legend(('Frequency component','Model of a frequency component'))

    W_baseline_train.append(W_model_baseline[idxc,:])
    H_baseline_train.append(H_model_baseline[idxc,:])

for n in range(len(files_baseline_test)):

    dict_models_baseline = sio.loadmat(path_directory+baseline_test+files_baseline_test[n])
    W_baseline = dict_models_baseline["W_baseline"]
    H_baseline = dict_models_baseline["H_baseline"]
    H_model_baseline = dict_models_baseline["H_model_baseline"]
    W_model_baseline = dict_models_baseline["W_model_baseline"]

    #ax30.plot(W_baseline[idxc,:])
    #ax30.plot(W_model_baseline[idxc,:],'--')
    ax30.set_title('Model of a time component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax30.set_xlabel('Time (min)',fontsize=12)
    ax30.set_xticks([0,6,12,18,24,28])
    ax30.set_xticklabels([0,1,2,3,4,5])
    ax30.set_ylabel('Time coefficients',fontsize=12)
    ax30.legend(('Time component','Model of a time component'))

    #ax40.plot(H_baseline[idxc,:])
    #ax40.plot(H_model_baseline[idxc,:],'--')
    ax40.set_title('Model of a frequency component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax40.set_xlabel('Frequency (Hz)',fontsize=12)
    ax40.set_xticks([0,125,250,375,500])
    ax40.set_xticklabels([0,32,64,96,128],fontsize=10)
    ax40.set_ylabel('Frequency coefficients',fontsize=12)
    ax40.legend(('Frequency component','Model of a frequency component'))

    W_baseline_test.append(W_model_baseline[idxc,:])
    H_baseline_test.append(H_model_baseline[idxc,:])

for n in range(len(files_baseline_out)):

    dict_models_baseline = sio.loadmat(path_directory+baseline_out+files_baseline_out[n])
    W_baseline = dict_models_baseline["W_baseline"]
    H_baseline = dict_models_baseline["H_baseline"]
    H_model_baseline = dict_models_baseline["H_model_baseline"]
    W_model_baseline = dict_models_baseline["W_model_baseline"]

    #ax30.plot(W_baseline[idxc,:])
    #ax30.plot(W_model_baseline[idxc,:],'--')
    ax30.set_title('Model of a time component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax30.set_xlabel('Time (min)',fontsize=12)
    ax30.set_xticks([0,6,12,18,24,28])
    ax30.set_xticklabels([0,1,2,3,4,5])
    ax30.set_ylabel('Time coefficients',fontsize=12)
    ax30.legend(('Time component','Model of a time component'))

    #ax40.plot(H_baseline[idxc,:])
    #ax40.plot(H_model_baseline[idxc,:],'--')
    ax40.set_title('Model of a frequency component of an interictal state (channel: HR1, patient 1)',fontsize=14)
    ax40.set_xlabel('Frequency (Hz)',fontsize=12)
    ax40.set_xticks([0,125,250,375,500])
    ax40.set_xticklabels([0,32,64,96,128],fontsize=10)
    ax40.set_ylabel('Frequency coefficients',fontsize=12)
    ax40.legend(('Frequency component','Model of a frequency component'))

    W_baseline_out.append(W_model_baseline[idxc,:])
    H_baseline_out.append(H_model_baseline[idxc,:])

W_preictal_all = np.concatenate((W_preictal_train,W_preictal_test,W_preictal_out))
H_preictal_all = np.concatenate((H_preictal_train,H_preictal_test,H_preictal_out))

W_baseline_all = np.concatenate((W_baseline_train,W_baseline_test,W_baseline_out))
H_baseline_all = np.concatenate((H_baseline_train,H_baseline_test,H_baseline_out))

W_preictal_avg = np.mean(W_preictal_all,axis=0)
H_preictal_avg = np.mean(H_preictal_all,axis=0)

W_baseline_avg = np.mean(W_baseline_all,axis=0)
H_baseline_avg = np.mean(H_baseline_all,axis=0)

ax10.plot(W_preictal_avg,color='black')
ax20.plot(H_preictal_avg,color='black')

ax30.plot(W_baseline_avg,color='black')
ax40.plot(H_baseline_avg,color='black')

ax10.set_ylim([int(min(np.hstack((W_baseline_avg,W_preictal_avg)))-10), int(max(np.hstack((W_baseline_avg,W_preictal_avg)))+10)])
ax20.set_ylim([min(np.hstack((H_baseline_avg,H_preictal_avg)))-0.01, max(np.hstack((H_baseline_avg,H_preictal_avg)))+0.01])
ax30.set_ylim([int(min(np.hstack((W_baseline_avg,W_preictal_avg)))-10), int(max(np.hstack((W_baseline_avg,W_preictal_avg)))+10)])
ax40.set_ylim([min(np.hstack((H_baseline_avg,H_preictal_avg)))-0.01, max(np.hstack((H_baseline_avg,H_preictal_avg)))+0.01])

fig1.subplots_adjust(left=0.08, bottom=0.10, right=0.91, top=0.90, wspace=0.14, hspace=0.21)
fig1.show()
