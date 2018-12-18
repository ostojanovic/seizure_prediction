
import pandas as pd
import numpy as np
import nimfa
from matplotlib import pyplot as plt

ident = '109602'
patient_id = '109602'
run_nr = '1'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
# preictal_directory = 'patient_'+ident+'_extracted_seizures/data_clinical_'+patient_id+'/models_preictal/'+run_nr+'/'
# baseline_directory = 'patient_'+ident+'_extracted_seizures/data_baseline_'+patient_id+'/models_baseline/'+run_nr+'/'

# put all of it in a for loop for [1:100]
data = # load SpecR here

# NMF
rank = 1
nmf = nimfa.Nmf(data, rank=rank, seed='random_vcol', max_iter=200,
                update='euclidean', objective='conn', conn_change=40,
                n_run=50, track_factor=True)
nmf_fit = nmf()

W = nmf_fit.basis()
H = nmf_fit.coef()

# plt.figure(figsize=(25,15))
# plt.subplots_adjust(top=0.89, bottom=0.1, left=0.06, right=0.95, hspace=0.27, wspace=0.24)
# plt.subplot(121)
# plt.plot(wavelength, W[:,0])
# plt.xlabel("Wavelength", fontsize=20)
# plt.title("NMF, first component", fontsize=22)
#
# plt.subplot(122)
# plt.plot(wavelength, W[:,1])
# plt.xlabel("Wavelength", fontsize=20)
# plt.title("NMF, second component", fontsize=22)
#
# plt.figure(figsize=(25,15))
# plt.plot(wavelength, W)
# plt.xlabel("Wavelength", fontsize=20)
# plt.title("NMF, both components", fontsize=22)
# plt.legend(["first", "second"], fontsize=16)

plt.figure(figsize=(25,15))
f1 = plt.imshow(W.T, aspect="auto", cmap='RdBu_r')
plt.title("NMF, both components", fontsize=22)
c1 = plt.axes([0.91, 0.11, 0.02, 0.77])
plt.colorbar(mappable = f1, cax = c1)

plt.figure(figsize=(25,15))
f2 = plt.imshow(nmf_fit.summary()["connectivity"], aspect="auto", cmap='RdBu_r')
plt.title("NMF connectivity matrix", fontsize=22)
# The connectivity matrix C is a symmetric matrix which shows the shared membership of the samples:
# entry C_ij is 1 iff sample i and sample j belong to the same cluster, 0 otherwise.
c2 = plt.axes([0.91, 0.11, 0.02, 0.77])
plt.colorbar(mappable = f2, cax = c2)
plt.show()

# save models
# savename_baseline = strcat(path_directory,baseline_directory,'models_baseline/',run_nr,'/',sample,'/Model_',savename_part{2});
# save(savename_baseline,'sample','run_nr','patient_id','W_baseline','H_baseline','W_model_baseline','H_model_baseline','W_parameters_baseline','H_parameters_baseline','Models_baseline')
