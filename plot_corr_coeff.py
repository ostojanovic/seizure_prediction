
from __future__ import unicode_literals
import os, matplotlib
import scipy.io as sio
import numpy as np
from matplotlib import rc
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Times New Roman"

"""
This script plots three figures of correlation coefficients:
    * 4 comparisons on a train set
    * 4 comparisons on a test set
    * preictal-preictal comparicson on a test set
"""

ident = '11502'
patient_id = '11502'
run_nr = '18'

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/'
coeff_directory = 'patient_'+ident+'_extracted_seizures/corr_coeff_'+patient_id+'/50-50/'+run_nr

################################################################### training set - data preparation #######################################################################################
files = os.listdir(path_directory+coeff_directory+'/train/')

dict_corr_matrix1 = sio.loadmat(path_directory+coeff_directory+'/train/'+files[0])
dict_corr_matrix2 = sio.loadmat(path_directory+coeff_directory+'/train/'+files[1])
dict_corr_matrix3 = sio.loadmat(path_directory+coeff_directory+'/train/'+files[2])
dict_corr_matrix4 = sio.loadmat(path_directory+coeff_directory+'/train/'+files[3])

AV_baseline_baseline = dict_corr_matrix1["corr_matrix_AV_baseline_baseline"]
AV_baseline_preictal = dict_corr_matrix2["corr_matrix_AV_baseline_preictal"]
AV_preictal_baseline = dict_corr_matrix3["corr_matrix_AV_preictal_baseline"]
AV_preictal_preictal = dict_corr_matrix4["corr_matrix_AV_preictal_preictal"]

###################################################################### evaluation set - data preparation ####################################################################################
files_test = os.listdir(path_directory+coeff_directory+'/test/')
files_out = os.listdir(path_directory+coeff_directory+'/out-of-sample/')

corr_dict_AV_baseline_baseline_test = sio.loadmat(path_directory+coeff_directory+'/test/'+files_test[0])
corr_dict_AV_baseline_preictal_test = sio.loadmat(path_directory+coeff_directory+'/test/'+files_test[1])
corr_dict_AV_preictal_baseline_test = sio.loadmat(path_directory+coeff_directory+'/test/'+files_test[2])
corr_dict_AV_preictal_preictal_test = sio.loadmat(path_directory+coeff_directory+'/test/'+files_test[3])

corr_dict_AV_baseline_baseline_out = sio.loadmat(path_directory+coeff_directory+'/out-of-sample/'+files_out[0])
corr_dict_AV_baseline_preictal_out = sio.loadmat(path_directory+coeff_directory+'/out-of-sample/'+files_out[1])
corr_dict_AV_preictal_baseline_out = sio.loadmat(path_directory+coeff_directory+'/out-of-sample/'+files_out[2])
corr_dict_AV_preictal_preictal_out = sio.loadmat(path_directory+coeff_directory+'/out-of-sample/'+files_out[3])

corr_matrix_AV_baseline_baseline_test = corr_dict_AV_baseline_baseline_test["corr_matrix_AV_baseline_baseline"]
corr_matrix_AV_baseline_preictal_test = corr_dict_AV_baseline_preictal_test["corr_matrix_AV_baseline_preictal"]
corr_matrix_AV_preictal_baseline_test = corr_dict_AV_preictal_baseline_test["corr_matrix_AV_preictal_baseline"]
corr_matrix_AV_preictal_preictal_test = corr_dict_AV_preictal_preictal_test["corr_matrix_AV_preictal_preictal"]

corr_matrix_AV_baseline_baseline_out = corr_dict_AV_baseline_baseline_out["corr_matrix_AV_baseline_baseline"]
corr_matrix_AV_baseline_preictal_out = corr_dict_AV_baseline_preictal_out["corr_matrix_AV_baseline_preictal"]
corr_matrix_AV_preictal_baseline_out = corr_dict_AV_preictal_baseline_out["corr_matrix_AV_preictal_baseline"]
corr_matrix_AV_preictal_preictal_out = corr_dict_AV_preictal_preictal_out["corr_matrix_AV_preictal_preictal"]

AV_baseline_baseline_eval = np.concatenate((corr_matrix_AV_baseline_baseline_test,corr_matrix_AV_baseline_baseline_out))
AV_baseline_preictal_eval = np.concatenate((corr_matrix_AV_baseline_preictal_test,corr_matrix_AV_baseline_preictal_out))
AV_preictal_baseline_eval = np.concatenate((corr_matrix_AV_preictal_baseline_test,corr_matrix_AV_preictal_baseline_out))
AV_preictal_preictal_eval = np.concatenate((corr_matrix_AV_preictal_preictal_test,corr_matrix_AV_preictal_preictal_out))

########################################################################### figure - train set ############################################################################################
fig,ax1 = plt.subplots(2, 2, figsize=(19.5,10.2))

img1 = ax1[0,0].imshow(AV_baseline_baseline,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_baseline_baseline)[1],0,np.shape(AV_baseline_baseline)[0]])
ax1[0,0].set_xlabel('Channel',fontsize=18)
ax1[0,0].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax1[0,0].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax1[0,0].set_ylabel('Individual measurement period',fontsize=18)
ax1[0,0].set_yticks(np.arange(0.5,np.shape(AV_baseline_baseline)[0]))
ax1[0,0].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_baseline_baseline)[0]+1)),fontsize=16)
ax1[0,0].set_title('Average interictal vs. individual interictal measurements',fontsize=18)

img3 = ax1[0,1].imshow(AV_baseline_preictal,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_baseline_preictal)[1],0,np.shape(AV_baseline_preictal)[0]])
ax1[0,1].set_xlabel('Channel',fontsize=18)
ax1[0,1].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax1[0,1].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax1[0,1].set_ylabel('Individual measurement period',fontsize=18)
ax1[0,1].set_yticks(np.arange(0.5,np.shape(AV_baseline_preictal)[0]))
ax1[0,1].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_baseline_preictal)[0]+1)),fontsize=16)
ax1[0,1].set_title('Average interictal vs. individual preictal measurements',fontsize=18)

img4 = ax1[1,0].imshow(AV_preictal_baseline,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_preictal_baseline)[1],0,np.shape(AV_preictal_baseline)[0]])
ax1[1,0].set_xlabel('Channel',fontsize=18)
ax1[1,0].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax1[1,0].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax1[1,0].set_ylabel('Individual measurement period',fontsize=18)
ax1[1,0].set_yticks(np.arange(0.5,np.shape(AV_preictal_baseline)[0]))
ax1[1,0].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_preictal_baseline)[0]+1)),fontsize=16)
ax1[1,0].set_title('Average preictal vs. individual interictal measurements',fontsize=18)

img2 = ax1[1,1].imshow(AV_preictal_preictal,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_preictal_preictal)[1],0,np.shape(AV_preictal_preictal)[0]])
ax1[1,1].set_xlabel('Channel',fontsize=18)
ax1[1,1].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax1[1,1].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax1[1,1].set_ylabel('Individual measurement period',fontsize=18)
ax1[1,1].set_yticks(np.arange(0.5,np.shape(AV_preictal_preictal)[0]))
ax1[1,1].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_preictal_preictal)[0]+1)),fontsize=16)
ax1[1,1].set_title('Average preictal vs. individual preictal measurements',fontsize=18)

fig.tight_layout()
fig.subplots_adjust(left=0.06, bottom=0.07, right=0.92, top=0.93, wspace=0.13,hspace=0.24)
colorbar1 = fig.colorbar(img1, ax=ax1.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar1.ax.tick_params(labelsize=16)
fig.text(0.95, 0.5, 'Correlation coefficient', va='center', rotation='vertical',fontsize=18)
fig.show()

########################################################################### figure - evaluation set ###########################################################################
fig2,ax2 = plt.subplots(2, 2, figsize=(19.5,10.2))

for tick in ax2[0,0].get_xticklabels():
    tick.set_visible(False)

for tick in ax2[0,1].get_xticklabels():
    tick.set_visible(False)

for tick in ax2[0,1].get_yticklabels():
    tick.set_visible(False)

for tick in ax2[1,1].get_yticklabels():
    tick.set_visible(False)

img5 = ax2[0,0].imshow(AV_baseline_baseline_eval,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_baseline_baseline_eval)[1],0,np.shape(AV_baseline_baseline_eval)[0]])
ax2[0,0].set_ylabel('Individual measurement period',fontsize=18)
ax2[0,0].set_yticks(np.arange(0.5,np.shape(AV_baseline_baseline_eval)[0]))
ax2[0,0].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_baseline_baseline_eval)[0]+1)),fontsize=16)
ax2[0,0].set_title('Average interictal vs. individual interictal measurements',fontsize=18)
fig2.text(0.04,0.55,r"$\textbf{A}$",fontsize=18)

img7 = ax2[0,1].imshow(AV_baseline_preictal_eval,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_baseline_preictal_eval)[1],0,np.shape(AV_baseline_preictal_eval)[0]])
ax2[0,1].set_yticks(np.arange(0.5,np.shape(AV_baseline_preictal_eval)[0]))
ax2[0,1].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_baseline_preictal_eval)[0]+1)),fontsize=16)
ax2[0,1].set_title('Average interictal vs. individual preictal measurements',fontsize=18)
fig2.text(0.46,0.55,r"$\textbf{B}$",fontsize=18)

img8 = ax2[1,0].imshow(AV_preictal_baseline_eval,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_preictal_baseline_eval)[1],0,np.shape(AV_preictal_baseline_eval)[0]])
ax2[1,0].set_xlabel('Channel',fontsize=18)
ax2[1,0].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax2[1,0].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax2[1,0].set_ylabel('Individual measurement period',fontsize=18)
ax2[1,0].set_yticks(np.arange(0.5,np.shape(AV_preictal_baseline_eval)[0]))
ax2[1,0].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_preictal_baseline_eval)[0]+1)),fontsize=16)
ax2[1,0].set_title('Average preictal vs. individual interictal measurements',fontsize=18)
fig2.text(0.04,0.07,r"$\textbf{C}$",fontsize=18)

img6 = ax2[1,1].imshow(AV_preictal_preictal_eval,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_preictal_preictal_eval)[1],0,np.shape(AV_preictal_preictal_eval)[0]])
ax2[1,1].set_xlabel('Channel',fontsize=18)
ax2[1,1].set_xticks([0,5,10,15,20,25,30,35,40,45])
ax2[1,1].set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=16)
ax2[1,1].set_yticks(np.arange(0.5,np.shape(AV_preictal_preictal_eval)[0]))
ax2[1,1].set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_preictal_preictal_eval)[0]+1)),fontsize=16)
ax2[1,1].set_title('Average preictal vs. individual preictal measurements',fontsize=18)
fig2.text(0.46,0.07,r"$\textbf{D}$",fontsize=18)

fig2.tight_layout()
fig2.subplots_adjust(left=0.06, bottom=0.07, right=0.92, top=0.93, wspace=0.13,hspace=0.24)
colorbar2 = fig2.colorbar(img5, ax=ax2.ravel().tolist(),fraction=0.046, pad=0.04)
colorbar2.ax.tick_params(labelsize=16)
fig2.text(0.95, 0.5, 'Correlation coefficient', va='center', rotation='vertical',fontsize=18)
fig2.show()

########################################################################### figure - preictal vs. preictal ###########################################################################
fig3, ax3 = plt.subplots(1, 1, figsize=(19.5,10.2))

img7 = ax3.imshow(AV_preictal_preictal_eval,cmap='RdBu_r',aspect='auto',vmin=-1, vmax=1, extent=[0,np.shape(AV_preictal_preictal_eval)[1],0,np.shape(AV_preictal_preictal_eval)[0]])
ax3.set_xlabel('Channel',fontsize=24)
ax3.set_xticks([0,5,10,15,20,25,30,35,40,45])
ax3.set_xticklabels([0,5,10,15,20,25,30,35,40,45],fontsize=18)
ax3.set_ylabel('Individual measurement period',fontsize=24)
ax3.set_yticks(np.arange(0.5,np.shape(AV_preictal_preictal_eval)[0]))
ax3.set_yticklabels(np.char.mod("%d", np.arange(1,np.shape(AV_preictal_preictal_eval)[0]+1)),fontsize=22)
ax3.set_title('Average preictal vs. individual preictal measurements',fontsize=28)

fig3.tight_layout()
fig3.subplots_adjust(left=0.06, bottom=0.07, right=0.92, top=0.93, wspace=0.13,hspace=0.24)
colorbar3 = fig3.colorbar(img7, ax=ax3,fraction=0.046, pad=0.04)
colorbar3.ax.tick_params(labelsize=22)
fig3.text(0.95, 0.5, 'Correlation coefficient', va='center', rotation='vertical',fontsize=22)
fig3.show()
