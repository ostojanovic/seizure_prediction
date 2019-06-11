
import pickle
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

patient_id = "11502" #["11502", "25302", "59002", "62002", "97002", "109602"]
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_'+patient_id+'_extracted_seizures/'
num_channels = {"11502":48, "25302": 26, "59002": 94, "62002": 38, "97002": 91, "109602": 68}
coefficients = np.zeros((100,num_channels[patient_id],12))

for idx in range(100):
    with open(path+'prediction_models/smote_'+patient_id+"_"+str(idx)+".pickle", "rb") as f:
        file = pickle.load(f)
    coefficients[idx, :, : ] = file.coef_[0].reshape((num_channels[patient_id],12))
coefficients = coefficients.mean(axis=0)

fig = plt.figure(figsize=(10,6))
gr = gridspec.GridSpec(nrows=1, ncols=1)
fig.subplots_adjust(left=0.1, bottom=0.14, right=0.84, top=0.86, wspace=0.17, hspace=0.27)

ax = fig.add_subplot(gr[0])
img = plt.spy(coefficients, aspect="auto", cmap='Reds', vmin=0, vmax=1)

ax.set_title('SVM coefficients for regularization',fontsize=22)
ax.set_xlabel('NMF parameters',fontsize=22)
ax.set_ylabel('Channels',fontsize=22)

ax.set_xticks([0,3,7,11])
ax.set_xticklabels([1,4,8,12],fontsize=22)
ax.set_yticks([0,10,20,30,40])
ax.set_yticklabels([1,10,20,30,40],fontsize=22)
ax.tick_params(axis='both', labelsize=18, top=False, labeltop=False, labelbottom=True, length=8)

cbaxes = fig.add_axes([0.86, 0.14, 0.04, 0.72])
cb = plt.colorbar(img, ax=ax, cax=cbaxes)
cbaxes.tick_params(length=8, labelsize=22)

# fig.savefig("figures/svm_coefficients.pdf", pad_inches=0.4)

plt.show()
