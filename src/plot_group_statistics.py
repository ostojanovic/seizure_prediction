
import numpy as np
import pickle as pkl
import seaborn as sns
from collections import OrderedDict
import matplotlib
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib import gridspec, transforms
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

path_epilepsiae = "*" # path goes here
path_ecosystem = "*"  # path goes here

patients_epilepsiae = OrderedDict([]) # patient id's go here as keys and the number of channels as values
patients_ecosystem = OrderedDict([]) # patient id's go here as keys and the number of channels as values

with open(path_epilepsiae+"data/group_measures.pkl", "rb") as f:
    group_measures_epilepsiae = pkl.load(f)

with open(path_ecosystem+"data/group_measures.pkl", "rb") as f:
    group_measures_ecosystem = pkl.load(f)

fig = plt.figure(figsize=(12,8))
plt.subplots_adjust(right=0.92, left=0.12, top=0.93, bottom=0.08, hspace=0.23, wspace=0.2)

plt.subplot(221)

maximum_preictal_epilepsiae = []
for n in range(len(group_measures_epilepsiae["H_preictal_argmax_frequencies"])):
    for value in group_measures_epilepsiae["H_preictal_argmax_frequencies"][n]:
        maximum_preictal_epilepsiae.append(value)

sns.distplot(maximum_preictal_epilepsiae, color="blue", kde=False, norm_hist=True)
plt.title("Preictal states", fontsize=22)
plt.ylabel("EPILEPSIAE", fontsize=22)
plt.tick_params(axis='both', which='both', labelbottom=True, labelsize=18, length=8)
plt.xticks([0,100,200,300,400,500], [0,100,200,300,400,500])
plt.yticks([0.01, 0.02, 0.03, 0.04], [0.01, 0.02, 0.03, 0.04])
plt.ylim([0, 0.042])

plt.subplot(222)

maximum_interictal_epilepsiae = []
for n in range(len(group_measures_epilepsiae["H_interictal_argmax_frequencies"])):
    for value in group_measures_epilepsiae["H_interictal_argmax_frequencies"][n]:
        maximum_interictal_epilepsiae.append(value)

sns.distplot(maximum_interictal_epilepsiae, color="red", kde=False, norm_hist=True)
plt.tick_params(axis='both', which='both', labelsize=18, length=8)
plt.title("Interictal states", fontsize=22)
plt.xticks([0,100,200,300,400,500], [0,100,200,300,400,500])
plt.yticks([0.002, 0.004, 0.006, 0.008], [0.002, 0.004, 0.006, 0.008])
plt.ylim([0, 0.0085])

plt.subplot(223)

maximum_preictal_ecosystem = []
for n in range(len(group_measures_ecosystem["H_preictal_argmax_frequencies"])):
    for value in group_measures_ecosystem["H_preictal_argmax_frequencies"][n]:
        maximum_preictal_ecosystem.append(value)

sns.distplot(maximum_preictal_ecosystem, color="blue", kde=False, norm_hist=True)
plt.ylabel("Epilepsyecosystem", fontsize=22)
plt.tick_params(axis='both', which='both', labelbottom=True, labelsize=18, length=8)
plt.xticks([0,50,100,150,200], [0,50,100,150,200])
plt.yticks([0.005, 0.01], [0.005, 0.1])
plt.ylim([0, 0.015])

plt.subplot(224)

maximum_interictal_ecosystem = []
for n in range(len(group_measures_ecosystem["H_interictal_argmax_frequencies"])):
    for value in group_measures_ecosystem["H_interictal_argmax_frequencies"][n]:
        maximum_interictal_ecosystem.append(value)

sns.distplot(maximum_interictal_ecosystem, color="red", kde=False, norm_hist=True)
plt.tick_params(axis='both', which='both', labelsize=18, length=8)
plt.xticks([0,50,100,150,200], [0,50,100,150,200])
plt.yticks([0.01, 0.02], [0.01, 0.02])
plt.ylim([0, 0.025])

fig.text(0.09,0.55,r"$\textbf{A}$",fontsize=24)
fig.text(0.535,0.55,r"$\textbf{B}$",fontsize=24)
fig.text(0.09,0.08,r"$\textbf{C}$",fontsize=24)
fig.text(0.535,0.08,r"$\textbf{D}$",fontsize=24)

fig.savefig("../figures/group_statistics_maximum_frequency.pdf", pad_inches=0.4)
