
import pickle, json
import numpy as np

path = "/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/classification_and_prediction/results"

def specificity(true_negatives,false_positives):
    specificity = true_negatives/(true_negatives+false_positives)

    return specificity

def positive_predictive_value(true_positives,false_positives):
    positive_predictive_value = true_positives/(true_positives+false_positives)

    return positive_predictive_value

def negative_predictive_value(true_negatives,false_negatives):
    negative_predictive_value = true_negatives/(true_negatives+false_negatives)

    return negative_predictive_value

with open(path+"/50-50-merged/11502/all_results_11502.txt") as f1:
    balanced_11502 = json.load(f1)

with open(path+"/50-50-merged/25302_2/all_results_25302_2.txt") as f2:
    balanced_25302 = json.load(f2)

with open(path+"/50-50-merged/59002_2/all_results_59002_2.txt") as f3:
    balanced_59002 = json.load(f3)

with open(path+"/50-50-merged/62002_2/all_results_62002_2.txt") as f4:
    balanced_62002 = json.load(f4)

with open(path+"/50-50-merged/97002_3/all_results_97002_3.txt") as f5:
    balanced_97002 = json.load(f5)

with open(path+"/50-50-merged/109602/all_results_109602.txt") as f6:
    balanced_109602 = json.load(f6)

with open(path+"/60-40-merged/11502/all_results_11502.txt") as f7:
    imbalanced_11502 = json.load(f7)

with open(path+"/60-40-merged/25302_2/all_results_25302_2.txt") as f8:
    imbalanced_25302 = json.load(f8)

with open(path+"/60-40-merged/59002_2/all_results_59002_2.txt") as f9:
    imbalanced_59002 = json.load(f9)

with open(path+"/60-40-merged/62002_2/all_results_62002_2.txt") as f10:
    imbalanced_62002 = json.load(f10)

with open(path+"/60-40-merged/97002_3/all_results_97002_3.txt") as f11:
    imbalanced_97002 = json.load(f11)

with open(path+"/60-40-merged/109602/all_results_109602.txt") as f12:
    imbalanced_109602 = json.load(f12)

# 50-50 merged set
# confusion matrix svm: 11502
tp_svm_11502_balanced = np.round(balanced_11502[1]["avg_true_positives_svm"],decimals=2)
fp_svm_11502_balanced = np.round(balanced_11502[1]["avg_false_positives_svm"],decimals=2)
fn_svm_11502_balanced = np.round(balanced_11502[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_11502_balanced = np.round(balanced_11502[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 11502
tp_rdf_11502_balanced = np.round(balanced_11502[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_11502_balanced = np.round(balanced_11502[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_11502_balanced = np.round(balanced_11502[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_11502_balanced = np.round(balanced_11502[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 25302
tp_svm_25302_balanced = np.round(balanced_25302[1]["avg_true_positives_svm"],decimals=2)
fp_svm_25302_balanced = np.round(balanced_25302[1]["avg_false_positives_svm"],decimals=2)
fn_svm_25302_balanced = np.round(balanced_25302[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_25302_balanced = np.round(balanced_25302[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 25302
tp_rdf_25302_balanced = np.round(balanced_25302[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_25302_balanced = np.round(balanced_25302[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_25302_balanced = np.round(balanced_25302[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_25302_balanced = np.round(balanced_25302[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 59002
tp_svm_59002_balanced = np.round(balanced_59002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_59002_balanced = np.round(balanced_59002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_59002_balanced = np.round(balanced_59002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_59002_balanced = np.round(balanced_59002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 59002
tp_rdf_59002_balanced = np.round(balanced_59002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_59002_balanced = np.round(balanced_59002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_59002_balanced = np.round(balanced_59002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_59002_balanced = np.round(balanced_59002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 62002
tp_svm_62002_balanced = np.round(balanced_62002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_62002_balanced = np.round(balanced_62002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_62002_balanced = np.round(balanced_62002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_62002_balanced = np.round(balanced_62002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 62002
tp_rdf_62002_balanced = np.round(balanced_62002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_62002_balanced = np.round(balanced_62002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_62002_balanced = np.round(balanced_62002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_62002_balanced = np.round(balanced_62002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 97002
tp_svm_97002_balanced = np.round(balanced_97002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_97002_balanced = np.round(balanced_97002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_97002_balanced = np.round(balanced_97002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_97002_balanced = np.round(balanced_97002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 97002
tp_rdf_97002_balanced = np.round(balanced_97002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_97002_balanced = np.round(balanced_97002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_97002_balanced = np.round(balanced_97002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_97002_balanced = np.round(balanced_97002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 109602
tp_svm_109602_balanced = np.round(balanced_109602[1]["avg_true_positives_svm"],decimals=2)
fp_svm_109602_balanced = np.round(balanced_109602[1]["avg_false_positives_svm"],decimals=2)
fn_svm_109602_balanced = np.round(balanced_109602[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_109602_balanced = np.round(balanced_109602[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 109602
tp_rdf_109602_balanced = np.round(balanced_109602[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_109602_balanced = np.round(balanced_109602[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_109602_balanced = np.round(balanced_109602[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_109602_balanced = np.round(balanced_109602[1]["avg_true_negatives_rdf"],decimals=2)

tp_svm_balanced = [tp_svm_11502_balanced,tp_svm_25302_balanced,tp_svm_59002_balanced,tp_svm_62002_balanced,tp_svm_97002_balanced,tp_svm_109602_balanced]
fp_svm_balanced = [fp_svm_11502_balanced,fp_svm_25302_balanced,fp_svm_59002_balanced,fp_svm_62002_balanced,fp_svm_97002_balanced,fp_svm_109602_balanced]
fn_svm_balanced = [fn_svm_11502_balanced,fn_svm_25302_balanced,fn_svm_59002_balanced,fn_svm_62002_balanced,fn_svm_97002_balanced,fn_svm_109602_balanced]
tn_svm_balanced = [tn_svm_11502_balanced,tn_svm_25302_balanced,tn_svm_59002_balanced,tn_svm_62002_balanced,tn_svm_97002_balanced,tn_svm_109602_balanced]

tp_rdf_balanced = [tp_rdf_11502_balanced,tp_rdf_25302_balanced,tp_rdf_59002_balanced,tp_rdf_62002_balanced,tp_rdf_97002_balanced,tp_rdf_109602_balanced]
fp_rdf_balanced = [fp_rdf_11502_balanced,fp_rdf_25302_balanced,fp_rdf_59002_balanced,fp_rdf_62002_balanced,fp_rdf_97002_balanced,fp_rdf_109602_balanced]
fn_rdf_balanced = [fn_rdf_11502_balanced,fn_rdf_25302_balanced,fn_rdf_59002_balanced,fn_rdf_62002_balanced,fn_rdf_97002_balanced,fn_rdf_109602_balanced]
tn_rdf_balanced = [tn_rdf_11502_balanced,tn_rdf_25302_balanced,tn_rdf_59002_balanced,tn_rdf_62002_balanced,tn_rdf_97002_balanced,tn_rdf_109602_balanced]

# 60-40 merged set
# confusion matrix svm: 11502
tp_svm_11502_imbalanced = np.round(imbalanced_11502[1]["avg_true_positives_svm"],decimals=2)
fp_svm_11502_imbalanced = np.round(imbalanced_11502[1]["avg_false_positives_svm"],decimals=2)
fn_svm_11502_imbalanced = np.round(imbalanced_11502[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_11502_imbalanced = np.round(imbalanced_11502[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 11502
tp_rdf_11502_imbalanced = np.round(imbalanced_11502[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_11502_imbalanced = np.round(imbalanced_11502[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_11502_imbalanced = np.round(imbalanced_11502[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_11502_imbalanced = np.round(imbalanced_11502[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 25302
tp_svm_25302_imbalanced = np.round(imbalanced_25302[1]["avg_true_positives_svm"],decimals=2)
fp_svm_25302_imbalanced = np.round(imbalanced_25302[1]["avg_false_positives_svm"],decimals=2)
fn_svm_25302_imbalanced = np.round(imbalanced_25302[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_25302_imbalanced = np.round(imbalanced_25302[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 25302
tp_rdf_25302_imbalanced = np.round(imbalanced_25302[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_25302_imbalanced = np.round(imbalanced_25302[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_25302_imbalanced = np.round(imbalanced_25302[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_25302_imbalanced = np.round(imbalanced_25302[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 59002
tp_svm_59002_imbalanced = np.round(imbalanced_59002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_59002_imbalanced = np.round(imbalanced_59002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_59002_imbalanced = np.round(imbalanced_59002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_59002_imbalanced = np.round(imbalanced_59002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 59002
tp_rdf_59002_imbalanced = np.round(imbalanced_59002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_59002_imbalanced = np.round(imbalanced_59002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_59002_imbalanced = np.round(imbalanced_59002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_59002_imbalanced = np.round(imbalanced_59002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 62002
tp_svm_62002_imbalanced = np.round(imbalanced_62002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_62002_imbalanced = np.round(imbalanced_62002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_62002_imbalanced = np.round(imbalanced_62002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_62002_imbalanced = np.round(imbalanced_62002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 62002
tp_rdf_62002_imbalanced = np.round(imbalanced_62002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_62002_imbalanced = np.round(imbalanced_62002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_62002_imbalanced = np.round(imbalanced_62002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_62002_imbalanced = np.round(imbalanced_62002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 97002
tp_svm_97002_imbalanced = np.round(imbalanced_97002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_97002_imbalanced = np.round(imbalanced_97002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_97002_imbalanced = np.round(imbalanced_97002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_97002_imbalanced = np.round(imbalanced_97002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 97002
tp_rdf_97002_imbalanced = np.round(imbalanced_97002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_97002_imbalanced = np.round(imbalanced_97002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_97002_imbalanced = np.round(imbalanced_97002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_97002_imbalanced = np.round(imbalanced_97002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 109602
tp_svm_109602_imbalanced = np.round(imbalanced_109602[1]["avg_true_positives_svm"],decimals=2)
fp_svm_109602_imbalanced = np.round(imbalanced_109602[1]["avg_false_positives_svm"],decimals=2)
fn_svm_109602_imbalanced = np.round(imbalanced_109602[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_109602_imbalanced = np.round(imbalanced_109602[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 109602
tp_rdf_109602_imbalanced = np.round(imbalanced_109602[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_109602_imbalanced = np.round(imbalanced_109602[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_109602_imbalanced = np.round(imbalanced_109602[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_109602_imbalanced = np.round(imbalanced_109602[1]["avg_true_negatives_rdf"],decimals=2)

tp_svm_imbalanced = [tp_svm_11502_imbalanced,tp_svm_25302_imbalanced,tp_svm_59002_imbalanced,tp_svm_62002_imbalanced,tp_svm_97002_imbalanced,tp_svm_109602_imbalanced]
fp_svm_imbalanced = [fp_svm_11502_imbalanced,fp_svm_25302_imbalanced,fp_svm_59002_imbalanced,fp_svm_62002_imbalanced,fp_svm_97002_imbalanced,fp_svm_109602_imbalanced]
fn_svm_imbalanced = [fn_svm_11502_imbalanced,fn_svm_25302_imbalanced,fn_svm_59002_imbalanced,fn_svm_62002_imbalanced,fn_svm_97002_imbalanced,fn_svm_109602_imbalanced]
tn_svm_imbalanced = [tn_svm_11502_imbalanced,tn_svm_25302_imbalanced,tn_svm_59002_imbalanced,tn_svm_62002_imbalanced,tn_svm_97002_imbalanced,tn_svm_109602_imbalanced]

tp_rdf_imbalanced = [tp_rdf_11502_imbalanced,tp_rdf_25302_imbalanced,tp_rdf_59002_imbalanced,tp_rdf_62002_imbalanced,tp_rdf_97002_imbalanced,tp_rdf_109602_imbalanced]
fp_rdf_imbalanced = [fp_rdf_11502_imbalanced,fp_rdf_25302_imbalanced,fp_rdf_59002_imbalanced,fp_rdf_62002_imbalanced,fp_rdf_97002_imbalanced,fp_rdf_109602_imbalanced]
fn_rdf_imbalanced = [fn_rdf_11502_imbalanced,fn_rdf_25302_imbalanced,fn_rdf_59002_imbalanced,fn_rdf_62002_imbalanced,fn_rdf_97002_imbalanced,fn_rdf_109602_imbalanced]
tn_rdf_imbalanced = [tn_rdf_11502_imbalanced,tn_rdf_25302_imbalanced,tn_rdf_59002_imbalanced,tn_rdf_62002_imbalanced,tn_rdf_97002_imbalanced,tn_rdf_109602_imbalanced]

specificities_svm_balanced = []
specificities_rdf_balanced = []

specificities_svm_imbalanced = []
specificities_rdf_imbalanced = []

positive_predictive_value_svm_balanced = []
positive_predictive_value_rdf_balanced = []

positive_predictive_value_svm_imbalanced = []
positive_predictive_value_rdf_imbalanced = []

negative_predictive_value_svm_balanced = []
negative_predictive_value_rdf_balanced = []

negative_predictive_value_svm_imbalanced = []
negative_predictive_value_rdf_imbalanced = []

for n in np.arange(0,6):
    specificities_svm_balanced.append(specificity(tn_svm_balanced[n],fp_svm_balanced[n]))
    specificities_rdf_balanced.append(specificity(tn_rdf_balanced[n],fp_rdf_balanced[n]))
    specificities_svm_imbalanced.append(specificity(tn_svm_imbalanced[n],fp_svm_imbalanced[n]))
    specificities_rdf_imbalanced.append(specificity(tn_rdf_imbalanced[n],fp_rdf_imbalanced[n]))

    positive_predictive_value_svm_balanced.append(positive_predictive_value(tp_svm_balanced[n],fp_svm_balanced[n]))
    positive_predictive_value_rdf_balanced.append(positive_predictive_value(tp_rdf_balanced[n],fp_rdf_balanced[n]))
    positive_predictive_value_svm_imbalanced.append(positive_predictive_value(tp_svm_imbalanced[n],fp_svm_imbalanced[n]))
    positive_predictive_value_rdf_imbalanced.append(positive_predictive_value(tp_rdf_imbalanced[n],fp_rdf_imbalanced[n]))

    negative_predictive_value_svm_balanced.append(negative_predictive_value(tn_svm_balanced[n],fn_svm_balanced[n]))
    negative_predictive_value_rdf_balanced.append(negative_predictive_value(tn_rdf_balanced[n],fn_rdf_balanced[n]))
    negative_predictive_value_svm_imbalanced.append(negative_predictive_value(tn_svm_imbalanced[n],fn_svm_imbalanced[n]))
    negative_predictive_value_rdf_imbalanced.append(negative_predictive_value(tn_rdf_imbalanced[n],fn_rdf_imbalanced[n]))

variables = dict([('specificities_svm_balanced',specificities_svm_balanced),("specificities_rdf_balanced",specificities_rdf_balanced)])
variables["specificities_svm_imbalanced"] = specificities_svm_imbalanced
variables["specificities_rdf_imbalanced"] = specificities_rdf_imbalanced
variables["positive_predictive_value_svm_balanced"] = positive_predictive_value_svm_balanced
variables["positive_predictive_value_rdf_balanced"] = positive_predictive_value_rdf_balanced
variables["positive_predictive_value_svm_imbalanced"] = positive_predictive_value_svm_imbalanced
variables["positive_predictive_value_rdf_imbalanced"] = positive_predictive_value_rdf_imbalanced
variables["negative_predictive_value_svm_balanced"] = negative_predictive_value_svm_balanced
variables["negative_predictive_value_rdf_balanced"] = negative_predictive_value_rdf_balanced
variables["negative_predictive_value_svm_imbalanced"] = negative_predictive_value_svm_imbalanced
variables["negative_predictive_value_rdf_imbalanced"] = negative_predictive_value_rdf_imbalanced

with open(path+"/variables_additional_calculations.pickle","wb") as f:
    pickle.dump(variables,f)
