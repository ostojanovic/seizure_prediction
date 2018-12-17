
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

with open(path+"/original-merged/11502/all_results_11502.txt") as f1:
    original_11502 = json.load(f1)

with open(path+"/original-merged/25302_2/all_results_25302_2.txt") as f2:
    original_25302 = json.load(f2)

with open(path+"/original-merged/59002_2/all_results_59002_2.txt") as f3:
    original_59002 = json.load(f3)

with open(path+"/original-merged/62002_2/all_results_62002_2.txt") as f4:
    original_62002 = json.load(f4)

with open(path+"/original-merged/97002_3/all_results_97002_3.txt") as f5:
    original_97002 = json.load(f5)

with open(path+"/original-merged/109602/all_results_109602.txt") as f6:
    original_109602 = json.load(f6)

# original merged set
# confusion matrix svm: 11502
tp_svm_11502_original = np.round(original_11502[1]["avg_true_positives_svm"],decimals=2)
fp_svm_11502_original = np.round(original_11502[1]["avg_false_positives_svm"],decimals=2)
fn_svm_11502_original = np.round(original_11502[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_11502_original = np.round(original_11502[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 11502
tp_rdf_11502_original = np.round(original_11502[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_11502_original = np.round(original_11502[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_11502_original = np.round(original_11502[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_11502_original = np.round(original_11502[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 25302
tp_svm_25302_original = np.round(original_25302[1]["avg_true_positives_svm"],decimals=2)
fp_svm_25302_original = np.round(original_25302[1]["avg_false_positives_svm"],decimals=2)
fn_svm_25302_original = np.round(original_25302[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_25302_original = np.round(original_25302[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 25302
tp_rdf_25302_original = np.round(original_25302[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_25302_original = np.round(original_25302[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_25302_original = np.round(original_25302[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_25302_original = np.round(original_25302[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 59002
tp_svm_59002_original = np.round(original_59002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_59002_original = np.round(original_59002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_59002_original = np.round(original_59002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_59002_original = np.round(original_59002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 59002
tp_rdf_59002_original = np.round(original_59002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_59002_original = np.round(original_59002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_59002_original = np.round(original_59002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_59002_original = np.round(original_59002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 62002
tp_svm_62002_original = np.round(original_62002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_62002_original = np.round(original_62002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_62002_original = np.round(original_62002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_62002_original = np.round(original_62002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 62002
tp_rdf_62002_original = np.round(original_62002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_62002_original = np.round(original_62002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_62002_original = np.round(original_62002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_62002_original = np.round(original_62002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 97002
tp_svm_97002_original = np.round(original_97002[1]["avg_true_positives_svm"],decimals=2)
fp_svm_97002_original = np.round(original_97002[1]["avg_false_positives_svm"],decimals=2)
fn_svm_97002_original = np.round(original_97002[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_97002_original = np.round(original_97002[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 97002
tp_rdf_97002_original = np.round(original_97002[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_97002_original = np.round(original_97002[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_97002_original = np.round(original_97002[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_97002_original = np.round(original_97002[1]["avg_true_negatives_rdf"],decimals=2)

# confusion matrix svm: 109602
tp_svm_109602_original = np.round(original_109602[1]["avg_true_positives_svm"],decimals=2)
fp_svm_109602_original = np.round(original_109602[1]["avg_false_positives_svm"],decimals=2)
fn_svm_109602_original = np.round(original_109602[1]["avg_false_negatives_svm"],decimals=2)
tn_svm_109602_original = np.round(original_109602[1]["avg_true_negatives_svm"],decimals=2)

# confusion matrix rdf: 109602
tp_rdf_109602_original = np.round(original_109602[1]["avg_true_positives_rdf"],decimals=2)
fp_rdf_109602_original = np.round(original_109602[1]["avg_false_positives_rdf"],decimals=2)
fn_rdf_109602_original = np.round(original_109602[1]["avg_false_negatives_rdf"],decimals=2)
tn_rdf_109602_original = np.round(original_109602[1]["avg_true_negatives_rdf"],decimals=2)

tp_svm_original = [tp_svm_11502_original,tp_svm_25302_original,tp_svm_59002_original,tp_svm_62002_original,tp_svm_97002_original,tp_svm_109602_original]
fp_svm_original = [fp_svm_11502_original,fp_svm_25302_original,fp_svm_59002_original,fp_svm_62002_original,fp_svm_97002_original,fp_svm_109602_original]
fn_svm_original = [fn_svm_11502_original,fn_svm_25302_original,fn_svm_59002_original,fn_svm_62002_original,fn_svm_97002_original,fn_svm_109602_original]
tn_svm_original = [tn_svm_11502_original,tn_svm_25302_original,tn_svm_59002_original,tn_svm_62002_original,tn_svm_97002_original,tn_svm_109602_original]

tp_rdf_original = [tp_rdf_11502_original,tp_rdf_25302_original,tp_rdf_59002_original,tp_rdf_62002_original,tp_rdf_97002_original,tp_rdf_109602_original]
fp_rdf_original = [fp_rdf_11502_original,fp_rdf_25302_original,fp_rdf_59002_original,fp_rdf_62002_original,fp_rdf_97002_original,fp_rdf_109602_original]
fn_rdf_original = [fn_rdf_11502_original,fn_rdf_25302_original,fn_rdf_59002_original,fn_rdf_62002_original,fn_rdf_97002_original,fn_rdf_109602_original]
tn_rdf_original = [tn_rdf_11502_original,tn_rdf_25302_original,tn_rdf_59002_original,tn_rdf_62002_original,tn_rdf_97002_original,tn_rdf_109602_original]

specificities_svm_original = []
specificities_rdf_original = []

positive_predictive_value_svm_original = []
positive_predictive_value_rdf_original = []

negative_predictive_value_svm_original = []
negative_predictive_value_rdf_original = []

for n in np.arange(0,6):
    specificities_svm_original.append(specificity(tn_svm_original[n],fp_svm_original[n]))
    specificities_rdf_original.append(specificity(tn_rdf_original[n],fp_rdf_original[n]))

    positive_predictive_value_svm_original.append(positive_predictive_value(tp_svm_original[n],fp_svm_original[n]))
    positive_predictive_value_rdf_original.append(positive_predictive_value(tp_rdf_original[n],fp_rdf_original[n]))

    negative_predictive_value_svm_original.append(negative_predictive_value(tn_svm_original[n],fn_svm_original[n]))
    negative_predictive_value_rdf_original.append(negative_predictive_value(tn_rdf_original[n],fn_rdf_original[n]))

variables = dict([('specificities_svm_original',specificities_svm_original),("specificities_rdf_original",specificities_rdf_original)])
variables["positive_predictive_value_svm_original"] = positive_predictive_value_svm_original
variables["positive_predictive_value_rdf_original"] = positive_predictive_value_rdf_original
variables["negative_predictive_value_svm_original"] = negative_predictive_value_svm_original
variables["negative_predictive_value_rdf_original"] = negative_predictive_value_rdf_original

with open(path+"/variables_additional_calculations_original_set.pickle","wb") as f:
    pickle.dump(variables,f)
