
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';

baseline_train = 'patient_59002_extracted_seizures/59000102/data_baseline_59002_1/in-sample/baseline1_train_59002_1/';
baseline_test = 'patient_59002_extracted_seizures/59000102/data_baseline_59002_1/in-sample/baseline1_test_59002_1/';
baseline_out = 'patient_59002_extracted_seizures/59000102/data_baseline_59002_1/out-of-sample/baseline1_59002_1/';

dir_b_train = dir(strcat(path_directory,baseline_train,'*mat'));
dir_b_test = dir(strcat(path_directory,baseline_test,'*mat'));
dir_b_out = dir(strcat(path_directory,baseline_out,'*mat'));

dir_baseline = [dir_b_train; dir_b_test; dir_b_out];
num_baseline = size(dir_baseline,1);

clinical_train = 'patient_59002_extracted_seizures/59000102/data_clinical_59002_1/in-sample/preictal_train_59002_1/';
clinical_test = 'patient_59002_extracted_seizures/59000102/data_clinical_59002_1/in-sample/preictal_test_59002_1/';
clinical_out = 'patient_59002_extracted_seizures/59000102/data_clinical_59002_1/out-of-sample/preictal_59002_1/';

dir_c_train = dir(strcat(path_directory,clinical_train,'*mat'));
dir_c_test = dir(strcat(path_directory,clinical_test,'*mat'));
dir_c_out = dir(strcat(path_directory,clinical_out,'*mat'));

dir_clinical = [dir_c_train; dir_c_test; dir_c_out];
num_clinical = size(dir_clinical,1);

load(strcat(path_directory,clinical_train,dir_clinical(1).name))
el_preictal1 = electrode_sets.names;

load(strcat(path_directory,clinical_train,dir_clinical(2).name))
el_preictal2 = electrode_sets.names;

load(strcat(path_directory,clinical_test,dir_clinical(3).name))
el_preictal3 = electrode_sets.names;

load(strcat(path_directory,clinical_out,dir_clinical(4).name))
el_preictal4 = electrode_sets.names;

% check 1
find(ismember(el_preictal1,el_preictal2)==0)
find(ismember(el_preictal1,el_preictal3)==0)
find(ismember(el_preictal1,el_preictal4)==0)
% check 2
find(ismember(el_preictal2,el_preictal1)==0)
find(ismember(el_preictal2,el_preictal3)==0)
find(ismember(el_preictal2,el_preictal4)==0)
% check 3
find(ismember(el_preictal3,el_preictal1)==0)
find(ismember(el_preictal3,el_preictal2)==0)
find(ismember(el_preictal3,el_preictal4)==0)
% check 4
find(ismember(el_preictal4,el_preictal1)==0)
find(ismember(el_preictal4,el_preictal2)==0)
find(ismember(el_preictal4,el_preictal3)==0)


for i = 1:num_baseline-1

    load(strcat(path_directory,baseline_train,dir_baseline(i).name))
    el1 = electrode_sets.names;

    load(strcat(path_directory,baseline_train,dir_baseline(i+1).name))
    el2 = electrode_sets.names;

    dir_baseline(i).name
    dir_baseline(i+1).name

    % check 1
    find(ismember(el1,el2)==0)
    find(ismember(el1,el_preictal1)==0)
    find(ismember(el1,el_preictal2)==0)
    find(ismember(el1,el_preictal3)==0)
    find(ismember(el1,el_preictal4)==0)
    % check 2
    find(ismember(el2,el1)==0)
    find(ismember(el2,el_preictal1)==0)
    find(ismember(el2,el_preictal2)==0)
    find(ismember(el2,el_preictal3)==0)
    find(ismember(el2,el_preictal4)==0)
    % check 3
    find(ismember(el_preictal1,el1)==0)
    find(ismember(el_preictal2,el1)==0)
    find(ismember(el_preictal3,el1)==0)
    find(ismember(el_preictal4,el1)==0)
    % check 4
    find(ismember(el_preictal1,el2)==0)
    find(ismember(el_preictal2,el2)==0)
    find(ismember(el_preictal3,el2)==0)
    find(ismember(el_preictal4,el2)==0)
end
