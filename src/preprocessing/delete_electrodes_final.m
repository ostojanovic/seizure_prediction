
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_11502_extracted_seizures/data_clinical_11502/in-sample/preictal_train_11502';

%out-of-sample/preictal_11502';
%data_baseline_11502/out-of-sample/baseline1_11502';
%in-sample/baseline1_test_11502';

dir_baseline = dir(strcat(path_directory,base_directory,'/','*mat'));
num_files = size(dir_baseline,1);

for i = 1:num_files

    load(strcat(path_directory,base_directory,'/',dir_baseline(i).name))

    if size(electrode_sets.names,2) == 54

        electrode_sets.names(48:49) = [];       % 'G10', 'G11'
        first_half(:,48:49) = [];

        electrode_sets.names(36) = [];          % 'TBA4'
        first_half(:,36) = [];

        electrode_sets.names(22) = [];          % 'FR1'
        first_half(:,22) = [];

        electrode_sets.names(18) = [];          % 'IH5'
        first_half(:,18) = [];

        electrode_sets.names(13) = [];          % 'TO4'
        first_half(:,13) = [];

        disp('everything ok')
    else
        disp('size less than 54. do it manually')
    end

    if size(electrode_sets.names,2) == 48 && size(first_half,2) == 48
        savename = (strcat(path_directory,base_directory,'/',dir_baseline(i).name));
        save(savename, 'electrode_sets', 'first_half', 'selected_seizures');
        disp('everything ok')
    else
        disp('something is not right. check everything again.')
    end
end
