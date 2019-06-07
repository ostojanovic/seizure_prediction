
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_25302_extracted_seizures/25301102/data_clinical_25302_2/out-of-sample/preictal_25302_2';

%in-sample/preictal_test_25302_2';

%data_baseline_25302_2/out-of-sample/baseline1_25302_2';
%in-sample/baseline1_test_25302_2';

dir_baseline = dir(strcat(path_directory,base_directory,'/','*mat'));
num_files = size(dir_baseline,1);

for i = 1:num_files

    load(strcat(path_directory,base_directory,'/',dir_baseline(i).name))
    dir_baseline(i).name

    if  ~ismember(electrode_sets.names,'HRC5')

            electrode_sets.names(13:14) = [];     % 'HRC1', 'HRC2', 'HRC4'
            electrode_sets.names(11) = [];

            first_half(:,13:14) = [];
            first_half(:,11) = [];

    elseif ~ismember(electrode_sets.names,'HRC2')

            electrode_sets.names(14) = [];        % 'HRC1', 'HRC4', 'HRC5'
            electrode_sets.names(11:12) = [];

            first_half(:,14) = [];
            first_half(:,11:12) = [];

    elseif ~ismember(electrode_sets.names,'HRC1')

            electrode_sets.names(14) = [];        % 'HRC2', 'HRC4', 'HRC5'
            electrode_sets.names(11:12) = [];

            first_half(:,14) = [];
            first_half(:,11:12) = [];
    end

    disp('everything ok')

    if size(electrode_sets.names,2) == 26 && size(first_half,2) == 26
        savename = (strcat(path_directory,base_directory,'/',dir_baseline(i).name));
        save(savename, 'electrode_sets', 'first_half', 'selected_seizures');
        disp('everything ok')
    else
        disp('something is not right. check everything again.')
    end
end
