
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_59002_extracted_seizures/59001102/data_clinical_59002_2/out-of-sample/preictal_59002_2';
%in-sample/preictal_test_59002_2';

%data_baseline_59002_2/out-of-sample/baseline1_59002_2';
%in-sample/baseline1_test_59002_2';

dir_baseline = dir(strcat(path_directory,base_directory,'/','*mat'));
num_files = size(dir_baseline,1);

for i = 1:num_files

    load(strcat(path_directory,base_directory,'/',dir_baseline(i).name))
    dir_baseline(i).name

    if find(ismember(electrode_sets.names,'GB8')==1) & find(ismember(electrode_sets.names,'GB5')==1)

        electrode_sets.names(19) = [];      % 'GB5'
        electrode_sets.names(13) = [];      % 'GB8'
        electrode_sets.names(5) = [];       % 'GA1'

        first_half(:,19) = [];              % 'GB5'
        first_half(:,13) = [];              % 'GB8'
        first_half(:,5) = [];               % 'GA1'

    elseif find(ismember(electrode_sets.names,'GC6')==1) & find(ismember(electrode_sets.names,'GB5')==1)

        electrode_sets.names(19) = [];      % 'GB5'
        electrode_sets.names(15) = [];      % 'GB8'
        electrode_sets.names(5) = [];       % 'GA1'

        first_half(:,19) = [];              % 'GB5'
        first_half(:,15) = [];              % 'GB8'
        first_half(:,5) = [];               % 'GA1'

    elseif find(ismember(electrode_sets.names,'GB5')==1) & size(first_half,2)==97

        electrode_sets.names(19) = [];      % 'GB5'
        electrode_sets.names(15) = [];      % 'GC6'
        electrode_sets.names(5) = [];       % 'GA1'

        first_half(:,19) = [];
        first_half(:,15) = [];
        first_half(:,5) = [];

    elseif find(ismember(electrode_sets.names,'GB5')==1) & size(first_half,2)==96

        electrode_sets.names(18) = [];      % 'GB5'
        electrode_sets.names(14) = [];      % 'GC6'

        first_half(:,18) = [];
        first_half(:,14) = [];

    elseif find(ismember(electrode_sets.names,'GB8')==1) & size(first_half,2)==97

        electrode_sets.names(16) = [];      % 'GC6'
        electrode_sets.names(13) = [];      % 'GB8'
        electrode_sets.names(5) = [];       % 'GA1'

        first_half(:,16) = [];
        first_half(:,13) = [];
        first_half(:,5) = [];
    end

    disp('everything ok')
    if size(electrode_sets.names,2) == 94 && size(first_half,2) == 94
        savename = (strcat(path_directory,base_directory,'/',dir_baseline(i).name));
        save(savename, 'electrode_sets', 'first_half', 'selected_seizures');
        disp('everything ok')
    else
        disp('something is not right. check everything again.')
    end
end
