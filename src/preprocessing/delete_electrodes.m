
% this is an example how to delete electrodes from a dataset

path = '';  % path goes here

dir = dir(strcat(path,'/','*mat'));
num_files = size(dir,1);

for i = 1:num_files

    load(strcat(path,'/',dir(i).name))
    dir(i).name

    if ismember(electrode_sets.names,'GB7')==0 & size(first_half,2)==97

        electrode_sets.names(86) = [];      % 'TBC1'
        electrode_sets.names(69) = [];      % 'FBA2'
        electrode_sets.names(44) = [];      % 'GB5'
        electrode_sets.names(31) = [];      % 'GA6'
        electrode_sets.names(16) = [];      % 'GC2'
        electrode_sets.names(15) = [];      % 'GC1'

        first_half(:,86) = [];
        first_half(:,69) = [];
        first_half(:,44) = [];
        first_half(:,31) = [];
        first_half(:,16) = [];
        first_half(:,15) = [];

    elseif find(ismember(electrode_sets.names,'GC1')==1) & size(first_half,2)==97

        electrode_sets.names(86) = [];      % 'TBC1'
        electrode_sets.names(69) = [];      % 'FBA2'
        electrode_sets.names(44) = [];      % 'GB5'
        electrode_sets.names(31) = [];      % 'GA6'
        electrode_sets.names(16) = [];      % 'GC1'
        electrode_sets.names(15) = [];      % 'GB7'

        first_half(:,86) = [];
        first_half(:,69) = [];
        first_half(:,44) = [];
        first_half(:,31) = [];
        first_half(:,16) = [];
        first_half(:,15) = [];

    elseif find(ismember(electrode_sets.names,'GC2')==1) & size(first_half,2)==97

        electrode_sets.names(86) = [];      % 'TBC1'
        electrode_sets.names(69) = [];      % 'FBA2'
        electrode_sets.names(44) = [];      % 'GB5'
        electrode_sets.names(31) = [];      % 'GA6'
        electrode_sets.names(16) = [];      % 'GC2'
        electrode_sets.names(15) = [];      % 'GB7'

        first_half(:,86) = [];
        first_half(:,69) = [];
        first_half(:,44) = [];
        first_half(:,31) = [];
        first_half(:,16) = [];
        first_half(:,15) = [];

    elseif find(ismember(electrode_sets.names,'GC2')==1) & size(first_half,2)==92

        electrode_sets.names(15) = [];      % 'GC2'

        first_half(:,15) = [];
    end

    disp('everything ok')
    if size(electrode_sets.names,2) == 91 && size(first_half,2) == 91
        savename = (strcat(path,'/',dir(i).name));
        save(savename, 'electrode_sets', 'first_half', 'selected_seizures');
        disp('everything ok')
    else
        disp('something is not right. check everything again.')
    end
end
