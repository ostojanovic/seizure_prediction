
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_25302_extracted_seizures/25301102/data_baseline_25302_2';

load(strcat(path,base_directory,'/Data_baseline_60.mat'))

try
    data = double(Data_windows.baseline{1,1}.Data_for_channel_Set{1,1})';
    %data = [data,double(Data_windows.baseline{1,1}.Data_for_channel_Set{1,2})',double(Data_windows.baseline{1,1}.Data_for_channel_Set{1,3})'];
catch
    data = double(Data_windows.baseline{1,1})';

end

%% eye inspection

len = size(data,1);
pstart = len/4;
pend   = 3*len/4;

temp = data(:,1:end);

figure()
plot(temp(pstart:pend,1:15) + repmat(0:14,pend-pstart+1,1)*8000)
title('first 15 channels')

figure()
plot(temp(pstart:pend,16:end) + repmat(15:size(temp,2)-1,pend-pstart+1,1)*8000)
title('channels from 16 to the end')

% temp(:,31) = [];
% temp(:,11) = [];
%
% electrode_sets{2}.names = [];
% electrode_sets{1,1}.names(11) = [];

%% cutting in two halves

where_to_cut = len/2;

first_half = temp(1:where_to_cut,:);
second_half = temp(where_to_cut+1:end,:);

pstart_half = size(first_half,1)/4;
pend_half   = 3*size(first_half,1)/4;

figure()
plot(first_half(pstart_half:pend_half,:) + repmat(0:size(first_half,2)-1,pend_half-pstart_half+1,1)*8000)
title('first half')

figure()
plot(second_half(pstart_half:pend_half,:) + repmat(0:size(second_half,2)-1,pend_half-pstart_half+1,1)*8000)
title('second half')

%%

electrode_sets = electrode_sets{1,1};
electrode_sets.names = electrode_sets.names

savename_1 = strcat(path,base_directory,'/Data_baseline_60_1st_half.mat');
savename_2 = strcat(path,base_directory,'/Data_baseline_60_2nd_half.mat');

save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
