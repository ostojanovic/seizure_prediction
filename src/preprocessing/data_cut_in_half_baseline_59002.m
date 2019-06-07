
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_59002_extracted_seizures/59001102/data_baseline_59002_2';

load(strcat(path,base_directory,'/Data_baseline_125.mat'))

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
subplot(2,2,1)
plot(temp(pstart:pend,1:25) + repmat(0:24,pend-pstart+1,1)*8000)
title('first 25 channels')

subplot(2,2,2)
plot(temp(pstart:pend,26:50) + repmat(25:49,pend-pstart+1,1)*8000)
title('channels 26 to 50')

subplot(2,2,3)
plot(temp(pstart:pend,51:75) + repmat(50:74,pend-pstart+1,1)*8000)
title('channels 51 to 75')

subplot(2,2,4)
plot(temp(pstart:pend,76:end) + repmat(75:size(temp,2)-1,pend-pstart+1,1)*8000)
title('channels 76 to 99-ish')

temp(:,99) = [];
% temp(:,20) = [];
temp(:,13) = [];
% temp(:,5)  = [];

% electrode_sets{2}.names = [];
% electrode_sets{1,1}.names(20) = [];
electrode_sets{1,1}.names(13) = [];
% electrode_sets{1,1}.names(5) = [];

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

savename_1 = strcat(path,base_directory,'/Data_baseline_125_1st_half.mat');
savename_2 = strcat(path,base_directory,'/Data_baseline_125_2nd_half.mat');
save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
