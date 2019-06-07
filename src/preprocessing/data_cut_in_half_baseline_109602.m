
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_109602_extracted_seizures/data_baseline_109602';

load(strcat(path,base_directory,'/Data_baseline_19.mat'))   % change the name down there!!!

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
plot(temp(round(pstart):round(pend),1:18) + repmat(0:17,round(pend-pstart+1),1)*8000)
title('first 18 channels')

figure()
plot(temp(round(pstart):round(pend),19:37) + repmat(18:36,round(pend-pstart+1),1)*8000)
title('channels 19 to 37')

figure()
plot(temp(round(pstart):round(pend),38:56) + repmat(37:55,round(pend-pstart+1),1)*8000)
title('channels 38 to 56')

figure()
plot(temp(round(pstart):round(pend),57:75) + repmat(56:74,round(pend-pstart+1),1)*8000)
title('channels 57 to 75')

figure()
plot(temp(round(pstart):round(pend),76:end) + repmat(75:size(temp,2)-1,round(pend-pstart+1),1)*8000)
title('channels 76 to 100')
% temp(:,76:end) = [];  % surface+special electrodes
temp(:,62) = [];        % 'HRA2'
temp(:,61) = [];        % 'HRA1'
temp(:,57) = [];        % 'HL1'
temp(:,33) = [];        % 'TP4'
temp(:,25) = [];        % 'GA7'
temp(:,12) = [];        % 'GD1'
temp(:,7) = [];         % 'GC3'

% electrode_sets{3}.names = [];      % surface+special electrodes
% electrode_sets{2}.names = [];
electrode_sets{1,1}.names(62) = [];  % 'HRA2'
electrode_sets{1,1}.names(61) = [];  % 'HRA1'
electrode_sets{1,1}.names(57) = [];  % 'HL1'
electrode_sets{1,1}.names(33) = [];  % 'TP4'
electrode_sets{1,1}.names(25) = [];  % 'GA7'
electrode_sets{1,1}.names(12) = [];  % 'GD1'
electrode_sets{1,1}.names(7) = [];   % 'GC3'

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

savename_1 = strcat(path,base_directory,'/Data_baseline_19_1st_half.mat');
savename_2 = strcat(path,base_directory,'/Data_baseline_19_2nd_half.mat');
save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
