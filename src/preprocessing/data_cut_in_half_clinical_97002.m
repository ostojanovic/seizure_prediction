
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_97002_extracted_seizures/97002102/data_clinical_97002_3';

load(strcat(path,base_directory,'/Data_clinical_19.mat'))

try
    data = double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,1})';
    %data = [data,double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,2})',double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,3})'];
catch
    data = double(Data_windows.clinical{1,1})';
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
plot(temp(pstart:pend,26:51) + repmat(25:50,pend-pstart+1,1)*8000)
title('channels 26 to 51')

subplot(2,2,3)
plot(temp(pstart:pend,52:77) + repmat(51:76,pend-pstart+1,1)*8000)
title('channels 52 to 77')

subplot(2,2,4)
plot(temp(pstart:pend,78:end) + repmat(77:size(temp,2)-1,pend-pstart+1,1)*8000)
title('channels 78 to 98-ish')

temp(:,99:end)=[];
% temp(:,99) = [];
temp(:,15) = [];

% electrode_sets{2}.names = [];
electrode_sets{1,1}.names(15)=[];

%% cutting in two halves

f_sample     = 256;
offset_sec   = 5;
where_to_cut = len/2-f_sample*offset_sec;

first_half = temp(1:where_to_cut-1,:);
second_half = temp(where_to_cut:end,:);

pstart_half = round(size(first_half,1)/4);
pend_half   = round(3*size(first_half,1)/4);

figure()
plot(first_half(pstart_half:pend_half,:) + repmat(1:size(first_half,2),pend_half-pstart_half+1,1)*8000)
title('first half')

figure()
plot(second_half(pstart_half:pend_half,:) + repmat(0:size(second_half,2)-1,pend_half-pstart_half+1,1)*8000)
title('second half')

%%

electrode_sets = electrode_sets{1,1};
electrode_sets.names = electrode_sets.names

savename_1 = strcat(path,base_directory,'/Data_preictal_19.mat');
savename_2 = strcat(path,base_directory,'/Data_ictal_19.mat');
save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
