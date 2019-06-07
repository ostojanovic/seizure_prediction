
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_62002_extracted_seizures/62001102/data_clinical_62002_2';

load(strcat(path,base_directory,'/Data_clinical_7.mat'))

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
plot(temp(pstart:pend,1:10) + repmat(0:9,pend-pstart+1,1)*8000)
title('first 10 channels')

subplot(2,2,2)
plot(temp(pstart:pend,11:21) + repmat(10:20,pend-pstart+1,1)*8000)
title('channels 11 to 21')

subplot(2,2,3)
plot(temp(pstart:pend,22:33) + repmat(21:32,pend-pstart+1,1)*8000)
title('channels 22 to 33')

subplot(2,2,4)
plot(temp(pstart:pend,34:end) + repmat(33:size(temp,2)-1,pend-pstart+1,1)*8000)
title('channels 34 to 38-ish')

% temp(:,39:end)=[];
% temp(:,99) = [];
% temp(:,16) = [];

% electrode_sets{2}.names = [];
% electrode_sets{1,1}.names(16)=[];

%% cutting in two halves

f_sample     = 1024;
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

savename_1 = strcat(path,base_directory,'/Data_preictal_7.mat');
savename_2 = strcat(path,base_directory,'/Data_ictal_7.mat');
save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
