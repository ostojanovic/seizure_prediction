
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_11502_extracted_seizures/data_clinical_11502';

load(strcat(path,base_directory,'/Data_clinical_18.mat'))

% data_clinical_what_to_do: 3 4 5 7 9 10 11 12 13 14

% data_clinical_reref: 1 2 6 8 15 16 17
%26

% baseline_50Hz_component: 22 26 27 28
% baseline_what_to_do: 51 88 (subclinical: 59 97)

% data_baseline_reref: 5 11 14 15 36 48 57 60-1 74-5 81 92-3 97* 99

try
    data = double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,1})';
    data = [data,double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,2})',double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,3})'];
catch
    data = double(Data_windows.clinical{1,1})';

end

len = size(data,1);
pstart = len/4;
pend   = 3*len/4;

figure()
subplot(2,2,1)
plot(data(pstart:pend,1:29) + repmat(0:28,pend-pstart+1,1)*8000)

subplot(2,2,2)
plot(data(pstart:pend,30:58) + repmat(29:57,pend-pstart+1,1)*8000)

subplot(2,2,3)
plot(data(pstart:pend,59:79) + repmat(58:78,pend-pstart+1,1)*8000)

subplot(2,2,4)
plot(data(pstart:pend,80:83) + repmat(79:82,pend-pstart+1,1)*8000)

%% reref 1:58 channels
temp            = data(:,1:58);
std_sig         = std(temp);
cut_norm        = prctile(std_sig,75)*2;
norm_high_IDX   = find(std_sig<=cut_norm);
cut_norm        = prctile(std_sig,25)/2;
norm_low_IDX    = find(std_sig>=cut_norm);

consider_for_reref = intersect(norm_high_IDX,norm_low_IDX );

%% eye inspection
figure()
plot(temp(pstart:pend,consider_for_reref(1:20)) + repmat(0:19,pend-pstart+1,1)*8000)
title('first 20 channels')

figure()
plot(temp(pstart:pend,consider_for_reref(21:end)) + repmat(20:length(consider_for_reref)-1,pend-pstart+1,1)*8000)
title('channels from 21 to the end')

%consider_for_reref(37) = [];
%consider_for_reref(21) = [];

%% reref
av_signal          = mean(temp(:,consider_for_reref),2);
temp               = temp - repmat(av_signal,1,size(temp,2));

ref_dat=temp(:,consider_for_reref);

figure()
subplot(2,2,1)
plot(ref_dat(pstart:pend,1:29) + repmat(0:28,pend-pstart+1,1)*8000)
title('first 29 channels after rereferencing')
subplot(2,2,2)
plot(ref_dat(pstart:pend,30:length(consider_for_reref)) + repmat(29:length(consider_for_reref)-1,pend-pstart+1,1)*8000)
title('second 29-ish channels after rereferencing')

electrode_sets = electrode_sets{1,1};
electrode_sets.names = electrode_sets.names(consider_for_reref)

%%
savename = strcat(path,base_directory,'/Data_clinical_18_reref.mat');
save(savename,'ref_dat','Data_Window','electrode_sets','selected_seizures');

close all
