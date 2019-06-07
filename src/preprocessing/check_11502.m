
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_11502_extracted_seizures/data_clinical_11502';

load(strcat(path,base_directory,'/Data_clinical_26.mat'))
% data subclinical: 17 22 30 42 61 64 67 95 100 101 114 121 177 196 224 228
% 257 292 296 297

try
    data = double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,1})';
    data = [data,double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,2})',double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,3})'];
catch
    data = double(Data_windows.clinical{1,1})';

end

len = size(data,1);
pstart = len/4;
pend   = 3*len/4;

temp            = data(:,1:58);
std_sig         = std(temp);
cut_norm        = prctile(std_sig,75)*2;
norm_high_IDX   = find(std_sig<=cut_norm);
cut_norm        = prctile(std_sig,25)/2;
norm_low_IDX    = find(std_sig>=cut_norm);

consider_for_reref = intersect(norm_high_IDX,norm_low_IDX );
av_signal          = mean(temp(:,consider_for_reref),2);

figure
subplot(2,2,1)
plot(temp(pstart:pend,1:20) + repmat(0:19,pend-pstart+1,1)*8000)
subplot(2,2,2)
plot(temp(pstart:pend,21:40) + repmat(0:19,pend-pstart+1,1)*8000)

subplot(2,2,2)
plot(data(pstart:pend,21:43) + repmat(20:42,pend-pstart+1,1)*8000)

subplot(2,2,3)
plot(data(pstart:pend,44:66) + repmat(43:65,pend-pstart+1,1)*8000)

subplot(2,2,4)
plot(data(pstart:pend,67:83) + repmat(66:82,pend-pstart+1,1)*8000)
