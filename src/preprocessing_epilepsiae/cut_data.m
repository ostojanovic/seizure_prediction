
path = ''; % path goes here

load(strcat(path,'/'))  % filename goes here

data = double(Data_windows.clinical{1,1}.Data_for_channel_Set{1,1})';

%% eye inspection

len = size(data,1);
pstart = len/4;
pend   = 3*len/4;

inspection_matrix = data(:,1:end);

figure()
subplot(2,2,1)
plot(inspection_matrix(pstart:pend,1:14) + repmat(0:13,pend-pstart+1,1)*8000)
title('first 14 channels')

subplot(2,2,2)
plot(inspection_matrix(pstart:pend,15:end) + repmat(14:size(inspection_matrix,2)-1,pend-pstart+1,1)*8000)
title('channels 15 to 30-ish')

%% cutting in two halves

f_sample     = 512;
offset_sec   = 5;
where_to_cut = len/2-f_sample*offset_sec;

first_half = inspection_matrix(1:where_to_cut-1,:);
second_half = inspection_matrix(where_to_cut:end,:);

pstart_half = round(size(first_half,1)/4);
pend_half   = round(3*size(first_half,1)/4);

%% eye inspection again

figure()
plot(first_half(pstart_half:pend_half,:) + repmat(1:size(first_half,2),pend_half-pstart_half+1,1)*8000)
title('first half')

figure()
plot(second_half(pstart_half:pend_half,:) + repmat(0:size(second_half,2)-1,pend_half-pstart_half+1,1)*8000)
title('second half')

%% saving
savename_1 = strcat(path,base_directory,'/');   % new filename goes here
savename_2 = strcat(path,base_directory,'/');   % new filename goes here
save(savename_1,'first_half','electrode_sets','selected_seizures');
save(savename_2,'second_half','electrode_sets','selected_seizures');

close all
