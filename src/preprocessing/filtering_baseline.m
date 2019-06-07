
path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_25302_extracted_seizures/25301102/data_baseline_25302_2/in-sample/baseline1_train_25302_2';
%out-of-sample/baseline1_25302_2';

load(strcat(path,base_directory,'/Data_baseline_41_1st_half.mat'))

%% eye inspection
figure()
subplot(2,2,1)
plot(first_half(:,1:4) + repmat(1:4,size(first_half,1),1)*8000)
title('first 4 channels')

subplot(2,2,2)
plot(first_half(:,5:8) + repmat(1:4,size(first_half,1),1)*8000)
title('channels 5 to 8')

subplot(2,2,3)
plot(first_half(:,9:12) + repmat(1:4,size(first_half,1),1)*8000)
title('channels 9 to 12')

subplot(2,2,4)
plot(first_half(:,13:16) + repmat(1:4,size(first_half,1),1)*8000)
title('channels 13 to 16')


figure()
subplot(2,2,1)
plot(first_half(:,17:20) + repmat(1:4,size(first_half,1),1)*8000)
title('channels 17 to 20')

subplot(2,2,2)
plot(first_half(:,21:26) + repmat(1:6,size(first_half,1),1)*8000)
title('channels 21 to 26')


% lowpass filter
% case 1
fc = 1;
fs = 512;
fn = 512/2;
order = 7;

[b1,a1]  = butter(order,fc/fn,'low');
dataOut1 = filter(b1,a1,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut1)
legend('unfiltered signal', 'filtered signal')
title('Lowpass filter, cutoff frequency 1Hz, filter order 7')

% case 2
fc = 5;
order = 11;

[b2,a2]  = butter(order,fc/fn,'low');
dataOut2 = filter(b2,a2,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut2)
legend('unfiltered signal', 'filtered signal')
title('Lowpass filter, cutoff frequency 5Hz, filter order 11')

% case 3
fc = 35;
order = 10;

[b3,a3]  = butter(order,fc/fn,'low');
dataOut3 = filter(b3,a3,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut3)
legend('unfiltered signal', 'filtered signal')
title('Lowpass filter, cutoff frequency 35Hz, filter order 10')

% highpass filter
% case 4
fc = 0.03;
order = 2;

[b4,a4]  = butter(order,fc/fn,'high');
dataOut4 = filter(b4,a4,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut4)
legend('unfiltered signal', 'filtered signal')
title('Highpass filter, cutoff frequency 35Hz, filter order 10')

% case 5
fc = 50;
order = 10;

[b5,a5]  = butter(order,fc/fn,'high');
dataOut5 = filter(b5,a5,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut5)
legend('unfiltered signal', 'filtered signal')
title('Highpass filter, cutoff frequency 50Hz, filter order 10')

% case 6
fc = 50;
order = 32;

[b6,a6]  = butter(order,fc/fn,'high');
dataOut6 = filter(b6,a6,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut6)
legend('unfiltered signal', 'filtered signal')
title('Highpass filter, cutoff frequency 50Hz, filter order 32')

% bandpass filter
% case 7

d = designfilt('bandpassiir','FilterOrder',18,'HalfPowerFrequency1',5,'HalfPowerFrequency2',25,'SampleRate',512);
dataOut7=filter(d,first_half(:,1));
figure()
plot(first_half(:,1))
hold on
plot(dataOut7)
legend('unfiltered signal', 'filtered signal')
title('Bandpass filter, frequencies between 5 and 25Hz, filter order 10')

% bandstop filter
% case 8
for i = 1:size(first_half,2)
    d = designfilt('bandstopiir','FilterOrder',36,'HalfPowerFrequency1',0.05,'HalfPowerFrequency2',2,'SampleRate',512);
    dataOut8=filter(d,first_half(:,i));
    figure()
    plot(first_half(:,i))
    hold on
    plot(dataOut8)
    legend('unfiltered signal', 'filtered signal')
    title('Bandstop filter, cutoff frequencies between 55 and 75Hz, filter order 18')
end

%combination of bandpass and bandstop filter
%case 9

d = designfilt('bandpassiir','FilterOrder',36,'HalfPowerFrequency1',1,'HalfPowerFrequency2',55,'SampleRate',512);
dataOut7=filter(d,first_half(:,1));
figure()
plot(first_half(:,1))
hold on
plot(dataOut7)
legend('unfiltered signal', 'filtered signal')
title('Bandpass filter, cutoff frequencies between 5 and 25Hz, filter order 18')

d = designfilt('bandstopiir','FilterOrder',36,'HalfPowerFrequency1',1,'HalfPowerFrequency2',5,'SampleRate',512);
dataOut8=filter(d,dataOut7);
figure()
plot(first_half(:,1))
hold on
plot(dataOut8)
legend('unfiltered signal', 'filtered signal')
title('Bandstop filter, cutoff frequencies between 55 and 75Hz, filter order 18')


[b7,a7]  = butter(order,[fc1/fn, fc2/fn],'bandpass');
dataOut7 = filter(b7,a7,first_half(:,1));

figure()
plot(first_half(:,1))
hold on
plot(dataOut7)
legend('unfiltered signal', 'filtered signal')
title('Bandpass filter, cutoff frequencies between 5 and 75Hz, filter order 32')
