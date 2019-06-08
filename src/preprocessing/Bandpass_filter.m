function [ output_data] = Bandpass_filter(input_data,Properties)

"This script performs filtering of a signal with a bandpass filer. Written by: Gordon Pipa."

num_bands = size(Properties.Band,2);
for IDX_Band = 1:num_bands
    low_cutoff  = Properties.Band{IDX_Band}(1);
    high_cutoff = Properties.Band{IDX_Band}(2);
    if low_cutoff>0
        [b_high,a_high] = butter(5,2*low_cutoff./Properties.Fs, 'high');
        input_data = filter(b_high,a_high,input_data);
    end
    if high_cutoff>0
        [b_high,a_high] = butter(5,2*high_cutoff./Properties.Fs, 'low');
        input_data = filter(b_high,a_high,input_data);
    end
    output_data{IDX_Band}.Band_filtered = input_data;
    output_data{IDX_Band}.low_cutoff  = low_cutoff ;
    output_data{IDX_Band}.high_cutoff  = high_cutoff ;
    Percentage_ok = 1-mean(input_data(:)==NaN);
    output_data{IDX_Band}.Percentage_ok = 100*Percentage_ok;
 end
end
