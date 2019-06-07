function signals = extract_data(path,base_directory,filename,signals,force_write)
%load('ttt')
archstr = computer('arch');

% data from .data file
name = signals.header.file_path; %strcat(path,base_directory,'/',filename);
name(end-4:end) = '.data'
file = fopen(name,'rb');                   % Open file

data = int16(fread(file,[signals.header.num_channels signals.header.num_samples], 'int16'));
fclose(file);
clear file;                                                                    % Close file

%data = data.*signals.header.conversion_factor;                                % Multiply data with conversion factor
signals.data = data;
if size(data,1)~=signals.header.num_channels
    warning('header and data inconsistent')
end
if size(data,2)~=signals.header.num_samples
    warning('header and data inconsistent')
end

signals.header.computed_start_time = signals.header.hours*60*60+signals.header.minutes*60+signals.header.seconds;
signals.header.computed_stop_time  = signals.header.hours*60*60+signals.header.minutes*60+signals.header.seconds+signals.header.num_samples.*1/signals.header.sample_freq;
signals.header.File_name = filename;


if strcmp(archstr,'win64')
    %save(strcat(path,base_directory,'\mat_files\',strtok(dirstruct_data(i).name,'.header'),'_header','.mat'),'header');
    if exist(strcat(path,base_directory,'\mat_files\',strtok(filename,'.data'),'_data','.mat'))~=2 || force_write==1
        save(strcat(path,base_directory,'\mat_files\',strtok(filename,'.data'),'_data','.mat'),'signals');
    else
        disp('file already exist --> not saved again')
    end
    %save(strcat(path,base_directory,'\mat_files\',strtok(dirstruct_data(i).name,'.data'),'_all_Headers','.mat'),'All_Headers');
else
    if force_write==2
        
    else
        file_name = strcat(path,base_directory,'/mat_files/',strtok(filename,'.data'),'_data','.mat');
        if exist(file_name)~=2 || force_write==1
            save(file_name,'signals');
            disp('data file saved')
        else
            disp('file already exist --> not saved again')
        end
    end
end
end
