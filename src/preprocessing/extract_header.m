function signals = extract_header(path,base_directory,filename)
archstr = computer('arch');

if strcmp(archstr,'win64')
    % metadata from .head file
    metadata = importdata(strcat(path,base_directory,'\',filename));
else
    if isempty(strfind(filename,'/'))
        name = strcat(path,base_directory,'/',filename);
    else
        name = filename;
    end
    metadata = importdata(name);
end

% date and time
[~,time_file] = strtok(metadata{1,1},'=');
time_file     = time_file(2:end);
[IDX_colon]   = strfind(time_file,':');

signals.header.hours    = str2num(time_file((IDX_colon(1)-2):(IDX_colon(1)-1)));
signals.header.minutes  = str2num(time_file((IDX_colon(2)-2):(IDX_colon(2)-1)));
signals.header.seconds  = str2num(time_file((IDX_colon(2)+1):(length( time_file))));
signals.header.filename = filename;
signals.header.base_directory = base_directory;
[IDX_colon]   = strfind(time_file,'-');
signals.header.year  = str2num(time_file((IDX_colon(1)-4):(IDX_colon(1)-1)));
signals.header.month = str2num(time_file((IDX_colon(2)-2):(IDX_colon(2)-1)));
signals.header.day   = str2num(time_file((IDX_colon(2)+1):(IDX_colon(2)+2)));

% num_samples
[~,num_samples] = strtok(metadata{2,1},'=');
remain = strsplit(num_samples,'=');
num_samples = str2num(remain{1,2});
signals.header.num_samples = num_samples;
clear remain;

% sample frequency
[~,sample_freq] = strtok(metadata{3,1},'=');
sample_freq = str2num(sample_freq(2:end));
signals.header.sample_freq = sample_freq;

% conversion factor
[~,conversion_factor] = strtok(metadata{4,1},'=');
remain = strsplit(conversion_factor,'=');
conversion_factor = str2num(remain{1,2});
signals.header.conversion_factor = conversion_factor;
clear remain;

% num_channels
[~,num_channels] = strtok(metadata{5,1},'=');
remain = strsplit(num_channels,'=');
num_channels = str2num(remain{1,2});
signals.header.num_channels = num_channels;
clear remain;

% electrodes
[~,electrodes] = strtok(metadata{6,1},'=');
electrodes = electrodes(3:end-1);
electrodes = strsplit(electrodes,',');
signals.header.electrodes = electrodes;

% details about the patient
[~,pat_id] = strtok(metadata{7,1},'=');
pat_id = str2num(pat_id(2:end));
signals.header.pat_id = pat_id;

[~,adm_id] = strtok(metadata{8,1},'=');
adm_id = str2num(adm_id(2:end));
signals.header.adm_id = adm_id;

[~,rec_id] = strtok(metadata{9,1},'=');
rec_id = str2num(rec_id(2:end));
signals.header.rec_id = rec_id;

% duration & sample_bytes
[~,duration_in_sec] = strtok(metadata{10,1},'=');
duration_in_sec = str2num(duration_in_sec(2:end));
signals.header.duration_in_sec = duration_in_sec;

[~,sample_bytes] = strtok(metadata{11,1},'=');
sample_bytes = str2num(sample_bytes(2:end));
signals.header.sample_bytes = sample_bytes;
header = signals.header;

if strcmp(archstr,'win64')
    save(strcat(path,base_directory,'mat_files\',strtok(filename,'.header'),'_header','.mat'),'header');
else
    if isempty(strfind(filename,'/'))
         file_path = strcat(path,base_directory,'/mat_files/',strtok(filename,'.head'),'_header','.mat');
         %name = strcat(path,base_directory,'/',filename);
    else
        file_path = filename;
    end
   
    signals.header.file_path = name;
    if exist(file_path) == 2
        disp('file exist ---> not saved again')
    else
        file_name = strcat(path,base_directory,'/mat_files/',strtok(filename,'.head'),'_header','.mat');
        save(file_name,'header');
        disp('file saved')
    end
end
end