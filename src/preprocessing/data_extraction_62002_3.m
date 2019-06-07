
dbstop if error
archstr = computer('arch');
if strcmp(archstr,'win64')

else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_62002/adm_620102/rec_62002102';
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset

electrode_sets{1}.names{1} = 'TLA4';     % grid, depth and strip electrodes
electrode_sets{1}.names{2} = 'TLA1';
electrode_sets{1}.names{3} = 'TLA2';
electrode_sets{1}.names{4} = 'TLA3';
electrode_sets{1}.names{5} = 'TLB1';
electrode_sets{1}.names{6} = 'TLB4';
electrode_sets{1}.names{7} = 'TLB2';
electrode_sets{1}.names{8} = 'TLB3';
electrode_sets{1}.names{9} = 'TLC2';
electrode_sets{1}.names{10} = 'TLC3';
electrode_sets{1}.names{11} = 'TLC5';
electrode_sets{1}.names{12} = 'TLC6';
electrode_sets{1}.names{13} = 'TLC1';
electrode_sets{1}.names{14} = 'TLC4';
electrode_sets{1}.names{15} = 'TBA1';
electrode_sets{1}.names{16} = 'TBA3';
electrode_sets{1}.names{17} = 'TBA4';
electrode_sets{1}.names{18} = 'TBA2';
electrode_sets{1}.names{19} = 'TBB1';
electrode_sets{1}.names{20} = 'TBB3';
electrode_sets{1}.names{21} = 'TBB4';
electrode_sets{1}.names{22} = 'TBB2';
electrode_sets{1}.names{23} = 'TBC2';
electrode_sets{1}.names{24} = 'TBC3';
electrode_sets{1}.names{25} = 'TBC5';
electrode_sets{1}.names{26} = 'TBC1';
electrode_sets{1}.names{27} = 'TBC4';
electrode_sets{1}.names{28} = 'TBC6';
electrode_sets{1}.names{29} = 'HR3';
electrode_sets{1}.names{30} = 'HR5';
electrode_sets{1}.names{31} = 'HR6';
electrode_sets{1}.names{32} = 'HR7';
electrode_sets{1}.names{33} = 'HR9';
electrode_sets{1}.names{34} = 'HR10';
electrode_sets{1}.names{35} = 'HR2';
electrode_sets{1}.names{36} = 'HR1';
electrode_sets{1}.names{37} = 'HR4';
electrode_sets{1}.names{38} = 'HR8';

electrode_sets{1}.rereference = 0;
electrode_sets{1}.notch       = 1;
electrode_sets{1}.ztransform  = 0;

electrode_sets{2}.names{1} = 'ECG';         % other electrodes
electrode_sets{2}.rereference = 0;
electrode_sets{2}.notch       = 1;
electrode_sets{2}.ztransform  = 1;


num_files = length(dirstruct_data);
seizures = read_seizures(path,base_directory);
seizures = Find_Baseline_periods(seizures,Data_Window, [-6*60 10*60]);

temp_name = dirstruct_data(1).name;
temp_name = temp_name(1:(-1+findstr(temp_name,'_')));
if exist(strcat(path,base_directory,'/mat_files/', temp_name ,'_all_Headers','.mat'))~=2
    for i = 1:num_files
        try
            signals = extract_header(path,base_directory,dirstruct_head(i).name);
            signals = extract_data(path,base_directory,dirstruct_data(i).name,signals,0);
            header=signals.header;
            All_Headers{i} = header;

            if strcmp(archstr,'win64')
                save(strcat(path,base_directory,'\mat_files\', temp_name ,'_all_Headers','.mat'),'All_Headers');
            else
                save(strcat(path,base_directory,'/mat_files/', temp_name ,'_all_Headers','.mat'),'All_Headers');
            end
            disp(['file ' num2str(i) ' done'])


        catch
            disp(['file ' num2str(i) ' failed'])
        end
    end
else
    if strcmp(archstr,'win64')
        load(strcat(path,base_directory,'\mat_files\', temp_name ,'_all_Headers','.mat'),'All_Headers');
    else
        load(strcat(path,base_directory,'/mat_files/', temp_name ,'_all_Headers','.mat'),'All_Headers');
    end
end

for IDX1=1:size(seizures.baseline.start,2)
    delete(['Data_baseline_' num2str(IDX1) '_log.txt' ])
    diary(['Data_baseline_' num2str(IDX1) '_log.txt' ])
    diary on
    selected_seizures.baseline.start{1}    = seizures.baseline.start{IDX1} ;
    selected_seizures.baseline.end{1}      = seizures.baseline.end{IDX1} ;
    try
        Data_windows                       = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
        save(['Data_baseline_' num2str(IDX1)],'Data_windows','Data_Window','electrode_sets','selected_seizures');
    catch
        disp('nothing in there')
    end
    diary off
end
clear selected_seizures
for IDX1=1:size(seizures.clincal_seizures.start,2)
    delete(['Data_clinical_' num2str(IDX1) '_log.txt' ])
    diary(['Data_clinical_' num2str(IDX1) '_log.txt' ])
    diary on
    selected_seizures.clincal_seizures.start{1} = seizures.clincal_seizures.start{IDX1} ;
    selected_seizures.clincal_seizures.end{1}   = seizures.clincal_seizures.end{IDX1} ;
    try
        Data_windows                                = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
        save(['Data_clinical_' num2str(IDX1)],'Data_windows','Data_Window','electrode_sets','selected_seizures');
    catch
        disp('nothing in there')
    end
    diary off
end
clear selected_seizures
for IDX1=1:size(seizures.subclincal_seizures.start,2)
    delete(['Data_subclinical_' num2str(IDX1) '_log.txt' ])
    diary(['Data_subclinical_' num2str(IDX1) '_log.txt' ])
    diary on
    selected_seizures.subclincal_seizures.start{1} = seizures.subclincal_seizures.start{IDX1} ;
    selected_seizures.subclincal_seizures.end{1}   = seizures.subclincal_seizures.end{IDX1} ;
    try
        Data_windows                                   = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
        save(['Data_subclinical_' num2str(IDX1)],'Data_windows','Data_Window','electrode_sets','selected_seizures');
    catch
        disp('nothing in there')
    end
    diary off
end
