
dbstop if error

archstr = computer('arch');
if strcmp(archstr,'win64')
    path = 'Y:\projects\Data\intracranial_data\Freiburg_epilepsy_unit\';
    base_directory = 'pat_11502\adm_115102\rec_11500102';                           % the first patient
    dirstruct_data = dir([path base_directory '\*.data']);
    dirstruct_head = dir([path base_directory '\*.head']);
else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_11502/adm_115102/rec_11500102';                             % the second patient
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset


electrode_sets{1}.names{1} = 'HR1';         % grid, depth and strip electrodes
electrode_sets{1}.names{2} = 'HR2';
electrode_sets{1}.names{3} = 'HR3';
electrode_sets{1}.names{4} = 'HR4';
electrode_sets{1}.names{5} = 'HR5';
electrode_sets{1}.names{6} = 'HR6';
electrode_sets{1}.names{7} = 'HR7';
electrode_sets{1}.names{8} = 'HR8';
electrode_sets{1}.names{9} = 'HR9';
electrode_sets{1}.names{10}= 'HR10';
electrode_sets{1}.names{11}= 'TO1';
electrode_sets{1}.names{12}= 'TO2';
electrode_sets{1}.names{13}= 'TO3';
electrode_sets{1}.names{14}= 'TO4';
electrode_sets{1}.names{15}= 'IH1';
electrode_sets{1}.names{16}= 'IH2';
electrode_sets{1}.names{17}= 'IH3';
electrode_sets{1}.names{18}= 'IH4';
electrode_sets{1}.names{19}= 'IH5';
electrode_sets{1}.names{20}= 'IH6';
electrode_sets{1}.names{21}= 'IH7';
electrode_sets{1}.names{22}= 'IH8';
electrode_sets{1}.names{23}= 'FR1';
electrode_sets{1}.names{24}= 'FR2';
electrode_sets{1}.names{25}= 'FR3';
electrode_sets{1}.names{26}= 'FR4';
electrode_sets{1}.names{27}= 'FR5';
electrode_sets{1}.names{28}= 'FR6';
electrode_sets{1}.names{29}= 'FR7';
electrode_sets{1}.names{30}= 'FR8';
electrode_sets{1}.names{31}= 'FR9';
electrode_sets{1}.names{32}= 'FR10';
electrode_sets{1}.names{33}= 'FR11';
electrode_sets{1}.names{34}= 'FR12';
electrode_sets{1}.names{35}= 'TBA1';
electrode_sets{1}.names{36}= 'TBA2';
electrode_sets{1}.names{37}= 'TBA3';
electrode_sets{1}.names{38}= 'TBA4';
electrode_sets{1}.names{39}= 'TBB1';
electrode_sets{1}.names{40}= 'TBB2';
electrode_sets{1}.names{41}= 'TBB3';
electrode_sets{1}.names{42}= 'TBB4';
electrode_sets{1}.names{43}= 'G1';
electrode_sets{1}.names{44}= 'G2';
electrode_sets{1}.names{45}= 'G3';
electrode_sets{1}.names{46}= 'G4';
electrode_sets{1}.names{47}= 'G5';
electrode_sets{1}.names{48}= 'G6';
electrode_sets{1}.names{49}= 'G7';
electrode_sets{1}.names{50}= 'G8';
electrode_sets{1}.names{51}= 'G9';
electrode_sets{1}.names{52}= 'G10';
electrode_sets{1}.names{53}= 'G11';
electrode_sets{1}.names{54}= 'G12';
electrode_sets{1}.names{55}= 'G13';
electrode_sets{1}.names{56}= 'G14';
electrode_sets{1}.names{57}= 'G15';
electrode_sets{1}.names{58}= 'G16';

electrode_sets{1}.rereference = 0;
electrode_sets{1}.notch       = 1;
electrode_sets{1}.ztransform  = 0;

electrode_sets{2}.names{1} = 'FP1';                         % surface electrodes
electrode_sets{2}.names{2} = 'FP2';
electrode_sets{2}.names{3} = 'F3';
electrode_sets{2}.names{4} = 'F4';
electrode_sets{2}.names{5} = 'C3';
electrode_sets{2}.names{6} = 'C4';
electrode_sets{2}.names{7} = 'P3';
electrode_sets{2}.names{8} = 'P4';
electrode_sets{2}.names{9} = 'O1';
electrode_sets{2}.names{10}= 'O2';
electrode_sets{2}.names{11}= 'F7';
electrode_sets{2}.names{12}= 'F8';
electrode_sets{2}.names{13}= 'T3';
electrode_sets{2}.names{14}= 'T4';
electrode_sets{2}.names{15}= 'T5';
electrode_sets{2}.names{16}= 'T6';
electrode_sets{2}.names{17}= 'FZ';
electrode_sets{2}.names{18}= 'CZ';
electrode_sets{2}.names{19}= 'PZ';
electrode_sets{2}.names{20}= 'T1';
electrode_sets{2}.names{21}= 'T2';

electrode_sets{2}.rereference = 1;
electrode_sets{2}.notch       = 1;
electrode_sets{2}.ztransform  = 0;

electrode_sets{3}.names{1}= 'EOG1';
electrode_sets{3}.names{2}= 'EOG2';
electrode_sets{3}.names{3}= 'EMG';
electrode_sets{3}.names{4}= 'ECG';

electrode_sets{3}.rereference = 0;
electrode_sets{3}.notch       = 1;
electrode_sets{3}.ztransform  = 0;

num_files = length(dirstruct_data);
seizures = read_seizures(path,base_directory);
seizures = Find_Baseline_periods(seizures,Data_Window, [-6*60 10*60]);

temp_name = dirstruct_data(1).name;
temp_name = temp_name(1:(-1+findstr(temp_name,'_')));
if exist(strcat(path,base_directory,'/mat_files/', temp_name ,'_all_Headers','.mat'))~=2
    for i = 1:num_files
        try
            signals = extract_header(path,base_directory,dirstruct_head(i).name);
            signals = extract_data(path,base_directory,dirstruct_data(i).name,signals,1);
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
        Data_windows                            = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
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
        Data_windows                               = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
        save(['Data_subclinical_' num2str(IDX1)],'Data_windows','Data_Window','electrode_sets','selected_seizures');
    catch
        disp('nothing in there')
    end
    diary off
end
