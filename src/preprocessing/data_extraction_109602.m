
dbstop if error
archstr = computer('arch');
if strcmp(archstr,'win64')

else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_109602/adm_1096102/rec_109600102';
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset


electrode_sets{1}.names{1} = 'GC5';      % grid, depth and strip electrodes
electrode_sets{1}.names{2} = 'GB4';
electrode_sets{1}.names{3} = 'GB5';
electrode_sets{1}.names{4} = 'GB6';
electrode_sets{1}.names{5} = 'GB8';
electrode_sets{1}.names{6} = 'GC1';
electrode_sets{1}.names{7} = 'GC3';
electrode_sets{1}.names{8} = 'GC4';
electrode_sets{1}.names{9} = 'GC7';
electrode_sets{1}.names{10} = 'GD3';
electrode_sets{1}.names{11} = 'GC8';
electrode_sets{1}.names{12} = 'GD1';
electrode_sets{1}.names{13} = 'GD5';
electrode_sets{1}.names{14} = 'GD7';
electrode_sets{1}.names{15} = 'GD8';
electrode_sets{1}.names{16} = 'GA1';
electrode_sets{1}.names{17} = 'GB2';
electrode_sets{1}.names{18} = 'GA5';
electrode_sets{1}.names{19} = 'GA6';
electrode_sets{1}.names{20} = 'GA8';
electrode_sets{1}.names{21} = 'GB1';
electrode_sets{1}.names{22} = 'GA2';
electrode_sets{1}.names{23} = 'GA3';
electrode_sets{1}.names{24} = 'GA4';
electrode_sets{1}.names{25} = 'GA7';
electrode_sets{1}.names{26} = 'GB3';
electrode_sets{1}.names{27} = 'GB7';
electrode_sets{1}.names{28} = 'GC2';
electrode_sets{1}.names{29} = 'GC6';
electrode_sets{1}.names{30} = 'GD2';
electrode_sets{1}.names{31} = 'GD4';
electrode_sets{1}.names{32} = 'GD6';
electrode_sets{1}.names{33} = 'TP4';
electrode_sets{1}.names{34} = 'TP3';
electrode_sets{1}.names{35} = 'TP1';
electrode_sets{1}.names{36} = 'TP2';
electrode_sets{1}.names{37} = 'TBA1';
electrode_sets{1}.names{38} = 'TBA3';
electrode_sets{1}.names{39} = 'TBA4';
electrode_sets{1}.names{40} = 'TBA2';
electrode_sets{1}.names{41} = 'TBB3';
electrode_sets{1}.names{42} = 'TBB4';
electrode_sets{1}.names{43} = 'TBB1';
electrode_sets{1}.names{44} = 'TBB2';
electrode_sets{1}.names{45} = 'TBC3';
electrode_sets{1}.names{46} = 'TBC1';
electrode_sets{1}.names{47} = 'TBC6';
electrode_sets{1}.names{48} = 'TBC4';
electrode_sets{1}.names{49} = 'TBC2';
electrode_sets{1}.names{50} = 'TBC5';
electrode_sets{1}.names{51} = 'HL8';
electrode_sets{1}.names{52} = 'HL9';
electrode_sets{1}.names{53} = 'HL4';
electrode_sets{1}.names{54} = 'HL2';
electrode_sets{1}.names{55} = 'HL5';
electrode_sets{1}.names{56} = 'HL6';
electrode_sets{1}.names{57} = 'HL1';
electrode_sets{1}.names{58} = 'HL3';
electrode_sets{1}.names{59} = 'HL7';
electrode_sets{1}.names{60} = 'HL10';
electrode_sets{1}.names{61} = 'HRA1';
electrode_sets{1}.names{62} = 'HRA2';
electrode_sets{1}.names{63} = 'HRA3';
electrode_sets{1}.names{64} = 'HRA4';
electrode_sets{1}.names{65} = 'HRA5';
electrode_sets{1}.names{66} = 'HRB1';
electrode_sets{1}.names{67} = 'HRB2';
electrode_sets{1}.names{68} = 'HRB3';
electrode_sets{1}.names{69} = 'HRB4';
electrode_sets{1}.names{70} = 'HRB5';
electrode_sets{1}.names{71} = 'HRC1';
electrode_sets{1}.names{72} = 'HRC2';
electrode_sets{1}.names{73} = 'HRC3';
electrode_sets{1}.names{74} = 'HRC4';
electrode_sets{1}.names{75} = 'HRC5';

electrode_sets{1}.rereference = 0;
electrode_sets{1}.notch       = 1;
electrode_sets{1}.ztransform  = 0;

electrode_sets{2}.names{1} = 'T1';                         % surface electrodes
electrode_sets{2}.names{2} = 'T2';
electrode_sets{2}.names{3} = 'FP2';
electrode_sets{2}.names{4} = 'F3';
electrode_sets{2}.names{5} = 'F4';
electrode_sets{2}.names{6} = 'C3';
electrode_sets{2}.names{7} = 'C4';
electrode_sets{2}.names{8} = 'P3';
electrode_sets{2}.names{9} = 'P4';
electrode_sets{2}.names{10} = 'O1';
electrode_sets{2}.names{11} = 'O2';
electrode_sets{2}.names{12} = 'F7';
electrode_sets{2}.names{13} = 'F8';
electrode_sets{2}.names{14} = 'T3';
electrode_sets{2}.names{15} = 'T4';
electrode_sets{2}.names{16} = 'T5';
electrode_sets{2}.names{17} = 'FP1';
electrode_sets{2}.names{18} = 'T6';
electrode_sets{2}.names{19} = 'FZ';
electrode_sets{2}.names{20} = 'CZ';
electrode_sets{2}.names{21} = 'PZ';

electrode_sets{2}.rereference = 1;
electrode_sets{2}.notch       = 1;
electrode_sets{2}.ztransform  = 0;

electrode_sets{3}.names{1} = 'ECG';
electrode_sets{3}.names{2} = 'EOG1';
electrode_sets{3}.names{3} = 'EOG2';
electrode_sets{3}.names{4} = 'EMG';

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
