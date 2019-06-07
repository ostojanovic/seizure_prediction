
dbstop if error
archstr = computer('arch');
if strcmp(archstr,'win64')

else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_97002/adm_970102/rec_97002102';
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset

electrode_sets{1}.names{1} = 'GG5';               % grid, depth and strip electrodes
electrode_sets{1}.names{2} = 'GG3';
electrode_sets{1}.names{3} = 'GF3';
electrode_sets{1}.names{4} = 'GE7';
electrode_sets{1}.names{5} = 'GF1';
electrode_sets{1}.names{6} = 'GB2';
electrode_sets{1}.names{7} = 'GE4';
electrode_sets{1}.names{8} = 'GE5';
electrode_sets{1}.names{9} = 'GF5';
electrode_sets{1}.names{10} = 'GF6';
electrode_sets{1}.names{11} = 'GA1';
electrode_sets{1}.names{12} = 'GB3';
electrode_sets{1}.names{13} = 'GB4';
electrode_sets{1}.names{14} = 'GB6';
electrode_sets{1}.names{15} = 'GB7';
electrode_sets{1}.names{16} = 'GC1';
electrode_sets{1}.names{17} = 'GC2';
electrode_sets{1}.names{18} = 'GC5';
electrode_sets{1}.names{19} = 'GC3';
electrode_sets{1}.names{20} = 'GC6';
electrode_sets{1}.names{21} = 'GC7';
electrode_sets{1}.names{22} = 'GD1';
electrode_sets{1}.names{23} = 'GD2';
electrode_sets{1}.names{24} = 'GD4';
electrode_sets{1}.names{25} = 'GD5';
electrode_sets{1}.names{26} = 'GD6';
electrode_sets{1}.names{27} = 'GH3';
electrode_sets{1}.names{28} = 'GH7';
electrode_sets{1}.names{29} = 'GA3';
electrode_sets{1}.names{30} = 'GH6';
electrode_sets{1}.names{31} = 'GA4';
electrode_sets{1}.names{32} = 'GA6';
electrode_sets{1}.names{33} = 'GA7';
electrode_sets{1}.names{34} = 'GE1';
electrode_sets{1}.names{35} = 'GE2';
electrode_sets{1}.names{36} = 'GG1';
electrode_sets{1}.names{37} = 'GH4';
electrode_sets{1}.names{38} = 'GG4';
electrode_sets{1}.names{39} = 'GG7';
electrode_sets{1}.names{40} = 'GH1';
electrode_sets{1}.names{41} = 'GF4';
electrode_sets{1}.names{42} = 'GA2';
electrode_sets{1}.names{43} = 'GA5';
electrode_sets{1}.names{44} = 'GB1';
electrode_sets{1}.names{45} = 'GB5';
electrode_sets{1}.names{46} = 'GC4';
electrode_sets{1}.names{47} = 'GD3';
electrode_sets{1}.names{48} = 'GD7';
electrode_sets{1}.names{49} = 'GE3';
electrode_sets{1}.names{50} = 'GE6';
electrode_sets{1}.names{51} = 'GF2';
electrode_sets{1}.names{52} = 'GF7';
electrode_sets{1}.names{53} = 'GG2';
electrode_sets{1}.names{54} = 'GG6';
electrode_sets{1}.names{55} = 'GH2';
electrode_sets{1}.names{56} = 'GH5';
electrode_sets{1}.names{57} = 'FL2';
electrode_sets{1}.names{58} = 'FL4';
electrode_sets{1}.names{59} = 'FL3';
electrode_sets{1}.names{60} = 'FL6';
electrode_sets{1}.names{61} = 'FL1';
electrode_sets{1}.names{62} = 'FL5';
electrode_sets{1}.names{63} = 'TL4';
electrode_sets{1}.names{64} = 'TL1';
electrode_sets{1}.names{65} = 'TL3';
electrode_sets{1}.names{66} = 'TL2';
electrode_sets{1}.names{67} = 'FBA1';
electrode_sets{1}.names{68} = 'FBA3';
electrode_sets{1}.names{69} = 'FBA4';
electrode_sets{1}.names{70} = 'FBA2';
electrode_sets{1}.names{71} = 'FBB1';
electrode_sets{1}.names{72} = 'FBB3';
electrode_sets{1}.names{73} = 'FBB4';
electrode_sets{1}.names{74} = 'FBB2';
electrode_sets{1}.names{75} = 'TBA2';
electrode_sets{1}.names{76} = 'TBA3';
electrode_sets{1}.names{77} = 'TBA4';
electrode_sets{1}.names{78} = 'TBA1';
electrode_sets{1}.names{79} = 'TBB4';
electrode_sets{1}.names{80} = 'TBB2';
electrode_sets{1}.names{81} = 'TBB3';
electrode_sets{1}.names{82} = 'TBB1';
electrode_sets{1}.names{83} = 'TBC2';
electrode_sets{1}.names{84} = 'TBC3';
electrode_sets{1}.names{85} = 'TBC6';
electrode_sets{1}.names{86} = 'TBC5';
electrode_sets{1}.names{87} = 'TBC1';
electrode_sets{1}.names{88} = 'TBC4';
electrode_sets{1}.names{89} = 'HR3';
electrode_sets{1}.names{90} = 'HR2';
electrode_sets{1}.names{91} = 'HR10';
electrode_sets{1}.names{92} = 'HR7';
electrode_sets{1}.names{93} = 'HR6';
electrode_sets{1}.names{94} = 'HR9';
electrode_sets{1}.names{95} = 'HR5';
electrode_sets{1}.names{96} = 'HR4';
electrode_sets{1}.names{97} = 'HR1';
electrode_sets{1}.names{98} = 'HR8';

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
