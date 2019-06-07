
dbstop if error
archstr = computer('arch');
if strcmp(archstr,'win64')

else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_59002/adm_590102/rec_59000102';                             % the second patient
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset

electrode_sets{1}.names{1} = 'GB1';             % grid, depth and strip electrodes
electrode_sets{1}.names{2} = 'GB2';
electrode_sets{1}.names{3} = 'GB3';
electrode_sets{1}.names{4} = 'GB4';
electrode_sets{1}.names{5} = 'GA1';
electrode_sets{1}.names{6} = 'GA2';
electrode_sets{1}.names{7} = 'GA3';
electrode_sets{1}.names{8} = 'GA4';
electrode_sets{1}.names{9} = 'GA5';
electrode_sets{1}.names{10} = 'GA6';
electrode_sets{1}.names{11} = 'GA7';
electrode_sets{1}.names{12} = 'GA8';
electrode_sets{1}.names{13} = 'GB8';
electrode_sets{1}.names{14} = 'GC3';
electrode_sets{1}.names{15} = 'GC5';
electrode_sets{1}.names{16} = 'GC6';
electrode_sets{1}.names{17} = 'GC1';
electrode_sets{1}.names{18} = 'GD4';
electrode_sets{1}.names{19} = 'GD2';
electrode_sets{1}.names{20} = 'GB5';
electrode_sets{1}.names{21} = 'GB6';
electrode_sets{1}.names{22} = 'GB7';
electrode_sets{1}.names{23} = 'GC4';
electrode_sets{1}.names{24} = 'GC2';
electrode_sets{1}.names{25} = 'GC7';
electrode_sets{1}.names{26} = 'GC8';
electrode_sets{1}.names{27} = 'GD1';
electrode_sets{1}.names{28} = 'GD3';
electrode_sets{1}.names{29} = 'GD5';
electrode_sets{1}.names{30} = 'GD6';
electrode_sets{1}.names{31} = 'GD7';
electrode_sets{1}.names{32} = 'GD8';
electrode_sets{1}.names{33} = 'HR6';
electrode_sets{1}.names{34} = 'HR7';
electrode_sets{1}.names{35} = 'HR8';
electrode_sets{1}.names{36} = 'HR9';
electrode_sets{1}.names{37} = 'HR10';
electrode_sets{1}.names{38} = 'HR4';
electrode_sets{1}.names{39} = 'HR5';
electrode_sets{1}.names{40} = 'HR2';
electrode_sets{1}.names{41} = 'HR1';
electrode_sets{1}.names{42} = 'HR3';
electrode_sets{1}.names{43} = 'HL1';
electrode_sets{1}.names{44} = 'HL2';
electrode_sets{1}.names{45} = 'HL3';
electrode_sets{1}.names{46} = 'HL4';
electrode_sets{1}.names{47} = 'HL5';
electrode_sets{1}.names{48} = 'HL6';
electrode_sets{1}.names{49} = 'HL7';
electrode_sets{1}.names{50} = 'HL8';
electrode_sets{1}.names{51} = 'HL9';
electrode_sets{1}.names{52} = 'HL10';
electrode_sets{1}.names{53} = 'TLA1';
electrode_sets{1}.names{54} = 'TLA2';
electrode_sets{1}.names{55} = 'TLA3';
electrode_sets{1}.names{56} = 'TLA4';
electrode_sets{1}.names{57} = 'BLA1';
electrode_sets{1}.names{58} = 'BLA2';
electrode_sets{1}.names{59} = 'BLA3';
electrode_sets{1}.names{60} = 'BLA4';
electrode_sets{1}.names{61} = 'BLB1';
electrode_sets{1}.names{62} = 'BLB2';
electrode_sets{1}.names{63} = 'BLB3';
electrode_sets{1}.names{64} = 'BLB4';
electrode_sets{1}.names{65} = 'BLC3';
electrode_sets{1}.names{66} = 'BLC4';
electrode_sets{1}.names{67} = 'BLC5';
electrode_sets{1}.names{68} = 'BLC6';
electrode_sets{1}.names{69} = 'BLC1';
electrode_sets{1}.names{70} = 'BLC2';
electrode_sets{1}.names{71} = 'TRA1';
electrode_sets{1}.names{72} = 'TRA2';
electrode_sets{1}.names{73} = 'TRA3';
electrode_sets{1}.names{74} = 'TRA4';
electrode_sets{1}.names{75} = 'TRB1';
electrode_sets{1}.names{76} = 'TRB2';
electrode_sets{1}.names{77} = 'TRB3';
electrode_sets{1}.names{78} = 'TRB4';
electrode_sets{1}.names{79} = 'TRC1';
electrode_sets{1}.names{80} = 'TRC2';
electrode_sets{1}.names{81} = 'TRC3';
electrode_sets{1}.names{82} = 'TRC4';
electrode_sets{1}.names{83} = 'TRC5';
electrode_sets{1}.names{84} = 'TRC6';
electrode_sets{1}.names{85} = 'BRA2';
electrode_sets{1}.names{86} = 'BRA1';
electrode_sets{1}.names{87} = 'BRA3';
electrode_sets{1}.names{88} = 'BRA4';
electrode_sets{1}.names{89} = 'BRB4';
electrode_sets{1}.names{90} = 'BRB1';
electrode_sets{1}.names{91} = 'BRB2';
electrode_sets{1}.names{92} = 'BRB3';
electrode_sets{1}.names{93} = 'BRC1';
electrode_sets{1}.names{94} = 'BRC2';
electrode_sets{1}.names{95} = 'BRC3';
electrode_sets{1}.names{96} = 'BRC4';
electrode_sets{1}.names{97} = 'BRC5';
electrode_sets{1}.names{98} = 'BRC6';

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
