
dbstop if error
archstr = computer('arch');
if strcmp(archstr,'win64')

else
    path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
    base_directory = 'pat_25302/adm_253102/rec_25300102';                             % the second patient
    dirstruct_data = dir([path base_directory '/*.data']);
    dirstruct_head = dir([path base_directory '/*.head']);
end

Data_Window = [-5*60 5*60];       % start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset


electrode_sets{1}.names{1} = 'HRA1';         % depth electrodes
electrode_sets{1}.names{2} = 'HRA2';
electrode_sets{1}.names{3} = 'HRA3';
electrode_sets{1}.names{4} = 'HRA4';
electrode_sets{1}.names{5} = 'HRA5';
electrode_sets{1}.names{6} = 'HRB3';
electrode_sets{1}.names{7} = 'HRB5';
electrode_sets{1}.names{8} = 'HRB4';
electrode_sets{1}.names{9} = 'HRB1';
electrode_sets{1}.names{10} = 'HRB2';
electrode_sets{1}.names{11} = 'HRC5';
electrode_sets{1}.names{12} = 'HRC4';
electrode_sets{1}.names{13} = 'HRC3';
electrode_sets{1}.names{14} = 'HRC1';
electrode_sets{1}.names{15} = 'HRC2';
electrode_sets{1}.names{16} = 'HLA4';
electrode_sets{1}.names{17} = 'HLA3';
electrode_sets{1}.names{18} = 'HLA2';
electrode_sets{1}.names{19} = 'HLA5';
electrode_sets{1}.names{20} = 'HLA1';
electrode_sets{1}.names{21} = 'HLB1';
electrode_sets{1}.names{22} = 'HLB5';
electrode_sets{1}.names{23} = 'HLB3';
electrode_sets{1}.names{24} = 'HLB4';
electrode_sets{1}.names{25} = 'HLB2';
electrode_sets{1}.names{26} = 'HLC4';
electrode_sets{1}.names{27} = 'HLC2';
electrode_sets{1}.names{28} = 'HLC3';
electrode_sets{1}.names{29} = 'HLC5';
electrode_sets{1}.names{30} = 'HLC1';

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
if exist(strcat(path,base_directory,'/mat_files/', temp_name ,'_all_Headers','.mat'))==0
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
        Data_windows                           = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
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
    Data_windows                                   = Window_Seizure_Data2(selected_seizures,All_Headers,path,base_directory,Data_Window,electrode_sets);
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
