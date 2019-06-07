function Data_window = Window_Seizure_Data2(seizures,All_Headers,path,base_directory,Data_Window,channel_sets)

%temp1 = seizures.clincal_seizures.start{1};
%temp2 = seizures.clincal_seizures.end{1};
%temp3 = seizures.subclincal_seizures.start{1};
%temp4 = seizures.subclincal_seizures.end{1};
%clear seizures;

%seizures.clincal_seizures.start{1}    = temp1;
%seizures.clincal_seizures.end{1}      = temp2;
%seizures.subclincal_seizures.start{1} = temp3;
%seizures.subclincal_seizures.end{1}   = temp4;
% start and end time in seconds  realtive to onset : eg. [-10 0 ] means 10 second before till onset  [5 15 ] from 5 secs after onset till 15 seconds after onset

Fs = All_Headers{1}.sample_freq;  % Sampling Frequency

Fpass1 = 48;              % First Passband Frequency
Fstop1 = 49;              % First Stopband Frequency
Fstop2 = 51;              % Second Stopband Frequency
Fpass2 = 52;              % Second Passband Frequency
Dpass1 = 0.028774368332;  % First Passband Ripple
Dstop  = 0.001;           % Stopband Attenuation
Dpass2 = 0.057501127785;  % Second Passband Ripple
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass1 Fstop1 Fstop2 Fpass2]/(Fs/2), [1 0 ...
                          1], [Dpass1 Dstop Dpass2]);

% Calculate the coefficients using the FIRPM function.
b  = firpm(N, Fo, Ao, W, {dens});
Hd = dfilt.dffir(b);


try
    Number_of_clinical_s = size(seizures.clincal_seizures.start,2);
catch
    Number_of_clinical_s = 0;
end
try
    Number_of_baseline_s = size(seizures.baseline.start,2);
catch
    Number_of_baseline_s = 0;
end
try
    Number_of_subclinical_s = size(seizures.subclincal_seizures.start,2);
catch
    Number_of_subclinical_s = 0;
end

for IDXS = 1: Number_of_baseline_s
    
    seizure_onset         = seizures.baseline.start{IDXS};
    act_time_onset        = seizure_onset.day*24*60*60+seizure_onset.hours*60*60+ seizure_onset.minutes*60+ round(seizure_onset.seconds);       % actual time onset
    act_time_start_window = act_time_onset+Data_Window(1);
    act_time_end_window   = act_time_onset+Data_Window(2);
    Seizure_INDEX_range   = act_time_start_window:act_time_end_window;
    Window_Composed_IDX   = 0;
    file_starts_all       = [];
    
    for IDX_Header = 1:size(All_Headers,2)
        
        act_header   = All_Headers{IDX_Header};               % actual header
        file_starts  = act_header.day*24*60*60+act_header.hours*60*60+ act_header.minutes*60+ act_header.seconds;
        file_ends    = act_header.duration_in_sec+file_starts ;
        file_Time    = round(file_starts+((1:act_header.num_samples)./act_header.sample_freq));
        IDX_data_use = find(ismember(file_Time,Seizure_INDEX_range));
        
        if ~isempty(IDX_data_use)
            
            %act_header
            Window_Composed_IDX = Window_Composed_IDX+1;
            
            USE_Data{Window_Composed_IDX}.Header       = IDX_Header;
            USE_Data{Window_Composed_IDX}.IDX_data_use = IDX_data_use;
            USE_Data{Window_Composed_IDX}.file_starts  = file_starts;
            USE_Data{Window_Composed_IDX}.filename     = act_header.filename;
            USE_Data{Window_Composed_IDX}.file_path    = act_header.file_path;
            file_starts_all = [file_starts_all  file_starts];
        end
        
    end
    
    if Window_Composed_IDX==1
        
        signals = extract_header(path,base_directory,USE_Data{1}.file_path);
        signals = extract_data(path,base_directory,[USE_Data{1}.filename(1:(end-4)) 'data'] ,signals,2);  
        
        if size(channel_sets,2) == 1
            [~,location]=ismember(channel_sets.names,signals.header.electrodes);           
        else
            location = [];
            for i=1:size(channel_sets,2)
                [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                location = [location, loc];
            end 
        end
        signals.header.num_channels = size(location,2);

        signals.data = signals.data(location,:);
        Data_window.baseline{IDXS} = signals.data(:,USE_Data{1}.IDX_data_use);
        
        disp('Cutting Data')      
        disp(['Filename ' USE_Data{1}.filename(1:(end-4)) ]);
        disp(['Startindex ' num2str(min(USE_Data{1}.IDX_data_use))]);
        disp(['Lastindex ' num2str(max(USE_Data{1}.IDX_data_use))]);
       
        signals.header
    else
        
        [~,IDX_sort] = sort(file_starts_all);
        this_header =  All_Headers{1};
       
        for IDXsort=1:Window_Composed_IDX
            for IDXElec=1:size(this_header.electrodes,2)
                if this_header.electrodes{IDXElec} ~=All_Headers{IDXsort}.electrodes{IDXElec}
                    error('electrode setting is different across files')
                end
            end
        end
        for IDXset =1:size(channel_sets,2)
            Channel_Index_use = [];
            for IDXcha=1:size(channel_sets{IDXset}.names,2)
                act_channel_name =  channel_sets{IDXset}.names{IDXcha};
                for IDXcha2=1:size(this_header.electrodes,2)
                    if strcmp(this_header.electrodes{IDXcha2},  act_channel_name)
                          Channel_Index_use = [ Channel_Index_use IDXcha2];
                    end
                end
            end
            Set{IDXset}.Channel_Index_use=Channel_Index_use;
        end
        
        temp=[];
        for IDXsort=1:Window_Composed_IDX
            this_header =  All_Headers{IDXsort};
            file_read_IDX = IDX_sort(IDXsort);
            signals = extract_header(path,base_directory,USE_Data{file_read_IDX}.file_path);
            signals = extract_data(path,base_directory,[USE_Data{file_read_IDX}.filename(1:(end-4)) 'data'] ,signals,2);
            
            if size(channel_sets,2) == 1
                [~,location]=ismember(channel_sets.names,signals.header.electrodes);
            else
                location = [];
                for i=1:size(channel_sets,2)
                    [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                    location = [location, loc];
                end
            end
            signals.header.num_channels = size(location,2);
            signals.data = signals.data(location,:);
            
            disp('Concatinating Data')
            disp(['Segement ' num2str(IDXsort)]);
            disp(['Filename ' USE_Data{file_read_IDX}.filename(1:(end-4)) ]);
            disp(['Startindex ' num2str(min(USE_Data{file_read_IDX}.IDX_data_use) )]);
            disp(['Lastindex ' num2str(max(USE_Data{file_read_IDX}.IDX_data_use) )]);
            signals.header
            temp = [temp signals.data(:,USE_Data{file_read_IDX}.IDX_data_use)];
        end
        for IDXset =1:size(channel_sets,2)
           Channel_Index_use = Set{IDXset}.Channel_Index_use;
           temptemp = temp(Channel_Index_use,:);
           if channel_sets{IDXset}.ztransform==1
              temptemp = double(temptemp);
              stdtemp = 1./std((temptemp'));             
              for IDXCH=1:size( temptemp,1)
                  temptemp(IDXCH,:) =temptemp(IDXCH,:) .* stdtemp(IDXCH);
              end
              temptemp = 15000.*temptemp./max(abs(temptemp(:)));
           end
           if channel_sets{IDXset}.notch==1
               temptemp = double(temptemp);
               for IDXCH=1:size( temptemp,1)
                   temptemp(IDXCH,:) = filter(Hd,temptemp(IDXCH,:));                   
               end               
           end
           if channel_sets{IDXset}.rereference==1
               tempmedian = (repmat(median(temptemp ),length(Channel_Index_use),1));
               Data_window.baseline{IDXS}.Data_for_channel_Set{IDXset} =   int16(round(temptemp -tempmedian )) ;
           else
               Data_window.baseline{IDXS}.Data_for_channel_Set{IDXset} =   int16(round(temptemp));
           end
           
        end

    end
end

for IDXS = 1:Number_of_subclinical_s
    
    seizure_onset         = seizures.subclincal_seizures.start{IDXS};
    act_time_onset        = seizure_onset.day*24*60*60+seizure_onset.hours*60*60+ seizure_onset.minutes*60+ round(seizure_onset.seconds);       % actual time onset
    act_time_start_window = act_time_onset+Data_Window(1);
    act_time_end_window   = act_time_onset+Data_Window(2);
    Seizure_INDEX_range   = act_time_start_window:act_time_end_window;
    Window_Composed_IDX   = 0;
    file_starts_all       = [];
    
    for IDX_Header = 1:size(All_Headers,2)
        
        act_header   = All_Headers{IDX_Header};               % actual header
        file_starts  = act_header.day*24*60*60+act_header.hours*60*60+ act_header.minutes*60+ act_header.seconds;
        file_ends    = act_header.duration_in_sec+file_starts ;
        file_Time    = round(file_starts+((1:act_header.num_samples)./act_header.sample_freq));
        IDX_data_use = find(ismember(file_Time,Seizure_INDEX_range));
        
        if ~isempty(IDX_data_use)
            
            %act_header
            Window_Composed_IDX = Window_Composed_IDX+1;
            
            USE_Data{Window_Composed_IDX}.Header       = IDX_Header;
            USE_Data{Window_Composed_IDX}.IDX_data_use = IDX_data_use;
            USE_Data{Window_Composed_IDX}.file_starts  = file_starts;
            USE_Data{Window_Composed_IDX}.filename     = act_header.filename;
            USE_Data{Window_Composed_IDX}.file_path    = act_header.file_path;
            file_starts_all = [file_starts_all  file_starts];
        end
        
    end
    
    if Window_Composed_IDX==1
        
        signals = extract_header(path,base_directory,USE_Data{1}.file_path);
        signals = extract_data(path,base_directory,[USE_Data{1}.filename(1:(end-4)) 'data'] ,signals,2);
        Data_window.subclinical{IDXS} = signals.data(:,USE_Data{1}.IDX_data_use);
      
        if size(channel_sets,2) == 1
            [~,location]=ismember(channel_sets.names,signals.header.electrodes);
        else
            location = [];
            for i=1:size(channel_sets,2)
                [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                location = [location, loc];
            end
        end
        
        signals.header.num_channels = size(location,2);
        signals.data = signals.data(location,:);
        
        disp('Cutting Data')
        disp(['Filename ' USE_Data{1}.filename(1:(end-4)) ]);
        disp(['Startindex ' num2str(min(USE_Data{1}.IDX_data_use))]);
        disp(['Lastindex ' num2str(max(USE_Data{1}.IDX_data_use))]);
        
    else
        
        [~,IDX_sort] = sort(file_starts_all);
        this_header =  All_Headers{1};
       
        for IDXsort=1:Window_Composed_IDX
            for IDXElec=1:size(this_header.electrodes,2)
                if this_header.electrodes{IDXElec} ~=All_Headers{IDXsort}.electrodes{IDXElec}
                    error('electrode setting is different across files')
                end
            end
        end
        for IDXset =1:size(channel_sets,2)
            Channel_Index_use = [];
            for IDXcha=1:size(channel_sets{IDXset}.names,2)
                act_channel_name =  channel_sets{IDXset}.names{IDXcha};
                for IDXcha2=1:size(  this_header.electrodes,2)
                    if strcmp(this_header.electrodes{IDXcha2},  act_channel_name)
                          Channel_Index_use = [ Channel_Index_use IDXcha2];
                    end
                end
            end
            Set{IDXset}.Channel_Index_use=Channel_Index_use;
        end
        
        temp=[];
        for IDXsort=1:Window_Composed_IDX
            this_header =  All_Headers{IDXsort};
            file_read_IDX = IDX_sort(IDXsort);
            signals = extract_header(path,base_directory,USE_Data{file_read_IDX}.file_path);
            
            %signals = extract_header(path,base_directory,USE_Data{file_read_IDX}.filename);
            signals = extract_data(path,base_directory,[USE_Data{file_read_IDX}.filename(1:(end-4)) 'data'] ,signals,2);
            temp = [temp signals.data(:,USE_Data{file_read_IDX}.IDX_data_use)];
            
            if size(channel_sets,2) == 1
                [~,location]=ismember(channel_sets.names,signals.header.electrodes);
            else
                location = [];
                for i=1:size(channel_sets,2)
                    [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                    location = [location, loc];
                end
            end
            signals.header.num_channels = size(location,2);
            signals.data = signals.data(location,:);
            
            disp('Concatinating Data')
            disp(['Segement ' num2str(IDXsort)]);
            disp(['Filename ' USE_Data{file_read_IDX}.filename(1:(end-4)) ]);
            disp(['Startindex ' num2str(min(USE_Data{file_read_IDX}.IDX_data_use) )]);
            disp(['Lastindex ' num2str(max(USE_Data{file_read_IDX}.IDX_data_use) )]);
            signals.header
            
        end
        for IDXset =1:size(channel_sets,2)
           Channel_Index_use = Set{IDXset}.Channel_Index_use;
           temptemp = temp(Channel_Index_use,:);
           if channel_sets{IDXset}.ztransform==1
              temptemp = double(temptemp);
              stdtemp = 1./std((temptemp'));             
              for IDXCH=1:size( temptemp,1)
                  temptemp(IDXCH,:) =temptemp(IDXCH,:) .* stdtemp(IDXCH);
              end
              temptemp = 15000.*temptemp./max(abs(temptemp(:)));
           end
           if channel_sets{IDXset}.notch==1
               temptemp = double(temptemp);
               for IDXCH=1:size( temptemp,1)
                   temptemp(IDXCH,:) = filter(Hd,temptemp(IDXCH,:));                   
               end               
           end
           if channel_sets{IDXset}.rereference==1
               tempmedian = (repmat(median(temptemp ),length(Channel_Index_use),1));
               Data_window.subclinical{IDXS}.Data_for_channel_Set{IDXset} =   int16(round(temptemp -tempmedian )) ;
           else
               Data_window.subclinical{IDXS}.Data_for_channel_Set{IDXset} =   int16(round(temptemp));
           end
           
        end

    end
end

for IDXS = 1:Number_of_clinical_s
    
    seizure_onset         = seizures.clincal_seizures.start{IDXS};
    act_time_onset        = seizure_onset.day*24*60*60+seizure_onset.hours*60*60+ seizure_onset.minutes*60+ round(seizure_onset.seconds);
    act_time_start_window = act_time_onset+Data_Window(1);
    act_time_end_window   = act_time_onset+Data_Window(2);
    Seizure_INDEX_range   = act_time_start_window:act_time_end_window;
    Window_Composed_IDX   = 0;
    file_starts_all       = [];
    
    for IDX_Header = 1:size(All_Headers,2)
        
        act_header   = All_Headers{IDX_Header};
        file_starts  = act_header.day*24*60*60+act_header.hours*60*60+ act_header.minutes*60+ act_header.seconds;
        file_ends    = act_header.duration_in_sec+ file_starts ;
        file_Time    = round(file_starts+((1:act_header.num_samples)./act_header.sample_freq));
        IDX_data_use = find(ismember( file_Time,Seizure_INDEX_range));
        
        if ~isempty( IDX_data_use)
            
            %act_header
            Window_Composed_IDX = Window_Composed_IDX+1;
            
            USE_Data{Window_Composed_IDX}.Header       = IDX_Header;
            USE_Data{Window_Composed_IDX}.IDX_data_use = IDX_data_use;
            USE_Data{Window_Composed_IDX}.file_starts  = file_starts;
            USE_Data{Window_Composed_IDX}.filename = act_header.filename;
            USE_Data{Window_Composed_IDX}.file_path    = act_header.file_path;
            file_starts_all = [file_starts_all  file_starts ];
        end
        
    end
    if Window_Composed_IDX==1
        
        %signals = extract_header(path,base_directory,USE_Data{1}.filename);
        signals = extract_header(path,base_directory,USE_Data{1}.file_path);
        signals = extract_data(path,base_directory,[USE_Data{1}.filename(1:(end-4)) 'data'] ,signals,2);
        Data_window.clinical{IDXS} = signals.data(:,USE_Data{1}.IDX_data_use);
        
        if size(channel_sets,2) == 1
            [~,location]=ismember(channel_sets.names,signals.header.electrodes);
        else
            location = [];
            for i=1:size(channel_sets,2)
                [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                location = [location, loc];
            end
        end
        signals.header.num_channels = size(location,2);
        signals.data = signals.data(location,:);
        
        disp('Cutting Data')      
        disp(['Filename ' USE_Data{1}.filename(1:(end-4)) ]);
        disp(['Startindex ' num2str(min(USE_Data{1}.IDX_data_use))]);
        disp(['Lastindex ' num2str(max(USE_Data{1}.IDX_data_use))]);
    else
        [~,IDX_sort] = sort(file_starts_all);
        this_header =  All_Headers{1};
       
        for IDXsort=1:Window_Composed_IDX
            for IDXElec=1:size(this_header.electrodes,2)
                if this_header.electrodes{IDXElec} ~=All_Headers{IDXsort}.electrodes{IDXElec}
                    error('electrode setting is different across files')
                end
            end
        end
        for IDXset =1:size(channel_sets,2)
            Channel_Index_use = [];
            for IDXcha=1:size(channel_sets{IDXset}.names,2)
                act_channel_name =  channel_sets{IDXset}.names{IDXcha};
                for IDXcha2=1:size(  this_header.electrodes,2)
                    if strcmp(this_header.electrodes{IDXcha2},  act_channel_name)
                          Channel_Index_use = [ Channel_Index_use IDXcha2];
                    end
                end
            end
            Set{IDXset}.Channel_Index_use=Channel_Index_use;
        end
        
        temp=[];
        for IDXsort=1:Window_Composed_IDX
            this_header =  All_Headers{IDXsort};
            file_read_IDX = IDX_sort(IDXsort);
            signals = extract_header(path,base_directory,USE_Data{file_read_IDX}.file_path);
            signals = extract_data(path,base_directory,[USE_Data{file_read_IDX}.filename(1:(end-4)) 'data'] ,signals,2);
            temp = [temp signals.data(:,USE_Data{file_read_IDX}.IDX_data_use)];
            
            if size(channel_sets,2) == 1
                [~,location]=ismember(channel_sets.names,signals.header.electrodes);
            else
                location = [];
                for i=1:size(channel_sets,2)
                    [~,loc]=ismember(channel_sets{1,i}.names,signals.header.electrodes);
                    location = [location, loc];
                end
            end
            signals.header.num_channels = size(location,2);
            signals.data = signals.data(location,:);
            
            disp('Concatinating Data')
            disp(['Segement ' num2str(IDXsort)]);
            disp(['Filename ' USE_Data{file_read_IDX}.filename(1:(end-4)) ]);
            disp(['Startindex ' num2str(min(USE_Data{file_read_IDX}.IDX_data_use) )]);
            disp(['Lastindex ' num2str(max(USE_Data{file_read_IDX}.IDX_data_use) )]);
            signals.header
        end
  
        for IDXset =1:size(channel_sets,2)
           Channel_Index_use = Set{IDXset}.Channel_Index_use;
           temptemp = temp(Channel_Index_use,:);
           if channel_sets{IDXset}.ztransform==1
              temptemp = double(temptemp);
              stdtemp = 1./std((temptemp'));             
              for IDXCH=1:size( temptemp,1)
                  temptemp(IDXCH,:) =temptemp(IDXCH,:) .* stdtemp(IDXCH);
              end
              temptemp = 15000.*temptemp./max(abs(temptemp(:)));
           end
           if channel_sets{IDXset}.rereference==1               
               tempmedian = (repmat(median(temptemp ),length(Channel_Index_use),1));               
               Data_window.clinical{IDXS}.Data_for_channel_Set{IDXset} =   int16(temptemp -tempmedian ) ;
           else
               Data_window.clinical{IDXS}.Data_for_channel_Set{IDXset} =   int16(temptemp);
           end
           
        end
   
    end
    
end

%Data_window
