function seizures = read_seizures(path,base_directory)

archstr = computer('arch');

%% -------------------------------------------- clinical seizures -------------------------------------------------

if strcmp(archstr,'win64')
    metadata = importdata(strcat(path,base_directory,'\','seizurelist_clinical.txt'));
else
    metadata = importdata(strcat(path,base_directory,'/','seizurelist_clinical.txt'));
end

number_of_text_rows = size(metadata.textdata,1);
number_of_data_rows = size(metadata.data,1);
dbstop if error

IDXX=0;
for IDX = (1:number_of_data_rows)+(number_of_text_rows - number_of_data_rows)

    IDXX = IDXX+1;
    start_text = metadata.textdata{IDX,1};                      % for reading of the first two columns (start time and end time)
    end_text   = metadata.textdata{IDX,2};

    [IDX_colon] = strfind(start_text,':');
    clincal_seizures.start{IDXX}.hours   = str2num(start_text((IDX_colon(1)-2):(IDX_colon(1)-1)));
    clincal_seizures.start{IDXX}.minutes = str2num(start_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
    clincal_seizures.start{IDXX}.seconds = str2num(start_text((IDX_colon(2)+1):(length(start_text))));
    clincal_seizures.start{IDXX}.sample  = metadata.data(IDX-(number_of_text_rows - number_of_data_rows),1);        % sample onset

    [IDX_colon] = strfind(start_text,'-');
    clincal_seizures.start{IDXX}.year    = str2num(start_text((IDX_colon(1)-4):(IDX_colon(1)-1)));
    clincal_seizures.start{IDXX}.month   = str2num(start_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
    clincal_seizures.start{IDXX}.day     = str2num(start_text((IDX_colon(2)+1):(IDX_colon(2)+2)));

    [IDX_colon] = strfind(end_text,':');
    clincal_seizures.end{IDXX}.hours   = str2num(end_text((IDX_colon(1)-2):(IDX_colon(1)-1)));
    clincal_seizures.end{IDXX}.minutes = str2num(end_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
    clincal_seizures.end{IDXX}.seconds = str2num(end_text((IDX_colon(2)+1):(length(end_text))));
    clincal_seizures.end{IDXX}.sample  = metadata.data(IDX-(number_of_text_rows - number_of_data_rows),2);         % sample offset

    [IDX_colon] = strfind(end_text,'-');
    clincal_seizures.end{IDXX}.year    = str2num(end_text((IDX_colon(1)-4):(IDX_colon(1)-1)));
    clincal_seizures.end{IDXX}.month   = str2num(end_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
    clincal_seizures.end{IDXX}.day     = str2num(end_text((IDX_colon(2)+1):(IDX_colon(2)+2)));
end



%% -------------------------------------------- subclinical seizures -------------------------------------------------

if strcmp(archstr,'win64')
    metadata = importdata(strcat(path,base_directory,'\','seizurelist_subclinical.txt'));
else
    metadata = importdata(strcat(path,base_directory,'/','seizurelist_subclinical.txt'));
end

number_of_text_rows = size(metadata,1);

IDXX=0;
for IDX = (1:number_of_text_rows)

    IDXX = IDXX+1;
    start_text = metadata{IDX,1};

%     if size(start_text,2) > 26           % beacuse there are some measurements that don't have ending time of a seizure, so we exclude them

        [IDX_colon]  = strfind(start_text,':');
        IDX_colon(1) = [];
        subclincal_seizures.start{IDXX}.hours   = str2num(start_text((IDX_colon(1)-2):(IDX_colon(1)-1)));
        subclincal_seizures.start{IDXX}.minutes = str2num(start_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
        subclincal_seizures.start{IDXX}.seconds = str2num(start_text((IDX_colon(2)+1):(IDX_colon(2)+3)));

        subclincal_seizures.end{IDXX}.hours     = str2num(start_text((IDX_colon(3)-2):(IDX_colon(3)-1)));
        subclincal_seizures.end{IDXX}.minutes   = str2num(start_text((IDX_colon(4)-2):(IDX_colon(4)-1)));
        subclincal_seizures.end{IDXX}.seconds   = str2num(start_text((IDX_colon(4)+1):(IDX_colon(4)+2)));


        [IDX_colon] = strfind(start_text,'.');
        subclincal_seizures.start{IDXX}.day   = str2num(start_text((IDX_colon(1)-2):(IDX_colon(1)-1)));         % from 4 changed to 2
        subclincal_seizures.start{IDXX}.month = str2num(start_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
        subclincal_seizures.start{IDXX}.year  = str2num(start_text((IDX_colon(2)+2):(IDX_colon(2)+3)));

        subclincal_seizures.end{IDXX}.day    = str2num(start_text((IDX_colon(1)-2):(IDX_colon(1)-1)));          % from 4 changed to 2
        subclincal_seizures.end{IDXX}.month  = str2num(start_text((IDX_colon(2)-2):(IDX_colon(2)-1)));
        subclincal_seizures.end{IDXX}.year   = str2num(start_text((IDX_colon(2)+2):(IDX_colon(2)+3)));

        if subclincal_seizures.start{IDXX}.year < 10
            subclincal_seizures.start{IDXX}.year = subclincal_seizures.start{IDXX}.year+2000;
            subclincal_seizures.end{IDXX}.year   = subclincal_seizures.start{IDXX}.year;
        end

%     end
    seizures.subclincal_seizures = subclincal_seizures;
    seizures.clincal_seizures    = clincal_seizures;
end
end
