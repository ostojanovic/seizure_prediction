
%% loading data

path_directory = '';
patient_id = '';

dir = dir(strcat(path,'baseline_',patient_id,'/','*mat'));
num_files = size(dir,1);

fs = 1024;
Nr_Sliding_windows = 15;
nw   = round(16*(15/Nr_Sliding_windows));       % time-halfbandwidth product
nfft = 1024;                                    % number of dft points

for i=1:num_files

    load(strcat(path,'baseline_',patient_id,'/',dir(i).name));
    interictal = first_half;

    %% defining parameters for welch and multitaper method

    Nr_samples                    = size(interictal,1);
    Samples_per_sliding_window_B1 = floor(Nr_samples/Nr_Sliding_windows);
    NR_samples_per_welch_window   = round(min(Samples_per_sliding_window_B1/10));

    big_spectrogram_interictal = zeros(size(interictal,2),2*Nr_Sliding_windows-1,513);
    AV_baseline = zeros(size(interictal,2),513);

    %% welch and multitaper method

    for IDXC = 1:size(interictal,2)                   % channels to take

        for IDXW=1:(2*Nr_Sliding_windows-1)

            IDXS                               = (1:Samples_per_sliding_window_B1)+(IDXW-1)*floor(Samples_per_sliding_window_B1/2);
            data_vec                           = interictal(IDXS,IDXC);
            [~,f_hz]                           = pwelch(data_vec,NR_samples_per_welch_window,[],nfft,fs);
            spectrogram_interictal_MTM(IDXW,:) = pmtm(data_vec,nw,nfft);
        end

        % 50Hz removal
        [~,idx_to_take] = min((50-f_hz).^2);

        Mask50Hz = ones(size(spectrogram_interictal_MTM));
        Mask50Hz(:,idx_to_take-5:idx_to_take+5) = 0.* Mask50Hz(:,idx_to_take-5:idx_to_take+5);
        spectrogram_interictal_MTM = spectrogram_interictal_MTM.*Mask50Hz;

        AV_baseline_MTM  = mean(spectrogram_interictal_MTM);
        AV_baseline(IDXC,:) = AV_baseline_MTM;

        spectrogram_interictal_MTM_Base_Cor_1 = spectrogram_interictal_MTM./repmat(AV_baseline_MTM,size(spectrogram_interictal_MTM,1),1);

        big_spectrogram_interictal(IDXC,:,:) = spectrogram_interictal_MTM_Base_Cor_1;
    end

    disp('all channels done')

    big_spectrogram_interictal(:,:,idx_to_take-5:idx_to_take+5) = 0;     % to replace nans
    spectrogram_interictal = big_spectrogram_interictal;

    %% saving
    part_temp = strsplit(dir(i).name,'_baseline_');
    name_part  = strsplit(part_temp{2},'_1st');

    savename_interictal = strcat(path_directory,base_directory, 'spectrograms_baseline1_',dataset,'/spectrogram_baseline1_',name_part{1});
    savename_AV_baseline = strcat(path_directory,base_directory,'spectrograms_AV_baseline_',dataset,'/AV_spectrogram_',name_part{1});

    save(savename_AV_baseline,'AV_baseline');
    save(savename_interictal,'patient_id','spectrogram_baseline','electrode_sets');

end
