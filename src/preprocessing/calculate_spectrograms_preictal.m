
%% loading data

path = '';           % path goes here
patient_id = '';     % patient id goes here

dir_AV_baseline = dir(strcat(path,'/','*mat'));
dir_preictal = dir(strcat(path,'/','*mat'));
num_files = size(dir_preictal,1);

fs = 256;
Nr_Sliding_windows = 15;
nw   = round(16*(15/Nr_Sliding_windows));       % time-halfbandwidth product
nfft = 1024;                                    % number of dft points

for i=1:num_files

    load(strcat(path,'/',dir_preictal(i).name));
    preictal = first_half;

    % loading AV_baseline spectrogram
    num_baseline_files = size(dir_AV_baseline,1);
    
    t = clock;
    rng(floor((i*60 + t(6))*10000))
    idx_baseline = randi([1 num_baseline_files],1,1);
    AV_baseline_used = dir_AV_baseline(idx_baseline).name;

    load(strcat(path,'/',dir_AV_baseline(idx_baseline).name))

    %% defining parameters for welch and multitaper method

    Nr_samples                    = size(preictal,1);
    Samples_per_sliding_window_PI = floor(Nr_samples/Nr_Sliding_windows);
    NR_samples_per_welch_window   = round(min(Samples_per_sliding_window_PI/10));

    big_spectrogram_preictal = zeros(size(preictal,2),2*Nr_Sliding_windows-1,513);

    %% welch and multitaper method

    for IDXC = 1:size(preictal,2)                      % channels to take

        fprintf('%d\n',IDXC)

        for IDXW = 1:(2*Nr_Sliding_windows-1)

            % welch --> PW
            % multitaper method --> MTM
            IDXS                             = (1:Samples_per_sliding_window_PI)+(IDXW-1)*floor(Samples_per_sliding_window_PI/2);
            data_vec                         = preictal(IDXS,IDXC);
            [~,f_hz]                         = pwelch(data_vec,NR_samples_per_welch_window,[],nfft,fs);
            spectrogram_preictal_MTM(IDXW,:) = pmtm(data_vec,nw,nfft);
        end

        % 50Hz removal

        [~,idx_to_take]   = min((50-f_hz).^2);

        Mask50Hz            = ones(size(spectrogram_preictal_MTM));
        Mask50Hz(:,idx_to_take-5:idx_to_take+5) = 0.* Mask50Hz(:,idx_to_take-5:idx_to_take+5);
        spectrogram_preictal_MTM = spectrogram_preictal_MTM.*Mask50Hz;

        AV_baseline_MTM  = AV_baseline(IDXC,:);
        spectrogram_preictal_MTM_Base_Cor_1 = spectrogram_preictal_MTM./repmat(AV_baseline_MTM,size(spectrogram_preictal_MTM,1),1);

        % feeding data into big spectrogram matrices
        big_spectrogram_preictal(IDXC,:,:) = spectrogram_preictal_MTM_Base_Cor_1;
    end

    disp('all channels done')

    big_spectrogram_preictal(:,:,idx_to_take-5:idx_to_take+5) = 0;     % to replace nans
    spectrogram_preictal = big_spectrogram_preictal;

    %% saving
    savename_preictal = '';  % new path and name go here
    save(savename_preictal,'patient_id','spectrogram_preictal','electrode_sets', 'AV_baseline_used');
end
