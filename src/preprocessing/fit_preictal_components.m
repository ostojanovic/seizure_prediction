function [W_preictal,H_preictal,W_model_preictal,H_model_preictal,W_parameters_preictal,H_parameters_preictal,Models_preictal] = fit_preictal_components(path_directory,preictal_directory,files,num_files,num_channels,num_windows,sample,run_nr)

%% loading and setting the parameters
time_order = 2;
freq_order = 6;
nnmf_order = 1;

load(strcat(path_directory,preictal_directory,'spectrograms_preictal/',files(1).name))
mean_Spec = mean(spectrogram_preictal(1,:,:),1);
kill_IDX = find(mean(mean_Spec)==0);

num_freq_bins = size(mean_Spec,3)-size(kill_IDX,1);

W_preictal = zeros(num_channels, num_windows);
H_preictal = zeros(num_channels, num_freq_bins);

W_model_preictal = zeros(num_channels, num_windows);
H_model_preictal = zeros(num_channels, num_freq_bins);

W_parameters_preictal = zeros(num_channels, time_order+1);
H_parameters_preictal = zeros(num_channels, freq_order+3);

Models_preictal = zeros(num_channels, num_windows, num_freq_bins);

%% preprocessing

for i = 1:num_files

    fprintf('%d out of %d periods\n',i,num_files)

    load(strcat(path_directory,preictal_directory,'spectrograms_preictal/',files(i).name))

    spectrogram_preictal(:,:,kill_IDX) = [];       % get rid of the 50Hz component

    for IDXC = 1:num_channels

        temp = spectrogram_preictal(IDXC,:,:);

        % Compute the mean and median for eye inspection only
        mean_preictal_Spec   = squeeze(mean(temp,1));
        mean_preictal_SpecR  = squeeze(trimmean(temp,75,1));
        median_preictal_Spec = squeeze(median(temp,1));

        % preictal
        [w_preictal,h_preictal] = nnmf(mean_preictal_SpecR,nnmf_order);       % cp1 (W) is a time component; cp2 (H) is a frequency component
        [w_model_preictal,w_parameters_preictal] = fit_polynominal(w_preictal,time_order);
        [h_model_preictal,h_parameters_preictal] = fit_splines(h_preictal',freq_order,0);

        IDXM3 = 0;
        for IDXM1 = 1:size(w_preictal,2)
            for IDXM2 = 1:size(h_preictal,1)
                IDXM3 = IDXM3 +1;           
                Models_Preictal(IDXM3,:,:) = (w_model_preictal(IDXM1,:)'*h_model_preictal(IDXM2,:));   % prototype tf model
            end
        end

        W_preictal(IDXC,:) = w_preictal';
        H_preictal(IDXC,:) = h_preictal;

        W_model_preictal(IDXC,:) = w_model_preictal;
        H_model_preictal(IDXC,:) = h_model_preictal;

        W_parameters_preictal(IDXC,:) = w_parameters_preictal;
        H_parameters_preictal(IDXC,:) = h_parameters_preictal;

        Models_preictal(IDXC,:,:) = Models_Preictal;

    end

    %% saving
    savename_part = strsplit(files(i).name,'spectrogram_');
    savename_preictal = strcat(path_directory,preictal_directory,'models_preictal/',run_nr,'/',sample,'/Model_',savename_part{2});
    save(savename_preictal,'sample','run_nr','patient_id','W_preictal','H_preictal','W_model_preictal','H_model_preictal','W_parameters_preictal','H_parameters_preictal','Models_preictal')

end
