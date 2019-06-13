function [W_preictal,H_preictal,W_model_preictal,H_model_preictal,W_parameters_preictal,H_parameters_preictal,Models_preictal] = fit_preictal_components(path,files,num_files,num_channels,num_windows)

%% loading and setting the parameters
time_order = 2;
freq_order = 6;
nnmf_order = 1;

load(strcat(path,'spectrograms_preictal/',files(1).name))
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

    load(strcat(path,'spectrograms_preictal/',files(i).name))

    spectrogram_preictal(:,:,kill_IDX) = [];       % get rid of the 50Hz component

    for IDXC = 1:num_channels

        temp = spectrogram_preictal(IDXC,:,:);
        mean_preictal_SpecR  = squeeze(trimmean(temp,75,1));

        [w_preictal,h_preictal] = nnmf(mean_preictal_SpecR,nnmf_order);       % W is a time component; H is a frequency component
        [w_model_preictal,w_parameters_preictal] = fit_polynominal(w_preictal,time_order);
        [h_model_preictal,h_parameters_preictal] = fit_splines(h_preictal',freq_order,0);

        IDXM3 = 0;
        for IDXM1 = 1:size(w_preictal,2)
            for IDXM2 = 1:size(h_preictal,1)
                IDXM3 = IDXM3 +1;           
                Models_Preictal(IDXM3,:,:) = (w_model_preictal(IDXM1,:)'*h_model_preictal(IDXM2,:));  
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
    savename_preictal = '';     % new path and name go here
    save(savename_preictal,'patient_id','W_preictal','H_preictal','W_model_preictal','H_model_preictal','W_parameters_preictal','H_parameters_preictal','Models_preictal')

end
