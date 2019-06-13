function [W_interictal,H_interictal,W_model_interictal,H_model_interictal,W_parameters_interictal,H_parameters_interictal,Models_interictal] = fit_interictal_components(path,files,num_files,num_channels,num_windows)

%% loading and setting the parameters
time_order = 2;
freq_order = 6;
nnmf_order = 1;

load(strcat(path,'spectrograms_interictal/',files(1).name)) % for getting indices of freq.bins of 50Hz component
mean_Spec = mean(spectrogram_interictal_1(1,:,:),1);
kill_IDX = find(mean(mean_Spec)==0);

num_freq_bins = size(mean_Spec,3)-size(kill_IDX,1);

W_interictal = zeros(num_channels, num_windows);
H_interictal = zeros(num_channels, num_freq_bins);

W_model_interictal = zeros(num_channels, num_windows);
H_model_interictal = zeros(num_channels, num_freq_bins);

W_parameters_interictal = zeros(num_channels, time_order+1);
H_parameters_interictal = zeros(num_channels, freq_order+3);

Models_interictal = zeros(num_channels, num_windows, num_freq_bins);

%% preprocessing

for i = 1:num_files

    fprintf('%d out of %d periods \n',i,num_files)

    load(strcat(path,'spectrograms_interictal/',files(i).name))
    spectrogram_interictal_1(:,:,kill_IDX) = [];       % get rid of the 50Hz component

    for IDXC = 1:num_channels

        temp = spectrogram_interictal_1(IDXC,:,:);
        mean_interictal_SpecR  = squeeze(trimmean(temp,75,1));

        %% nnmf and modelling

        [w_interictal,h_interictal] = nnmf(mean_interictal_SpecR,nnmf_order);       % W is a time component; H is a frequency component
        [w_model_interictal,w_parameters_interictal] = fit_polynominal(w_interictal,time_order);
        [h_model_interictal,h_parameters_interictal] = fit_splines(h_interictal',freq_order,0);

        IDXM3 = 0;
        for IDXM1 = 1:size(w_interictal,2)
            for IDXM2 = 1:size(h_interictal,1)
                IDXM3 = IDXM3 +1;            
                Models_interictal(IDXM3,:,:) = (w_model_interictal(IDXM1,:)'*h_model_interictal(IDXM2,:));   
            end
        end

        W_interictal(IDXC,:) = w_interictal';
        H_interictal(IDXC,:) = h_interictal;

        W_model_interictal(IDXC,:) = w_model_interictal;
        H_model_interictal(IDXC,:) = h_model_interictal;

        W_parameters_interictal(IDXC,:) = w_parameters_interictal;
        H_parameters_interictal(IDXC,:) = h_parameters_interictal;

        Models_interictal(IDXC,:,:) = Models_interictal;

    end

    %% saving
    savename_interictal = '';   % new path and name go here
    save(savename_interictal,'patient_id','W_interictal','H_interictal','W_model_interictal','H_model_interictal','W_parameters_interictal','H_parameters_interictal','Models_interictal')

end

end
