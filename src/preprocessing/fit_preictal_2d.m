function [W_preictal,H_preictal,W_model_preictal,H_model_preictal,W_parameters_preictal,H_parameters_preictal,Models_preictal] = fit_preictal_2d(path_directory,preictal_directory,files,num_files,num_channels,num_windows,sample,run_nr)

%% loading and setting the parameters
time_order = 2;
freq_order = 6;
nnmf_order = 2;

load(strcat(path_directory,preictal_directory,'spectrograms_preictal/',files(1).name))
mean_Spec = mean(spectrogram_preictal(1,:,:),1);
kill_IDX = find(mean(mean_Spec)==0);

num_freq_bins = size(mean_Spec,3)-size(kill_IDX,1);

W_preictal = zeros(num_channels, num_windows, nnmf_order);
H_preictal = zeros(num_channels, num_freq_bins, nnmf_order);

W_model_preictal = zeros(num_channels, num_windows, nnmf_order);
H_model_preictal = zeros(num_channels, num_freq_bins, nnmf_order);

W_parameters_preictal = zeros(num_channels, time_order+1, nnmf_order);
H_parameters_preictal = zeros(num_channels, freq_order+3, nnmf_order);

Models_preictal = zeros(num_channels, num_windows, num_freq_bins, nnmf_order+nnmf_order);

%% preprocessing

for i = 1:num_files

    fprintf('%d out of %d periods\n',i,num_files)

    load(strcat(path_directory,preictal_directory,'spectrograms_preictal/',files(i).name))

    spectrogram_preictal(:,:,kill_IDX) = [];       % get rid of the 50Hz component

    for IDXC = 1:num_channels

        temp = spectrogram_preictal(IDXC,:,:);

        % Compute the mean and median for eye inspection only
        % for preictal
        mean_preictal_Spec   = squeeze(mean(temp,1));
        mean_preictal_SpecR  = squeeze(trimmean(temp,75,1));
        median_preictal_Spec = squeeze(median(temp,1));

        %% nnmf and modelling

        % Do the nnmf for the two cases.
        % Important is that time is modeled only with second order - that makes
        % interpretation very easy.
        % The fre is modelled with non linerly spaces cubic splines to consider the
        % frequency resolution that deceases for higher freqs.
        % The results is a prototype TF image based on the modelled nnmf.

        % The results is a prototype TF image based on the modelled nnmf.
        % If you have nnmf_order=1 you get one model.
        % For nnmf_order=2 you get 4. I have the feeling 1 works good; 4 doesn't.

        % preictal
        [w_preictal,h_preictal] = nnmf(mean_preictal_SpecR,nnmf_order);       % cp1 (W) is a time component; cp2 (H) is a frequency component
        [w_model_preictal,w_parameters_preictal] = fit_polynominal(w_preictal,time_order);
        [h_model_preictal,h_parameters_preictal] = fit_splines(h_preictal',freq_order,0);

        % Models_Preictal contains a single average model per periods for 1 IDXC channel
        IDXM3 = 0;
        for IDXM1 = 1:size(w_preictal,2)           % it's always 1, but IDXM1 is for model one == x_model
            for IDXM2 = 1:size(h_preictal,1)       % it's always 1, but IDXM2 is for two one == y_model
                IDXM3 = IDXM3 +1;           % it's always 1, but IDXM3 is for three one == Models_Preictal
                Models_Preictal(IDXM3,:,:) = (w_model_preictal(IDXM1,:)'*h_model_preictal(IDXM2,:));   % prototype tf model for baseline
            end
        end

        W_preictal(IDXC,:,:) = w_preictal;
        H_preictal(IDXC,:,:) = h_preictal';

        W_model_preictal(IDXC,:,:) = w_model_preictal';
        H_model_preictal(IDXC,:,:) = h_model_preictal';

        W_parameters_preictal(IDXC,:,:) = w_parameters_preictal';
        H_parameters_preictal(IDXC,:,:) = h_parameters_preictal';

        Models_preictal(IDXC,:,:,:) = permute(Models_Preictal,[2,3,1]);

    end

    %% saving

    savename_part = strsplit(files(i).name,'spectrogram_');

    savename_preictal = strcat(path_directory,preictal_directory,'2dim/',run_nr,'/',sample,'/Model_',savename_part{2});
    save(savename_preictal,'sample','run_nr','patient_id','W_preictal','H_preictal','W_model_preictal','H_model_preictal','W_parameters_preictal','H_parameters_preictal','Models_preictal')

end
