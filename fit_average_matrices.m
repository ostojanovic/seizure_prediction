
warning('off','all')

%% loading and setting the parameters

id = '109602';
patient_id = '109602';
run_numbers = [1:100];

baseline_path = strcat('/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_',id,'_extracted_seizures/data_baseline_',patient_id,'/');
preictal_path = strcat('/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/patient_',id,'_extracted_seizures/data_clinical_',patient_id,'/');

for run_nr = run_numbers

    AV_spectrogram_preictal = load(strcat(preictal_path,'AV_matrices/AV_spectrogram_preictal_',num2str(run_nr),'.mat'));
    AV_spectrogram_baseline = load(strcat(baseline_path,'AV_matrices/AV_spectrogram_baseline_',num2str(run_nr),'.mat'));

    num_periods_preictal = size(AV_spectrogram_preictal.AV_spectrogram_preictal,1);
    num_periods_baseline = size(AV_spectrogram_baseline.AV_spectrogram_baseline,1);

    all_spectrograms = [AV_spectrogram_preictal.AV_spectrogram_preictal; AV_spectrogram_baseline.AV_spectrogram_baseline];

    IDX_preictal = 1:num_periods_preictal;
    IDX_baseline = num_periods_preictal+1:num_periods_preictal+num_periods_baseline;

    num_periods   = size(all_spectrograms,1);
    num_channels  = size(all_spectrograms,2);
    num_windows   = size(all_spectrograms,3);

    time_order = 2;
    freq_order = 6;
    nnmf_order = 1;

    %% preprocessing

    temp = squeeze(all_spectrograms(:,1,:,:));     % all periods and its tf for one channel
    mean_Spec = squeeze(mean(all_spectrograms(:,1,:,:),1));
    kill_IDX = find(mean(mean_Spec)==0);
    all_spectrograms(:,:,:,kill_IDX) = [];       % get rid of the 50Hz component

    num_freq_bins = size(all_spectrograms,4);

    AV_W_preictal = zeros(num_channels, num_windows);
    AV_H_preictal = zeros(num_channels, num_freq_bins);

    AV_W_parameters_preictal = zeros(num_channels, time_order+1);
    AV_H_parameters_preictal = zeros(num_channels, freq_order+3);

    AV_W_baseline = zeros(num_channels, num_windows);
    AV_H_baseline = zeros(num_channels, num_freq_bins);

    AV_W_parameters_baseline = zeros(num_channels, time_order+1);
    AV_H_parameters_baseline = zeros(num_channels, freq_order+3);

    AV_Models_preictal = zeros(num_channels, num_windows, num_freq_bins);
    AV_Models_baseline = zeros(num_channels, num_windows, num_freq_bins);

    for IDXC = 1:num_channels

        fprintf('%d out of %d channels\n',IDXC,num_channels)

        temp = squeeze(all_spectrograms(:,IDXC,:,:));
        for IDX_f = 1:num_freq_bins     % frequency, now without 50Hz

            for IDX_time = 1:num_windows % time

                temp2d = temp(:,IDX_time,IDX_f);    % for all periods, specific time and frequency (for that 1 channel IDXC)

                YL = prctile(temp2d,10);                % lower pecentile; 10%
                YH = prctile(temp2d,90);                % upper pecentile; 90%

                [temp2ds,IDXSort] = sort(temp2d);     % sorting the elements in ascending order

                % The normal mean is dominated by outlieres. Here I get ride of
                % these and substitue the largest ones by he third larges and
                % the smallest. I do this per tf bin such that the data is not
                % corrupted for all tfs

                temp2d(temp2d==temp2ds(1))     = temp2ds(2);
                temp2d(temp2d==temp2ds(2))     = temp2ds(3);
                temp2d(temp2d==temp2ds(end))   = temp2ds(end-1);
                temp2d(temp2d==temp2ds(end-1)) = temp2ds(end-2);
                temp2dr(:,IDX_time,IDX_f)      = temp2d;    % for all periods, specific time and frequency (for that 1 channel IDXC)
                % but now without largest outliers
            end
        end

        % Compute the mean and median for eye inspection only
        % for preictal
        temp                 = temp2dr(IDX_preictal,:,:);
        mean_preictal_Spec   = squeeze(mean(temp,1));
        mean_preictal_SpecR  = squeeze(trimmean(temp,75,1)); % for 59002_1 should go 50% otherwise it gets back Nans
        median_preictal_Spec = squeeze(median(temp,1));

        % for baseline
        temp                 = temp2dr(IDX_baseline,:,:);
        mean_baseline_Spec   = squeeze(mean(temp,1));
        mean_baseline_SpecR  = squeeze(trimmean(temp,75,1));
        median_baseline_Spec = squeeze(median(temp,1));
        
        % save: mean_preictal_Spec, mean_preictal_SpecR, median_preictal_Spec 
        % for baseline and preictal, 
        % then do nmf in python

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
        [W_preictal,H_preictal] = nnmf(mean_preictal_SpecR,nnmf_order);       % cp1 (W) is a time component; cp2 (H) is a frequency component
        [W_model_preictal,W_parameters_preictal] = fit_polynominal(W_preictal,time_order);
        [H_model_preictal,H_parameters_preictal] = fit_splines(H_preictal',freq_order,0);

        IDXM3 = 0;
        for IDXM1 = 1:size(W_preictal,2)
            for IDXM2 = 1:size(H_preictal,1)
                IDXM3 = IDXM3 +1;
                Models_preictal(IDXM3,:,:) = (W_model_preictal(IDXM1,:)'*H_model_preictal(IDXM2,:));  % prototype tf model for preictal
            end
        end

        %     figure(1)
        %     subplot(2,2,1)
        %     plot(W_preictal);
        %     hold on
        %     plot(W_model_preictal,'--')
        %     title('Model of a time component for preictal state')
        %     legend('time comp', 'model','location', 'southeast')
        %     xlabel('time window')
        %
        %     subplot(2,2,2)
        %     plot(H_preictal');
        %     hold on
        %     plot(H_model_preictal,'--')
        %     title('Model of a frequency component for preictal state')
        %     legend('freq comp', 'model','location', 'southeast')
        %     xlabel('frequency bin')
        %
        %     figure(2)
        %     subplot(2,2,1)
        %     imagesc((W_preictal*H_preictal)')
        %     title('Prototype time-frequency image for preictal state')
        %     xlabel('time window')
        %     ylabel('frequency bin')
        %
        %     subplot(2,2,2)
        %     imagesc((W_model_preictal'*H_model_preictal)')
        %     title('Prototype time-frequency model for preictal state')
        %     xlabel('time window')
        %     ylabel('frequency bin')

        % baseline
        [W_baseline,H_baseline] = nnmf(mean_baseline_SpecR,nnmf_order);       % cp1 (W) is a time component; cp2 (H) is a frequency component
        [W_model_baseline,W_parameters_baseline] = fit_polynominal(W_baseline,time_order);
        [H_model_baseline,H_parameters_baseline] = fit_splines(H_baseline',freq_order,0);


        % Models_Baseline contains a single average model per periods for 1 IDXC channel
        IDXM3 = 0;
        for IDXM1 = 1:size(W_baseline,2)           % it's always 1, but IDXM1 is for model one == x_model
            for IDXM2 = 1:size(H_baseline,1)       % it's always 1, but IDXM2 is for two one == y_model
                IDXM3 = IDXM3 +1;           % it's always 1, but IDXM3 is for three one == Models_Baseline
                Models_Baseline(IDXM3,:,:) = (W_model_baseline(IDXM1,:)'*H_model_baseline(IDXM2,:));   % prototype tf model for baseline
            end
        end

        %     figure(1)
        %     subplot(2,2,3)
        %     plot(W_baseline);
        %     hold on
        %     plot(W_model_baseline,'--')
        %     title('Model of a time component for baseline state')
        %     legend('time comp', 'model','location', 'southeast')
        %     xlabel('time window')
        %
        %     subplot(2,2,4)
        %     plot(H_baseline');
        %     hold on
        %     plot(H_model_baseline,'--')
        %     title('Model of a frequency component for baseline state')
        %     legend('freq comp', 'model','location', 'southeast')
        %     xlabel('frequency bin')
        %
        %     figure(2)
        %     subplot(2,2,3)
        %     imagesc((W_baseline*H_baseline)')
        %     title('Prototype time-frequency image for baseline state')
        %     xlabel('time window')
        %     ylabel('frequency bin')
        %
        %     subplot(2,2,4)
        %     imagesc((W_model_baseline'*H_model_baseline)')
        %     title('Prototype time-frequency model for baseline state')
        %     xlabel('time window')
        %     ylabel('frequency bin')

        %% calculating correlation coefficient

        % Now you compare the prototype tf for preictal and baseline with the data for each period.
        % You find that the very simplified prototype makes already on the level of a single channel
        % a nearly perfect distingtion between preictal and baseline.

        for IDX_Period = 1:num_periods         % for all periods

            temp_2d = temp2dr(IDX_Period,:,:);      % temp_2d now has tf for a specific period taken from temp which is corrected for outliers

            % first row: correlataion between tf for all of the periods and tf for baseline
            % second row: correlataion between tf for all of the periods and tf for preictal

            for IDXM1 = 1:size(Models_Baseline,1)
                corrcoef(IDX_Period,IDXM1)                         = corr(reshape(temp_2d,1,[])',reshape(squeeze(Models_Baseline(IDXM1,:,:)),1,[])');
                corrcoef(IDX_Period,IDXM1+size(Models_Baseline,1)) = corr(reshape(temp_2d,1,[])',reshape(squeeze(Models_preictal(IDXM1,:,:)),1,[])');
            end
        end

        AV_W_preictal(IDXC,:) = W_preictal;
        AV_H_preictal(IDXC,:) = H_preictal;

        AV_W_parameters_preictal(IDXC,:) = W_parameters_preictal;
        AV_H_parameters_preictal(IDXC,:) = H_parameters_preictal;

        AV_W_baseline(IDXC,:) = W_baseline;
        AV_H_baseline(IDXC,:) = H_baseline;

        AV_W_parameters_baseline(IDXC,:) = W_parameters_baseline;
        AV_H_parameters_baseline(IDXC,:) = H_parameters_baseline;

        AV_Models_preictal(IDXC,:,:) = Models_preictal;
        AV_Models_baseline(IDXC,:,:) = Models_Baseline;

    end

    %% saving

    savename_baseline = strcat(baseline_path,'AV_models/AV_model_baseline_',num2str(run_nr),'.mat');
    savename_preictal = strcat(preictal_path,'AV_models/AV_model_preictal_',num2str(run_nr),'.mat');

    save(savename_baseline,'patient_id','run_nr','AV_W_baseline','AV_H_baseline','AV_W_parameters_baseline','AV_H_parameters_baseline','AV_Models_baseline')
    save(savename_preictal,'patient_id','run_nr','AV_W_preictal','AV_H_preictal','AV_W_parameters_preictal','AV_H_parameters_preictal','AV_Models_preictal')

end
