
warning('off','all')
sample = 0; % 1 for 'in-sample'; 0 for 'out-of-sample'

%% loading and setting the parameters
dataset = 'test';
patient_id = '25302_2';

directory_path = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';

if sample == 1
    clinical_path = 'patient_25302_extracted_seizures/25301102/data_clinical_25302_2/in-sample/';
elseif sample == 0
    clinical_path = 'patient_25302_extracted_seizures/25301102/data_clinical_25302_2/out-of-sample/';
end

if sample == 1
    dir_clinical = dir(strcat(directory_path,clinical_path,'spectrograms_preictal_',dataset,'/','*.mat'));
elseif sample == 0
    dir_clinical = dir(strcat(directory_path,clinical_path,'spectrograms_preictal_',patient_id,'/','*.mat'));
end

num_periods = size(dir_clinical,1);

if sample == 1
    load(strcat(directory_path,clinical_path,'spectrograms_preictal_',dataset,'/',dir_clinical(1).name))
elseif sample == 0
    load(strcat(directory_path,clinical_path,'spectrograms_preictal_',patient_id,'/',dir_clinical(1).name))
end

num_channels = size(spectrogram_preictal,1);
num_windows  = size(spectrogram_preictal,2);

time_order = 2;
freq_order = 6;
nnmf_order = 1;

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

for i = 1:num_periods

    fprintf('%d out of %d periods\n',i,num_periods)

    if sample == 1
        load(strcat(directory_path,clinical_path,'spectrograms_preictal_',dataset,'/',dir_clinical(i).name))
    elseif sample == 0
        load(strcat(directory_path,clinical_path,'spectrograms_preictal_',patient_id,'/',dir_clinical(i).name))
    end

    spectrogram_preictal(:,:,kill_IDX) = [];       % get rid of the 50Hz component

    for IDXC = 1:num_channels

        fprintf('%d out of %d channels\n',IDXC,num_channels)

        temp = spectrogram_preictal(IDXC,:,:);
%         for IDX_f = 1:num_freq_bins     % frequency, now without 50Hz
%
%             for IDX_time = 1:num_windows % time
%
%                 temp2d = temp(:,IDX_time,IDX_f);    % for all periods, specific time and frequency (for that 1 channel IDXC)
%
%                 YL = prctile(temp2d,10);                % lower pecentile; 10%
%                 YH = prctile(temp2d,90);                % upper pecentile; 90%
%
%                 [temp2ds,IDXSort] = sort(temp2d);     % sorting the elements in ascending order
%
%                 % The normal mean is dominated by outlieres. Here I get ride of
%                 % these and substitue the largest ones by he third larges and
%                 % the smallest. I do this per tf bin such that the data is not
%                 % corrupted for all tfs
%
%                 temp2d(temp2d==temp2ds(1))     = temp2ds(2);
%                 temp2d(temp2d==temp2ds(2))     = temp2ds(3);
%                 temp2d(temp2d==temp2ds(end))   = temp2ds(end-1);
%                 temp2d(temp2d==temp2ds(end-1)) = temp2ds(end-2);
%                 temp2dr(:,IDX_time,IDX_f)      = temp2d;    % for all periods, specific time and frequency (for that 1 channel IDXC)
%                 % but now without largest outliers
%             end
%         end

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

%         figure(1)
%         subplot(2,2,1)
%         plot(w_preictal);
%         hold on
%         plot(w_model_baseline,'--')
%         title('Model of a time component for baseline state')
%         legend('time comp', 'model','location', 'southeast')
%         xlabel('time window')
%
%         subplot(2,2,2)
%         plot(h_baseline');
%         hold on
%         plot(h_model_baseline,'--')
%         title('Model of a frequency component for baseline state')
%         legend('freq comp', 'model','location', 'southeast')
%         xlabel('frequency bin')
%
%         subplot(2,2,3)
%         imagesc((w_baseline*h_baseline)')
%         title('Prototype time-frequency image for baseline state')
%         xlabel('time window')
%         ylabel('frequency bin')
%
%         subplot(2,2,4)
%         imagesc((w_model_baseline'*h_model_baseline)')
%         title('Prototype time-frequency model for baseline state')
%         xlabel('time window')
%         ylabel('frequency bin')

        W_preictal(IDXC,:) = w_preictal';
        H_preictal(IDXC,:) = h_preictal;

        W_model_preictal(IDXC,:) = w_model_preictal;
        H_model_preictal(IDXC,:) = h_model_preictal;

        W_parameters_preictal(IDXC,:) = w_parameters_preictal;
        H_parameters_preictal(IDXC,:) = h_parameters_preictal;

        Models_preictal(IDXC,:,:) = Models_Preictal;


    end

    %% saving

    savename_part = strsplit(dir_clinical(i).name,'spectrogram_');

    if sample == 1
        savename_preictal = strcat(directory_path,clinical_path,'models_preictal_',dataset,'/Model_',savename_part{2});
    elseif sample == 0
        savename_preictal = strcat(directory_path,clinical_path,'models_preictal_',patient_id,'/Model_',savename_part{2});
    end

    save(savename_preictal,'patient_id','W_preictal','H_preictal','W_model_preictal','H_model_preictal','W_parameters_preictal','H_parameters_preictal','Models_preictal')

end
