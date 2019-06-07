
%% loading data
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_59002_extracted_seizures/59001102/data_baseline_59002_2/in-sample/';
%out-of-sample/';

patient_id = '59002_2';
dataset = 'train';               % 'test' or 'train'

% dir_baseline1 = dir(strcat(path_directory,base_directory,'baseline1_',patient_id,'/','*mat'));

dir_baseline1 = dir(strcat(path_directory,base_directory,'baseline1_',dataset,'_',patient_id,'/','*mat'));
num_files = size(dir_baseline1,1);

fs = 1024;           % !!!!!!!!!
Nr_Sliding_windows = 15;
nw   = round(16*(15/Nr_Sliding_windows));       % time-halfbandwidth product
nfft = 1024;                                    % number of dft points

i = str2num(getenv('SGE_TASK_ID'));

% for i=1:num_files

% fprintf('%d\n',i);

% load(strcat(path_directory,base_directory,'baseline1_',patient_id,'/',dir_baseline1(i).name));
load(strcat(path_directory,base_directory,'baseline1_',dataset,'_',patient_id,'/',dir_baseline1(i).name));
baseline_1 = first_half;
clear first_half;

%% defining parameters for welch and multitaper method

Nr_samples                    = size(baseline_1,1);
Samples_per_sliding_window_B1 = floor(Nr_samples/Nr_Sliding_windows);
NR_samples_per_welch_window   = round(min(Samples_per_sliding_window_B1/10));

big_spectrogram_baseline_1 = zeros(size(baseline_1,2),2*Nr_Sliding_windows-1,513);
AV_baseline = zeros(size(baseline_1,2),513);

%% welch and multitaper method

for IDXC = 1:size(baseline_1,2)                   % channels to take

    for IDXW=1:(2*Nr_Sliding_windows-1)

        % welch --> PW
        % multitaper method --> MTM
        IDXS                               = (1:Samples_per_sliding_window_B1)+(IDXW-1)*floor(Samples_per_sliding_window_B1/2);
        data_vec                           = baseline_1(IDXS,IDXC);
        [~,f_hz]                           = pwelch(data_vec,NR_samples_per_welch_window,[],nfft,fs);
        spectrogram_baseline_1_MTM(IDXW,:) = pmtm(data_vec,nw,nfft);
    end

    % 50Hz removal
    [~,idx_to_take] = min((50-f_hz).^2);

    Mask50Hz = ones(size(spectrogram_baseline_1_MTM));
    Mask50Hz(:,idx_to_take-5:idx_to_take+5) = 0.* Mask50Hz(:,idx_to_take-5:idx_to_take+5);
    spectrogram_baseline_1_MTM = spectrogram_baseline_1_MTM.*Mask50Hz;

    AV_baseline_MTM  = mean(spectrogram_baseline_1_MTM);
    AV_baseline(IDXC,:) = AV_baseline_MTM;

    spectrogram_baseline_1_MTM_Base_Cor_1 = spectrogram_baseline_1_MTM./repmat(AV_baseline_MTM,size(spectrogram_baseline_1_MTM,1),1);

    % feeding data into big spectrogram matrices
    big_spectrogram_baseline_1(IDXC,:,:) = spectrogram_baseline_1_MTM_Base_Cor_1;
end

disp('all channels done')

big_spectrogram_baseline_1(:,:,idx_to_take-5:idx_to_take+5) = 0;     % to replace nans
spectrogram_baseline_1 = big_spectrogram_baseline_1;

%     for IDXCH=1:11
%         subplot(3,4,IDXCH)
%         test = (squeeze(big_spectrogram_baseline_1(IDXCH,:,:)))
%         imagesc(test')
%         colorbar
%     end
%     subplot(3,4,12)
%     test = squeeze(mean(big_spectrogram_baseline_1,1)); %(squeeze(big_spectrogram_baseline_1(IDXCH,:,:)))
%     imagesc(test')
%     colorbar

%% saving part
part_temp = strsplit(dir_baseline1(i).name,'_baseline_');
name_part  = strsplit(part_temp{2},'_1st');

% savename_baseline_1 = strcat(path_directory,base_directory, 'spectrograms_baseline1_',patient_id,'/spectrogram_baseline1_',name_part{1});
% savename_AV_baseline = strcat(path_directory,base_directory,'spectrograms_AV_baseline1_',patient_id,'/AV_spectrogram_',name_part{1});

savename_baseline_1 = strcat(path_directory,base_directory, 'spectrograms_baseline1_',dataset,'/spectrogram_baseline1_',name_part{1});
savename_AV_baseline = strcat(path_directory,base_directory,'spectrograms_AV_baseline_',dataset,'/AV_spectrogram_',name_part{1});

save(savename_AV_baseline,'AV_baseline');
save(savename_baseline_1,'patient_id','spectrogram_baseline_1','electrode_sets');

% end
