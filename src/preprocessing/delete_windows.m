
path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';
base_directory = 'patient_25302_extracted_seizures/25301102/data_baseline_25302_2/out-of-sample/spectrograms_baseline1_25302_2';
%in-sample/spectrograms_baseline1_train';

dir_preictal = dir(strcat(path_directory,base_directory,'/','*mat'));
num_files = size(dir_preictal,1);               % change the name down there !!!!!!!!!!!!!

for i = 3%1:num_files

    load(strcat(path_directory,base_directory,'/', dir_preictal(i).name))
    num_channels = size(spectrogram_baseline_1,1);

    dir_preictal(i).name
    size(electrode_sets.names,2)

%     for IDXC = 1:6 %num_channels
%
%         subplot(3,2,IDXC)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end

%     figure()
%     for IDXC = 7:12 %num_channels
%
%         subplot(3,2,IDXC-6)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%    figure()
%     for IDXC = 13:18 %num_channels
%
%         subplot(3,2,IDXC-12)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%     figure()
%     for IDXC = 19:24 %num_channels
%
%         subplot(3,2,IDXC-18)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%     figure()
%     for IDXC = 25:30 %num_channels
%
%         subplot(3,2,IDXC-24)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%     figure()
%     for IDXC = 31:36 %num_channels
%
%         subplot(3,2,IDXC-30)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%     figure()
%     for IDXC = 37:42 %num_channels
%
%         subplot(3,2,IDXC-36)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
%
%     figure()
%     for IDXC = 43:48 %num_channels
%
%         subplot(3,2,IDXC-42)
%         imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
%         set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
%     end
end

windows_cut = 1;
% spectrogram_baseline_1(:,1,:) = [];
%
% repeat = spectrogram_baseline_1(:,size(spectrogram_baseline_1,2),:);
% spectrogram_baseline_1 = [spectrogram_baseline_1 repeat];
%
% savename = strcat(path_directory,base_directory,'/spectrogram_baseline1_59_edited.mat');
% save(savename, 'spectrogram_baseline_1', 'electrode_sets', 'patient_id', 'windows_cut')
