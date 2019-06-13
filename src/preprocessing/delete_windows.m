
% this is an example how to delete windows in a five minute interval
path = '';   % path goes here

dir = dir(strcat(path,'/','*mat'));
num_files = size(dir,1);            

for i = 1:num_files

    load(strcat(path,'/', dir(i).name))
    num_channels = size(spectrogram_baseline_1,1);

    dir_preictal(i).name
    size(electrode_sets.names,2)

    for IDXC = 1:6 %num_channels

        subplot(3,2,IDXC)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 7:12 %num_channels

        subplot(3,2,IDXC-6)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

   figure()
    for IDXC = 13:18 %num_channels

        subplot(3,2,IDXC-12)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 19:24 %num_channels

        subplot(3,2,IDXC-18)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 25:30 %num_channels

        subplot(3,2,IDXC-24)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 31:36 %num_channels

        subplot(3,2,IDXC-30)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 37:42 %num_channels

        subplot(3,2,IDXC-36)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end

    figure()
    for IDXC = 43:48 %num_channels

        subplot(3,2,IDXC-42)
        imagesc(flipud(squeeze(spectrogram_preictal(IDXC,:,:))'));
        set(gca,'Clim',[0 2*prctile(reshape(spectrogram_preictal(IDXC,:,:),1,[]),70)])
    end
end

windows_cut = 1;
spectrogram_baseline_1(:,1,:) = [];

repeat = spectrogram_baseline_1(:,size(spectrogram_baseline_1,2),:);
spectrogram_baseline_1 = [spectrogram_baseline_1 repeat];

savename = '';
save(savename, 'spectrogram_baseline_1', 'electrode_sets', 'patient_id', 'windows_cut')
