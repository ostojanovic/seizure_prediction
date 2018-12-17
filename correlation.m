
id = '109602';
patient_id = '109602';
sample = 'out-of-sample';       % 'train' or 'test' or 'out-of-sample'
run_nr = getenv('SGE_TASK_ID');

%% loading data and setting the parameters

path_directory = '/net/store/ni/projects/Data/intracranial_data/Freiburg_epilepsy_unit/';

baseline_directory = strcat('patient_',id,'_extracted_seizures/data_baseline_',patient_id,'/60-40/');
preictal_directory = strcat('patient_',id,'_extracted_seizures/data_clinical_',patient_id,'/');

dir_baseline = dir(strcat(path_directory,baseline_directory,'models_baseline/',num2str(run_nr),'/',sample,'/*.mat'));
dir_preictal = dir(strcat(path_directory,preictal_directory,'models_preictal2/',num2str(run_nr),'/',sample,'/*.mat'));

load((strcat(path_directory,baseline_directory,'AV_models/AV_model_baseline_',num2str(run_nr),'.mat')));
load((strcat(path_directory,preictal_directory,'AV_models2/AV_model_preictal_',num2str(run_nr),'.mat')));

num_channels = size(AV_Models_preictal,1);
num_periods_baseline = size(dir_baseline,1);
num_periods_preictal = size(dir_preictal,1);

class_labels_baseline = zeros(num_periods_baseline,1);
class_labels_preictal = ones(num_periods_preictal,1);

%% correlation coefficient between average baseline and individual preictal

corr_matrix_AV_baseline_preictal = zeros(num_periods_preictal,num_channels);

for IDXP = 1:num_periods_preictal         % for all periods

    load(strcat(preictal_directory,'models_preictal2/',num2str(run_nr),'/',sample,'/',dir_preictal(IDXP).name))

    for IDXC = 1:num_channels

        temp_preictal = squeeze(Models_preictal(IDXC,:,:));

        corr_matrix_AV_baseline_preictal(IDXP,IDXC) = corr(reshape(temp_preictal,1,[])',reshape(squeeze(AV_Models_baseline(IDXC,:,:)),1,[])');

    end
end

%% correlation coefficient between average baseline and individual baseline

corr_matrix_AV_baseline_baseline = zeros(num_periods_baseline,num_channels);

for IDXP = 1:num_periods_baseline         % for all periods

    load(strcat(baseline_directory,'models_baseline/',num2str(run_nr),'/',sample,'/',dir_baseline(IDXP).name))

    for IDXC = 1:num_channels

        temp_baseline = squeeze(Models_baseline(IDXC,:,:));

        corr_matrix_AV_baseline_baseline(IDXP,IDXC) = corr(reshape(temp_baseline,1,[])',reshape(squeeze(AV_Models_baseline(IDXC,:,:)),1,[])');

    end
end


%% correlation coefficient between average preictal and individual baseline

corr_matrix_AV_preictal_baseline = zeros(num_periods_baseline,num_channels);

for IDXP = 1:num_periods_baseline         % for all periods

    load(strcat(baseline_directory,'models_baseline/',num2str(run_nr),'/',sample,'/',dir_baseline(IDXP).name))

    for IDXC = 1:num_channels

        temp_baseline = squeeze(Models_baseline(IDXC,:,:));

        corr_matrix_AV_preictal_baseline(IDXP,IDXC) = corr(reshape(squeeze(AV_Models_preictal(IDXC,:,:)),1,[])',reshape(temp_baseline,1,[])');

    end
end

%% correlation coefficient between average preictal and individual preictal

corr_matrix_AV_preictal_preictal = zeros(num_periods_preictal,num_channels);

for IDXP = 1:num_periods_preictal         % for all periods

    load(strcat(preictal_directory,'models_preictal2/',num2str(run_nr),'/',sample,'/',dir_preictal(IDXP).name))

    for IDXC = 1:num_channels

        temp_preictal = squeeze(Models_preictal(IDXC,:,:));

        corr_matrix_AV_preictal_preictal(IDXP,IDXC) = corr(reshape(squeeze(AV_Models_preictal(IDXC,:,:)),1,[])',reshape(temp_preictal,1,[])');

    end
end

%% save

savename1 = strcat(path_directory,'patient_',id,'_extracted_seizures/','corr_coeff_',patient_id,'/60-40-2/',num2str(run_nr),'/',sample,'/corr_matrix_AV_baseline_baseline.mat');
savename2 = strcat(path_directory,'patient_',id,'_extracted_seizures/','corr_coeff_',patient_id,'/60-40-2/',num2str(run_nr),'/',sample,'/corr_matrix_AV_baseline_preictal.mat');
savename3 = strcat(path_directory,'patient_',id,'_extracted_seizures/','corr_coeff_',patient_id,'/60-40-2/',num2str(run_nr),'/',sample,'/corr_matrix_AV_preictal_baseline.mat');
savename4 = strcat(path_directory,'patient_',id,'_extracted_seizures/','corr_coeff_',patient_id,'/60-40-2/',num2str(run_nr),'/',sample,'/corr_matrix_AV_preictal_preictal.mat');

save(savename1,'patient_id','corr_matrix_AV_baseline_baseline','class_labels_baseline', 'run_nr')
save(savename2,'patient_id','corr_matrix_AV_baseline_preictal','class_labels_preictal', 'run_nr')
save(savename3,'patient_id','corr_matrix_AV_preictal_baseline','class_labels_baseline', 'run_nr')
save(savename4,'patient_id','corr_matrix_AV_preictal_preictal','class_labels_preictal', 'run_nr')
