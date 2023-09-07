function param = parameter_setting(test_folder, feat_mode)
% data paths
param.root = pwd;
param.pathTmpl = strcat(param.root, '\data\', test_folder, '\template\');
param.pathSear = strcat(param.root, '\data\', test_folder, '\search-image\');
param.pathMDSA = strcat(param.root, '\data\', test_folder, '\UDA-SA-training\');
param.pathVOC = strcat(param.root, '\data\', '\pascal-voc\');
dir_data = dir(strcat(param.pathTmpl, '\*.bmp'));
param.tmpl_name = dir_data;

% gt
load(strcat(param.root, '\data\', test_folder, '\gt-result\', test_folder, '.mat'));
param.bndboxLocGT = bndboxLoc;

% svm
param.hnm_threshold_score = -1.00000; % threshold for keeping HNs
param.train_max_mine_iterations = 1;  
param.train_max_images_per_iteration = 200000;
param.train_keep_nsv_multiplier = 3; 
param.train_max_mined_images = 200000; 
param.train_svm_c = .01; 
param.pos_cof = 50; 
param.HNM_1st_iteration_Imgs = 50;  

param.nms_overlap = 0.5;

param.feat_mode = feat_mode; % RawPixel, HOG, or CNN
switch param.feat_mode  % pca dimension
    case 'RawPixel', param.pcaD = 2500;
    case 'HOG', param.pcaD = 900;
    case 'CNN'
        param.pcaD = 1024;
        net = load(strcat(param.root, '\imagenet-vgg-verydeep-16.mat')); 
        param.net = vl_simplenn_tidy(net);
end
end