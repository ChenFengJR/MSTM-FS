function mstm_fs_main()
% Official code for 
%   Chen Feng et al., "Multi-spectral template matching based object detection
%   in a few-shot learning manner," Inf. Sci. 2023.
% -------------------------------------
% Author: Chen Feng
% School: School of Automation in Huazhong Univerity of Science and Technology
% Copyright all reserved
% ------------------------------------- 

% set parameters
test_folder = '1-vs-nir'; % '1-vs-nir' or '2-vs-lwir'
feat_mode = 'HOG'; % RawPixel, HOG, or CNN

% load parameters
param = parameter_setting(test_folder, feat_mode);

% learn multi-domain subspace alignment
param = MDSA(param);

bndboxLoc = [];
for tmpl_i = 1:length(param.tmpl_name)
    fprintf('test case %03d\n', tmpl_i);
    % learn template-wise metric for each case
    tmpl_name = param.tmpl_name(tmpl_i).name;
    tmpl = imread([param.pathTmpl tmpl_name]);
    [t_h, t_w, ~] = size(tmpl);
    tmpl = L2_norm(featExt(tmpl, param)); 
    tmpl = (tmpl - param.SA_meanS)./param.SA_stdS * param.Xs;
    tmpl = L2_norm(tmpl);
    param.tmpl = mstm_fs_svm(param, tmpl);
    sear = imread([param.pathSear tmpl_name]);
    % start test
    bndboxLoc_ = mstm_fs_detection(sear, t_h, t_w, param);
    bndboxLoc = [bndboxLoc; bndboxLoc_];
end
mAP = cal_mAP(bndboxLoc, param);
fprintf('mAP for %s: %.4f', test_folder, mAP);
end

