function param = MDSA(param)
%multi-domain subspace alignment (MDSA)

fprintf('Subspace alignment between multiple domains\n');
mdsa_transmat_file_name = ['MDSA_trans_' param.feat_mode '.mat'];
if exist(mdsa_transmat_file_name, 'file') == 2
    load(mdsa_transmat_file_name);
    param.Xs = Xs; param.Xt = Xt; param.Xm = Xm;
    param.SA_meanS = SA_meanS; param.SA_stdS = SA_stdS;
    param.SA_meanT = SA_meanT; param.SA_stdT = SA_stdT;
    param.SA_meanM = SA_meanM; param.SA_stdM = SA_stdM;
    return
end
%% load the training data for MDSA
addrS = [param.pathMDSA 'IR-Domain\']; 
addrT = [param.pathMDSA 'VS-Domain\'];
addrM = param.pathVOC;
%% augment data with geometric transformations
fprintf('dataAug... ');
augS = dataAug(addrS, 'aug'); 
augT = dataAug(addrT, 'aug');
augM = dataAug(addrM, 'no aug');
%% random crop the images to have ~35,000 samples in each domain
fprintf('randCrop... ');
[augS, augT, augM] = randCrop(augS, augT, augM);
%% extract features
fprintf('ftsExt...');
Src = featExt_MDSA(augS, param);
Tgt = featExt_MDSA(augT, param);
Mid = featExt_MDSA(augM, param);
%% MDSA
% whitening
[Src, param.SA_meanS, param.SA_stdS] = zStandard(Src);
[Tgt, param.SA_meanT, param.SA_stdT] = zStandard(Tgt);
[Mid, param.SA_meanM, param.SA_stdM] = zStandard(Mid);
% PCA
Xss = pca(Src); Xs = Xss(:,1:param.pcaD);
Xtt = pca(Tgt); Xt = Xtt(:,1:param.pcaD);
Xmm = pca(Mid); Xm = Xmm(:,1:param.pcaD);
param.Xs = Xs*Xs'*Xm; % Template-FSNeg
param.Xt = Xt*Xt'*Xm; % Target-FSNeg
param.Xm = Xm;
clear Xs Xt Xm;

Xs = param.Xs; Xt = param.Xt; Xm = param.Xm;
[SA_meanS, SA_stdS, SA_meanT, SA_stdT, SA_meanM, SA_stdM] = deal(...
    param.SA_meanS, param.SA_stdS, ...
    param.SA_meanT, param.SA_stdT, ...
    param.SA_meanM, param.SA_stdM);
save(mdsa_transmat_file_name, 'Xs', 'Xt', 'Xm', 'SA_meanS', 'SA_stdS', 'SA_meanT', 'SA_stdT', 'SA_meanM', 'SA_stdM');
end

function [output, mean_input, std_input] = zStandard(input)
% ZCA whitening

mean_input = mean(input,1); 
std_input = std(input);
output = (input - repmat(mean_input, size(input,1), 1))./repmat(std_input, size(input,1), 1);

end

function feat = featExt_MDSA(augI, param)
feat_tmp = featExt(augI{1}, param);
feat = zeros(length(augI), size(feat_tmp, 2));
for i = 1:length(augI)
    feat(i,:) = featExt(augI{i}, param);  
end
feat = L2_norm(feat);
end

function [croppedS, croppedT, croppedM] = randCrop(augS, augT, augM)
% data should be cells of imgs
% imgS and imgS must be larger than tmpl both in width and height by at
% least tmpl size.

nCrop_ST = round(35000/numel(augS)); 
nCrop_M = round(35000/numel(augM)); 
rT = 50; cT = 50;
for i = 1:numel(augS)
    imgS = augS{i}; 
    imgT = augT{i}; 
    randrow = randperm(size(imgS,1)-rT, nCrop_ST);
    randcol = randperm(size(imgS,2)-cT, nCrop_ST);
    for j = 1:nCrop_ST
        [r, c] = deal(randrow(j), randcol(j));
        croppedS{nCrop_ST*(i-1)+j} = imgS(r:r+rT-1, c:c+cT-1, :);
        croppedT{nCrop_ST*(i-1)+j} = imgT(r:r+rT-1, c:c+cT-1, :);
    end
end

count = 0;
for i = 1:numel(augM)
    imgM = augM{i};
    if (size(imgM,1)-rT) < 0 || (size(imgM,2)-cT) < 0
        continue
    end
    count = count + 1;
    randrow = randperm(size(imgM,1)-rT, nCrop_M);
    randcol = randperm(size(imgM,2)-cT, nCrop_M);
    for j = 1:nCrop_M
        [r, c] = deal(randrow(j), randcol(j));
        croppedM{nCrop_M*(count-1)+j} = imgM(r:r+rT-1, c:c+cT-1, :);
    end
end
end