function model_w = mstm_fs_svm(param, tmpl)
% mstm_fs_svm for feature selection

%% Obtain negative features
fprintf('Obtaining negSet... ');
neg_set_fts_path = [param.root '\voc-feat\voc-feat-' param.feat_mode '.mat'];
if exist(neg_set_fts_path, 'file') == 2
    load(neg_set_fts_path);
else
    dir_path = dir([param.pathVOC '*.bmp']);
    negSet_voc = arrayfun(@(x)(rgb2gray(imread([param.pathVOC dir_path(x).name]))),...
        1:length(dir_path), 'un', 0);
    feat_tmp = featExt(negSet_voc{1}, param);
    voc_feat = zeros(length(negSet_voc), size(feat_tmp, 2));
    for i = 1:length(negSet_voc)
        voc_feat(i,:) = featExt(negSet_voc{i}, param);  
    end
    save(neg_set_fts_path, 'voc_feat');
end

voc_feat = L2_norm(voc_feat);

%% MDSA
voc_feat = (voc_feat - repmat(param.SA_meanM, size(voc_feat, 1), 1))./...
    repmat(param.SA_stdM, size(voc_feat, 1), 1) * param.Xm;
voc_feat = L2_norm(voc_feat);

param.voc_feat = voc_feat;

%% exemplar svm
param.model.wtrace{1} = tmpl';  % initial weight; one-column vector
param.model.x = tmpl; % template feature
param.model.btrace{1} = 0;

% hard negative mining
keep_going = 1;
param.iteration = 0; 
param.total_mines = 0; % toal mined images 

while keep_going == 1
  %Get the name of the next chunk file to write
  param = ESVM_train_iteration(param); % training_function : esvm_update_svm
  % prepare enough mining samples for training
  if ((param.total_mines >= param.train_max_mined_images) || ...
           (param.iteration == param.train_max_mine_iterations))
      keep_going = 0;      
  end
  
  if keep_going==0
      break;
  end
  param.iteration = param.iteration + 1;
end 
model_w = param.model.wtrace{end};
end

function tr_para = ESVM_train_iteration(tr_para)

% mining: output rs (rs.x for feature; rs.s for score)
tr_para = ESVM_minen(tr_para); % the following iteration
supery = [ones(size(tr_para.model.x,1),1); -ones(size(tr_para.model.svxs,1),1)];
newx = double([tr_para.model.x; tr_para.model.svxs]);
% update the model; preserve every w when updating 
svm_model = svmtrain(supery, newx, sprintf(['-s 0 -t 0 -c' ...
                    ' %f -w1 %.9f -q'], tr_para.train_svm_c, tr_para.pos_cof));
svm_weights = full(sum(svm_model.SVs .* repmat(svm_model.sv_coef,1, ...
                                size(svm_model.SVs,2)),1));   
w = svm_weights'; tr_para.model.wtrace{end+1} = w;  % w: column vector
b = svm_model.rho; tr_para.model.btrace{end+1} = b;
r = tr_para.model.svxs*w-b;  
svs = find(r >= -1.0000);  % hard negatives are negatives in 'twixt Margin; svs = find(abs(r) <= -1.0000); 
[ss, bb] = sort(r, 'descend');
total_length = min(tr_para.train_keep_nsv_multiplier*length(svs),length(bb));
tr_para.model.svxs = tr_para.model.svxs(bb(1:total_length),:);
fprintf(1, 'Negatives within the margin: %d, max score = %.3f || ', total_length, ss(1));
fprintf(1, 'Support vectors: %d \n', size(svm_model.SVs, 1));
end

function tr_para = ESVM_minen(tr_para)
% output concatenated arrays of negative representations

if ~isfield(tr_para.model, 'svxs') || isempty(tr_para.model.svxs)
  tr_para.model.svxs = [];
end

%%
fprintf('### ITERATION %02d \n',tr_para.iteration);
sampled = tr_para.voc_feat;
score = sampled*tr_para.model.wtrace{end}-tr_para.model.btrace{end};  % score = feat*m.model.wtrace{end};```````
idx = find(score > tr_para.hnm_threshold_score);
[ss, bb] = sort(score, 'descend');
kept = min(length(bb), tr_para.train_keep_nsv_multiplier*length(idx));
tr_para.total_mines = tr_para.total_mines+1;
if kept>0   
    featNeg = sampled(bb(1:kept),:);
    fprintf(1,'Newly added: %05d || Kept Negatives: %04d ,max = %.3f \n',...
        size(featNeg, 1), kept, ss(1));
end
tr_para.model.svxs = cat(1, tr_para.model.svxs, featNeg);

end