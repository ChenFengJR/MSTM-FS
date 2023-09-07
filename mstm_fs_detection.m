function bndboxLoc = mstm_fs_detection(sear, t_h, t_w, param)
[s_h, s_w, ~] = size(sear);
numi = 20;   % horizontal scanning windows' num
numj = 20;   % vertical scanning windows' num
stri = floor((s_h-t_h)/(numi-1));  % row    -- height
strj = floor((s_w-t_w)/(numj-1));  % column -- width
sear_feat = zeros(numi*numj, size(param.tmpl, 1));
for i = 1:numi           % scan horizontally line by line
    for j = 1:numj       
        II = sear(1+stri*(i-1):t_h+stri*(i-1),1+strj*(j-1):t_w+strj*(j-1),:);
        ntag = numj*(i-1)+j;
        sear_feat(ntag,:) = featExt(II, param);  
    end
end
sear_feat = L2_norm(sear_feat);
% sear_feat = (sear_feat - param.SA_meanT)./param.SA_stdT * param.Xt;
% sear_feat = L2_norm(sear_feat);
scored = sear_feat * param.tmpl;
[aa, bb] = sort(scored);   % esvm_nms scores ascendantly
boxes(:,1) = 1+strj*mod(bb-1, numj);
boxes(:,2) = 1+stri*(ceil(bb/numj)-1);
boxes(:,3) = strj*mod(bb-1, numj)+t_w;
boxes(:,4) = stri*(ceil(bb/numj)-1)+t_h;
boxes(:,5) = aa;
topboxes = esvm_nms(boxes, param.nms_overlap);
bndboxLoc = topboxes(1, 1:4);
end