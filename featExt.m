function feat = featExt(Img, param)
% extract features

if size(Img, 3) == 3
    Img = rgb2gray(Img);
end
if size(Img, 1) ~= 50 || size(Img, 2) ~= 50
    Img = imresize(Img, [50, 50], 'nearest');
end

switch param.feat_mode
    case 'RawPixel', feat = double(reshape(Img,1,[]));   
    case 'HOG', feat = extractHOGFeatures(Img);   
    case 'CNN', feat = featExt_CNN(Img, param.net);  
    otherwise, error('wrong mode!');
end

end

function feat = featExt_CNN(img, net)
% This funciton is used to extract features of imgs in CNN 
% input: img, net, params(model_parameters) including sp_n, layer_num, 
%        model_type(1 for 'with FC' and 2 for 'without FC ')
%        "img" can be path, bmp, jpg
layer_num = 12; 
sp_n = 2;
switch class(img)
    case {'double', 'uint8'}, img = single(img);  % ~isa(img, 'single')
    case 'char', img = single(imread(img)); % path
    otherwise, error('Input IMG path or IMG file!');
end
img = gray2rgb(img);
size_im = size(img);
norm(:,:,1) = net.meta.normalization.averageImage(1)*ones(size_im(1:2));
norm(:,:,2) = net.meta.normalization.averageImage(2)*ones(size_im(1:2));
norm(:,:,3) = net.meta.normalization.averageImage(3)*ones(size_im(1:2));
img = img - norm;
res = vl_simplenn_noFC(net, img);
res = table({res(layer_num).x}.','VariableNames',{'x'});
conv_img = res.x{1,1};
[r_tf, c_tf, ~] = size(conv_img);
win_h = ceil(r_tf/sp_n); win_w = ceil(c_tf/sp_n);  
str_h = floor(r_tf/sp_n); str_w = floor(c_tf/sp_n);
if (str_h==0)||(str_w==0)    
    feat = conv_img;
else
    feat = vl_nnpool(conv_img,[win_h,win_w],'Stride',[str_h,str_w],'Method','max');
end
feat = reshape(feat,1,[]);
end
