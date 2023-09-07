function I_aug = dataAug(addrI, aug_mode)
% Data Augmentation with different operations.

Ipath = dir(addrI); 
Ipath(1:2) = []; 
totalLen = length(Ipath);

I_aug = [];
for i = 1:totalLen
    I_ori = imread(fullfile(addrI, Ipath(i).name)); 
    if size(I_ori, 3) == 3
        I_ori = rgb2gray(I_ori);
    end
    if strcmp(aug_mode, 'aug')
        I_aug_ = TransRot(I_ori);
    else
        I_aug_ = {I_ori};
    end
    I_aug = [I_aug; I_aug_];
end
I_aug = reshape(I_aug, 1, []);
end

function I_aug = TransRot(I_ori)
j = 0;
t2l = [2 4]; t2r = [3 5]; rot = [-5 -2 3 6];
%% trans2left
for n = t2l
    j = j+1;
    I_ori(:,:) = [I_ori(:,n+1:end) I_ori(:,1:n)];
    I_aug{j} = I_ori;
end
%% trans2right
for n = t2r
    j = j+1;
    I_ori(:,:) = [I_ori(:,end-n+1:end) I_ori(:,1:end-n)];
    I_aug{j} = I_ori;
end
%% rotate 
for angle = rot
    j = j+1;
    I_aug{j} = imrotate(I_ori, angle, 'bilinear', 'crop');
end

end