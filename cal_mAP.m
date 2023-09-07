function mAP = cal_mAP(bndboxLoc, param)
bndboxIoU = get_IoU(double(bndboxLoc), double(param.bndboxLocGT));
Threshold = 0:0.05:1.0;
j = 1;
SucR(1) = 1;
for i = Threshold(2:end)
    j = j+1;
    SucR(j) = length(find(bndboxIoU >= i)) /length(bndboxIoU);
end
mAP = sum((Threshold(2:end)-Threshold(1:end-1)).*(SucR(2:end)+SucR(1:end-1)))/2;
end

function o = get_IoU(a, b)
% Compute the symmetric intersection over union overlap by each row between 
% a set of bounding boxes a and a set of bounding boxes b;
if size(a,1) ~= size(b,1)
    error('Error in size of bounding box sets!');
end

x1 = max(a(:,1), b(:,1));
y1 = max(a(:,2), b(:,2));
x2 = min(a(:,3), b(:,3));
y2 = min(a(:,4), b(:,4));

w = x2-x1+1;
h = y2-y1+1;
interS = w.*h;
aarea = (a(:,3)-a(:,1)+1).*(a(:,4)-a(:,2)+1);
barea = (b(:,3)-b(:,1)+1).*(b(:,4)-b(:,2)+1);
o = interS./(aarea+barea-interS);

o(w<=0) = 0;
o(h<=0) = 0;
end