function img = gray2rgb(img)
cha = size(img,3);
if cha == 1
    imgt = img;
    img(:,:,1) = imgt;
    img(:,:,2) = imgt;
    img(:,:,3) = imgt;
end
end