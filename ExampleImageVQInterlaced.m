%   This script is essentially the same as ExampleImageVQ, but with added
%    interlacing of the input data in the spatial domain prior to
%    quantization, to demonstrate the effect data layout has on VQ
%    compression.
%    
%   We load the same example image (Mushroom.jpeg), but this time
%   rearranging the pixels into 4 sub-images, one with each pixel from a
%   2x2 block (upper-left to lower-right) before encoding this new image
%   with the same method as before. This can be considered a crude form of
%   a hybrid VQ-wavelet approach for compression, since gradients should
%   look better than the straightforward approach.
%    

%% Load and convert image type
img = single(imread('Mushroom.jpg')) / 255.0;
%imshow(img);

%% Convert the image into 4 sub-images (or, interlace the image)
imgUL = img(1:2:end, 1:2:end, :);
imgUR = img(1:2:end, 2:2:end, :);
imgDL = img(2:2:end, 1:2:end, :);
imgDR = img(2:2:end, 2:2:end, :);
imgBlk = [imgUL imgUR; imgDL imgDR];
imshow(imgBlk);

%% Convert image into blocks
blockLen = 2;
blocks = im2col(imgBlk(:, :, 1), [blockLen blockLen], 'distinct');
blocks = [blocks; im2col(img(:, :, 2), [blockLen blockLen], 'distinct')];
blocks = [blocks; im2col(img(:, :, 3), [blockLen blockLen], 'distinct')];

%% Create/Encode VQ Dictionary and Indices
[dict, idx] = GenVQDictMEX(blocks, 256);

%% Store Dictionary and Indices to disk
save('DictMushroomInterlaced.mat');

%% Later, load Dictionary and Indices from disk
load('DictMushroomInterlaced.mat');

%% Then, decode the compressed VQ blocks
newBlocks = DecodeVQ(dict, idx);

%% And separate into color channels
newImgR = col2im(newBlocks(1:4, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgG = col2im(newBlocks(5:8, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgB = col2im(newBlocks(9:12, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImg = cat(3, newImgR, newImgG, newImgB);

%% Second to last, revert the image interlacing
newImgUL = newImg(1:end/2, 1:end/2, :);
newImgUR = newImg(1:end/2, end/2 + 1 : end, :);
newImgDL = newImg(end/2 + 1 : end, 1:end/2, :);
newImgDR = newImg(end/2 + 1 : end, end/2 + 1 : end, :);

newImg(1:2:end, 1:2:end, :) = newImgUL;
newImg(1:2:end, 2:2:end, :) = newImgUR;
newImg(2:2:end, 1:2:end, :) = newImgDL;
newImg(2:2:end, 2:2:end, :) = newImgDR;

%% Finally, plot results
PlotImageVQResults(img, newImg, idx);