%   This script is very similar to ExampleImageVQ, but with all of the
%    processing happening on the GPU. In order to run this example, you'll
%    need (as of Feb 2020) a MatLAB installation with the Parallel Tolboox,
%    plus an Nvidia GPU with at least 1GB of VRAM. I hope Octave can have
%    some of this functionality later on too, maybe with either OpenCL or
%    Vulkan.

%% Load and convert image type
img = gpuArray(single(imread('Mushroom.jpg')) / 255.0);
%imshow(img);

%% Convert image into blocks
blockLen = 2;
blocks = im2col(img(:, :, 1), [blockLen blockLen], 'distinct');
blocks = [blocks; im2col(img(:, :, 2), [blockLen blockLen], 'distinct')];
blocks = [blocks; im2col(img(:, :, 3), [blockLen blockLen], 'distinct')];
blocks = single(blocks); % MatLAB *really* likes doubles, 
                         % but they tank GPU performance

%% Create/Encode VQ Dictionary and Indices
[dict, idx] = GenVQDict(blocks, 256, [], true);
%[dict, idx] = GenVQDictFast(blocks, 256, true);

%% Store Dictionary and Indices to disk
save('DictMushroom.mat');

%% Later, load Dictionary and Indices from disk
load('DictMushroom.mat');

%% Then, decode the compressed VQ blocks
newBlocks = DecodeVQ(dict, idx);
newBlocks = gather(newBlocks); % <- because col2im() won't accept
                               %    gpuArray types :P

%% And separate into color channels
newImgR = col2im(newBlocks(1:4, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgG = col2im(newBlocks(5:8, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgB = col2im(newBlocks(9:12, :), ...
    [blockLen, blockLen], [360 540], 'distinct');

%% Finally, plot results
newImg = cat(3, newImgR, newImgG, newImgB);
PlotImageVQResults(img, newImg, idx);

