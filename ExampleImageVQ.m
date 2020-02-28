%   This script uses the three provided functions to exemplify a typical
%    VQ workflow: reshaping (and optionally weighting) the data,
%    generating a dictionary, storing said dictionary and indices in some
%    space-saving format, and then loading and decoding them again to
%    compare against the original data.
%
%   Here we load the provided example image (Mushroom.jpg), partition
%    it in 2x2 blocks, flatten said blocks to 4x1 vectors, make our
%    dictionary, store the dictionary and indices as 8bpp values, and
%    finally undo the process to compare the image against the original,
%    in terms of error.
%
%   In the end, this produces an image with an approximately 12:1 fixed
%    compression ratio, and very similar charcteristics to the texture
%    format used by Videologic's PowerVR2 GPU, used in the Sega Dreamcast
%    videogame console. Block artifacts are clearly visible, but having
%    such a nice fixed compression ratio and decoding speed is a good
%    demonstration of some of the capabilities of VQ.

%% Load and convert image type
img = single(imread('Mushroom.jpg')) / 255.0;
%imshow(img);

%% Convert image into blocks
blockLen = 2;
blocks = im2col(img(:, :, 1), [blockLen blockLen], 'distinct');
blocks = [blocks; im2col(img(:, :, 2), [blockLen blockLen], 'distinct')];
blocks = [blocks; im2col(img(:, :, 3), [blockLen blockLen], 'distinct')];

%% Create/Encode VQ Dictionary and Indices
[dict, idx] = GenVQDict(blocks, 256);

%% Store Dictionary and Indices to disk
save('DictMushroom.mat');

%% Later, load Dictionary and Indices from disk
load('DictMushroom.mat');

%% Then, decode the compressed VQ blocks
newBlocks = DecodeVQ(dict, idx);

%% And separate into color channels
newImgR = col2im(newBlocks(1:(blockLen^2), :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgG = col2im(newBlocks((blockLen^2 + 1):(2*(blockLen^2)), :), ...
    [blockLen, blockLen], [360 540], 'distinct');

newImgB = col2im(newBlocks((2*(blockLen^2) + 1):(3*(blockLen^2)), :), ...
    [blockLen, blockLen], [360 540], 'distinct');

%% Finally, plot results
newImg = cat(3, newImgR, newImgG, newImgB);
PlotImageVQResults(img, newImg, idx);

