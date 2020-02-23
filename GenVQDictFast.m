function [codebook, vqIdx] = GenVQDictFast(dataset, numVecs, useGPU)
%GENVQDICTFAST Generates a VQ Codebook using the MVS_f algorithm
%   This function generates a _reasonable_ VQ codebook given the desired
%     number of entries. As opposed to GenVQDict, it does *not* accepts
%     partial codebooks.
%
%   It is also optimized for GPUs using gpuArray(), providing a significant
%     speed boost over regular CPUs.
%
%
%   Input Arguments:
%
%   'dataset' - Your original data for generating the codebook from.
%     Type: 2D/matrix, floating point (from single to gpuArray:double)
%     Organization: 1D samples in every column, dimensions in every row
%
%   'numVecs' - Amount of entries in your codebook.
%     Type: 0D/scalar, integer or integer-convertible
%
%   'useGPU' - Optional, flag to indicate GPU (CUDA) acceleration.
%     Type: 0D/scalar, logical

%% Optional arguments checking
    if ~exist('useGPU', 'var')
        useGPU = false;
    end

    randomVecsFromCodebook = randi([1 size(dataset, 2)], 1, numVecs - 1);
    codebook = dataset(:, randomVecsFromCodebook);
    [codebook, vqIdx] = GenVQDict(dataset, numVecs, codebook, useGPU);
end
