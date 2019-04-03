function indices = VQEncode(input, codebook, useGPU)
%VQENCODE Encodes input data given a VQ codebook
%
%   This function encodes data into the given codebook under a
%   least-squares criteria; that is, it finds the codebook entry that is
%   closest to a given data point. It also runs fine on Nvidia GPUs using
%   gpuArray() input data, and that makes the algorithm significantly
%   faster.
%
%   Instead of calculating the distance directly against every sample using
%   the minus operator (because  MatLAB is stupid enough to try allocating
%   `prod([size(input) size(codebook)])` entries, making this solution
%   infeasible for any worthwhile dataset), we instead calculate the
%   distance of all input samples for each codebook entry individually, and
%   then at the end find the one whose distance is smallest. This also
%   works much better for GPUs, and will be faster as long as the number
%   input samples is much larger than codebook entries.
%
%   Input Arguments:
%
%   'input' - Your data samples to be encoded.
%     Type: 2D/matrix, floating point (from single to gpuArray:double)
%     Organization: 1D samples in every column, dimensions in every row
%
%   'codebook' - Codebook to calculate the indices from.
%     Type: 2D floating point matrix (from single to gpuArray:double)
%     Organization: 1D entries in every column, dimensions in every row
%
%   Output Arguments:
%
%   'indices' - Array of indices mapping samples to codebook entries.
%     Type: 1D/scalar integer array
%     Organization: one index per input sample/column

%% Size and type checking
    assert(isnumeric(input), "Input isn't a numeric type");
    assert(isnumeric(codebook), "Codebook isn't a numeric type");
    
    if ~exist('useGPU', 'var')
        useGPU = false;
    end
    assert(islogical(useGPU), "Veriable 'useGPU' must be logical");
    if useGPU
        try
            gpuDevice();
        catch
            useGPU = false;
            assert(0, "Your computer or GPU doesn't support CUDA");
        end
    end
    
    assert(len(size(input)) == 2, sprintf( ...
        'Input data must be a 2D variable; is %dD instead', ...
        len(size(input))));
    
    assert(len(size(codebook)) == 2, sprintf( ...
        'Codebook must be a 2D variable; is %dD instead', ...
        len(size(codebook))));
    
    assert(size(codebook, 1) == size(input, 1), sprintf( ...
        "Input (%d) and codebook (%d) don't have same number of rows", ...
        size(input, 1), size(codebook, 1)));
    
    
%% Distance calculating
    if useGpu
        distances = gpuArray(single(zeros(...
            size(codebook, 2), size(input, 2))));
    else
        distances = zeros(size(codebook, 2), size(input, 2));
    end
    for x = 1 : size(codebook, 2)
        distanceToVec = input - codebook(:, x);
        distances(x, :) = sum(distanceToVec.^2); % If you want to change
                                                 %  metric, like inf or
    end                                          %  0 or 1-norm, do it
    [~, indices] = min(distances);               %  here!
end

