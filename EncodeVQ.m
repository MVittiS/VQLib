function indices = EncodeVQ(input, codebook, useGPU)
%ENCODEVQ Encodes input data given a VQ codebook
%
%   This function encodes data into the given codebook under a
%     least-squares criteria; that is, it finds the codebook entry that is
%     closest to a given data point. It also runs fine on Nvidia GPUs using
%     gpuArray() input data, which makes the algorithm significantly faster.
%     
%
%   Instead of calculating the distance directly against every sample using
%     the minus operator (because  MatLAB is stupid enough to try allocating
%     `prod([size(input) size(codebook)])` entries, making this solution
%     infeasible for any worthwhile dataset), we instead calculate the
%     distance of all input samples for each codebook entry individually,
%     and then at the end find the one whose distance is smallest. 
%   This also works much better for GPUs, and will be faster as long as the
%     number of input samples is much larger than codebook entries, and the
%     GPU has enough memory to hold two copies of the dataset.
%   If not, we keep trying by using a bissecting sort until we can find a
%     subset of data that fits the GPU and proceed from there on.
%   
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

%% Type checking
    assert(isnumeric(input), "Input isn't a numeric type");
    assert(isnumeric(codebook), "Codebook isn't a numeric type");
    
    if ~exist('useGPU', 'var')
        useGPU = false;
    end

    assert(islogical(useGPU), "useGPU must be logical");
    assert(isscalar(useGPU), "useGPU must be scalar");
    
%% Size checking
    assert(numel(size(input)) == 2, sprintf( ...
        'Input data must be a 2D variable; is %dD instead', ...
        numel(size(input))));
    
    assert(numel(size(codebook)) == 2, sprintf( ...
        'Codebook must be a 2D variable; is %dD instead', ...
        numel(size(codebook))));
    
    assert(size(codebook, 1) == size(input, 1), sprintf( ...
        "Input (%d) and codebook (%d) don't have same number of rows", ...
        size(input, 1), size(codebook, 1)));
    
%% Feature checking
    if useGPU
        try
            gpuDevice();
        catch
            useGPU = false;
            warning("Couldn't initialize gpu. Falling back to software mode.");
        end
    end
    
    if useGPU
        if ~existsOnGPU(input)
            input = gpuArray(input);
            warning("Input to EncodeVQ wasn't in GPU memory. Performing copy.");
        end
        if ~existsOnGPU(codebook)
            codebook = gpuArray(codebook);
            warning("Codebook to EncodeVQ wasn't in GPU memory. Performing copy.");
        end
    end

    
%% Function Body
    if useGPU
        distances = gpuArray(single(zeros(...
            size(codebook, 2), size(input, 2))));
    else
        distances = zeros(size(codebook, 2), size(input, 2));
    end

    subDivisions = 1;

    for x = 1 : size(codebook, 2)
        couldCalculate = false;
        while couldCalculate ~= true
            try
                delta = ceil(size(input, 2) / subDivisions);
                offset = 1;
                for divisions = 1 : subDivisions
                    range = offset : min(offset + delta, size(input, 2));
                    distanceToVec = input(:, range) - codebook(:, x);

                    % If you want to change the metric, like inf or 0 or
                    %  1-norm instead of 2-norm, do it here!
                    distances(x, range) = sum(distanceToVec.^2);

                    offset = offset + delta + 1;
                end
                couldCalculate = true;
            catch
                subDivisions = subDivisions + 1;
            end
        end
    end
    [~, indices] = min(distances);
    indices = int32(indices);
end

