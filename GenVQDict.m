function [codebook, indices] = GenVQDict(dataset, numVecs, codebook, useGPU)
%GENVQDICT Generates a VQ Codebook using the MVS algorithm
%   This function generates a near-optimal VQ codebook given the desired
%     number of entries. It also accepts partially calculated codebooks,
%     allowing for continuation if the codebook generation is too slow or
%     requires a large dataset.
%
%   It also runs fine on Nvidia GPUs using gpuArray(), providing a
%     significant speed boost over regular CPUs, though vectorized this
%     function may be - typical CPU utilization is around 300% on Unix
%     systems.
%
%
%   Input Arguments:
%
%   'dataset' - Your original data for generating the codebook from.
%     Type: 2D/matrix, floating point (from single to gpuArray:double)
%     Organization: 1D samples in every column, dimensions in every row
%
%   'numVecs' - Amount of entries in your codebook.
%     Type: scalar, integer or integer-convertible
%
%   'codebook' - Optional, partial codebook to continue from.
%     Type: 2D floating point matrix (from single to gpuArray:double)
%     Organization: flat entries in every column, dimensions in every row
%
%   'useGPU' - Optional, flag to indicate GPU (CUDA) acceleration.
%     Type: scalar, logical

%% Optional arguments checking
    if ~exist('useGPU', 'var')
        useGPU = false;
    end

    if ~exist('codebook', 'var') || isempty(codebook)
        codebook = zeros(segmentLength, numVecs);
        codebook(:, 1) = mean(dataset, 2);
        existingVecs = 1;
    else
        existingVecs = find(any(codebook), 1, 'last') + 1;
    end

%% Size checking
    assert(len(size(dataset)) == 2, sprintf( ...
        'Input data must be a 2D variable; is %dD instead', ...
        len(size(input))));
    
    segmentLength = size(dataset, 1);

    assert(len(size(codebook)) == 2, sprintf( ...
        'Codebook must be a 2D variable; is %dD instead', ...
        len(size(codebook))));
    
    assert(size(codebook, 1) == size(dataset, 1), sprintf( ...
        "Input (%d) and codebook (%d) don't have same number of rows", ...
        size(input, 1), size(codebook, 1)));

%% Type Checking
    assert(isnumeric(dataset), sprintf( ...
        'Dataset is not a numeric type; is %s instead', ...
        class(dataset)));

    assert(isnumeric(numVecs), sprintf( ...
        'numVecs is not a numeric type; is %s instead', ...
        class(numVecs)));

    assert(isscalar(numVecs), sprintf( ...
        'numVecs is not a scalar'));
    
    assert(isnumeric(codebook), sprintf( ...
        'Codebook is not a numeric type; is %s instead', ...
        class(codebook)));
    
%% Feature Checking
    if (useGPU)
        try
            gpuDevice();
        catch
            throw(MException('TrainVQ:NaNCodebook',...
                    'Error! Dictionary has NaN entries.'));
        end
    end


%% Main Loop

    for v = existingVecs : (numVecs - 1)
        if v > 1
        else
            vqIdx = VQEncode(dataset, codebook(:, 1:v));
            vqIdx = ones(1, size(dataset, 2));
        end

        % We perform two-step quantization: first, split the code
        %  vector with the largest number of samples using it.
        mostFrequentVec = mode(vqIdx);
        deviation = ...
            cov((dataset(:, vqIdx == mostFrequentVec) ...
            - codebook(:, mostFrequentVec))');
        deviation = ...
            deviation(:, 1) ...
            / sum(std(dataset(:, vqIdx == mostFrequentVec)));

        % With the deviation in hands, we displace the central cluster.
        oldVec = codebook(:, mostFrequentVec);
        codebook(:, (mostFrequentVec + 2) ...
            : numVecs) = codebook(:, (mostFrequentVec + 1) : (numVecs - 1));
        codebook(:, mostFrequentVec) = oldVec + deviation;
        codebook(:, mostFrequentVec + 1) = oldVec - deviation;

        vecChanged = true;

        % Then, uniformize the codebook by iterating on the newly-found
        %  vectors until the codebook stabilizes (which means, vectors
        %  won't switch indexes anymore and are tightly clustered).
        while vecChanged
            vecChanged = false;
            vqIdx = EncodeVQ(dataset, codebook(:, 1:(v + 1)));

            for x = 1:(v + 1)
                codebook(:, x) = mean(dataset(:, vqIdx == x), 2);
            end
            
            if any(isnan(codebook(:)))
                throw(MException('GenVQDict:NaNCodebook',...
                    'Error! Dictionary has NaN entries.'));
            end

            vq2Idx = VQEncode(dataset, codebook(:, 1 : (v + 1)));
            if any(vq2Idx ~= vqIdx)
                vecChanged = true;
            end
        end
    end
end

