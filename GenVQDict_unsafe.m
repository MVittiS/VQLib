function [codebook, vqIdx] = GenVQDict_unsafe(dataset, numVecs)
%GENVQDICTUNSAFE Generates a VQ Codebook using the MVS algorithm
%   This function generates a near-optimal VQ codebook given the desired
%     number of entries. It also accepts partially calculated codebooks,
%     allowing for continuation if the codebook generation is too slow or
%     requires a large dataset.
%
%   This is the unsafe variant, ready for MatLAB Coder.
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

%% Variable Setup
    segmentLength = size(dataset, 1);
    
    codebook = single(zeros(segmentLength, numVecs));
    codebook(:, 1) = mean(dataset, 2);
    vqIdx = ones(1, size(dataset, 2));

%% Main Loop

    for v = 1 : (numVecs - 1)
        if v > 1
            vqIdx = EncodeVQ_unsafe(dataset, codebook(:, 1:v));
        else
            vqIdx = ones(1, size(dataset, 2));
        end

        % We perform two-step quantization: first, split the code
        %  vector with the largest number of samples using it.
        mostFrequentVec = mode(vqIdx, 2);
        coder.varsize('deviation');
        deviation = ...
            cov((dataset(:, vqIdx == mostFrequentVec) ...
            - codebook(:, mostFrequentVec))');
        deviation = ...
            deviation(:, 1) ...
            / sum(std(dataset(:, vqIdx == mostFrequentVec), 0, 1), 2);

        % With the deviation in hands, we displace the central cluster.
        oldVec = codebook(:, mostFrequentVec);
        codebook(:, (mostFrequentVec + 2) ...
            : numVecs) = codebook(:, (mostFrequentVec + 1) : (numVecs - 1));
        codebook(:, mostFrequentVec) = oldVec + deviation;
        codebook(:, mostFrequentVec + 1) = oldVec - deviation;

        vqIdx = EncodeVQ_unsafe(dataset, codebook(:, 1 : (v + 1)));
        vecChanged = true;

        % Then, uniformize the codebook by iterating on the newly-found
        %  vectors until the codebook stabilizes (which means, vectors
        %  won't switch indexes anymore and are tightly clustered). Or,
        %  iteration limit is reached, just because we don't have infinite
        %  computer time to waste.
        iterLimit = 300;
        iterations = 1;
        while vecChanged && iterations < ceil((iterLimit / sqrt(v)))
            vecChanged = false;
            for x = 1:(v + 1)
                codebook(:, x) = mean(dataset(:, vqIdx == x), 2);
            end

            vq2Idx = EncodeVQ_unsafe(dataset, codebook(:, 1 : (v + 1)));
            if any(vq2Idx ~= vqIdx)
                vecChanged = true;
                vqIdx = vq2Idx;
            end
            iterations = iterations + 1;
        end
    end
end

