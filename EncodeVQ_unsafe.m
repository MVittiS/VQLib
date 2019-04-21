function indices = EncodeVQ_unsafe(input, codebook)
%ENCODEVQUNSAFE Encodes input data given a VQ codebook
%
%   This function encodes data into the given codebook under a
%     least-squares criteria; that is, it finds the codebook entry that is
%     closest to a given data point. 
%
%   This is the unsafe variant that does not support bissection.
%
%   Instead of calculating the distance directly against every sample using
%     the minus operator (because  MatLAB is stupid enough to try to
%     allocate `prod([size(input) size(codebook)])` entries, making this 
%     solution infeasible for any worthwhile dataset), we instead calculate
%     the distance of all input samples for each unique codebook entry, and
%     then at the end find the one whose distance is smallest.   
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

    
%% Function Body
    
    distances = single(zeros(size(codebook, 2), size(input, 2)));
    
    % You can optionally use 'parfor' here, if your dataset is large enough
    %  to compensate for MatLAB's overhead.
    for x = 1 : size(codebook, 2)
        distanceToVec = input - codebook(:, x);
        % If you want to change the metric, like inf or 0 or
        %  1-norm instead of 2-norm, do it here!
        distances(x, :) = sum(distanceToVec.^2, 1);
    end
    [~, indices] = min(distances);
end

