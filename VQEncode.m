function indices = VQEncode(input, codebook)
%VQENCODE Encodes input data given a VQ codebook
%
%   This function encodes data into the given codebook under a
%   least-squares criteria; that is, it finds the codebook entry that is
%   closest to a given data point. It also runs fine on Nvidia GPUs using
%   gpuArray() input data, and that makes the algorithm significantly
%   faster.
%
%   Instead of calculating the distance directly for every sample using the
%   minus operator (because  MatLAB is stupid enough to try allocating
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

    

end

