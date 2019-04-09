function output = DecodeVQ(codebook, indices)
%DECODEVQ Reconstructs a VQ-compressed signal, given codebook and indices
%   Since doing an index expansion (a.k.a. VQ decompression) is trivial
%     in MatLAB, this function is just provided for completeness' sake.
%
%
%   Input Arguments:
%
%   'codebook' - Codebook to decode the indices with.
%     Type: 2D floating point matrix (from single to gpuArray:double)
%     Organization: flat entries in every column, dimensions in every row
%
%   'indices' - Indices to reconstruct the original signal from codebook.
%     Type: 1D row or column, integer or integer-convertible.

%% Type checking
    assert(isnumeric(codebook), sprintf( ...
        'Codebook is not a numeric type; is %s instead', ...
        class(codebook)));

    assert(isnumeric(indices), sprintf( ...
        'Codebook is not a numeric type; is %s instead', ...
        class(codebook)));


%% Function Body
    output = codebook(:, indices);
    output = output(:);
end