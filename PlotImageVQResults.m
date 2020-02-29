function PlotImageVQResults(originalImg, vqImg, idxs, blockSize)
%PLOTIMAGEVQRESULTS Plots data about a VQ-encoding result for images
%   This function plots four images regarding the results of an image VQ
%    compression operation:
%   - Original Image
%   - Compressed/Decoded Image
%   - Difference Image (between original and compressed)
%   - Index Usage Image
    
    figure;
    
    subplot(2, 2, 1);
    imshow(originalImg);
    title('Original Image');
    
    subplot(2, 2, 2);
    imshow(vqImg);
    title('Compressed Image');
    
    subplot(2, 2, 3);
    imshow(0.5 + originalImg - vqImg);
    title('Difference/Error Image');
    
    subplot(2, 2, 4);
    imagesc(reshape(idxs, [], size(originalImg, 2) / blockSize));
    title('VQ Indices');
end
