function [rows, cols] = trackFeatures(imgsDoubleGray, dispRow, dispCol, SSDkernelSize, thresholdDetecting, thresholdMatching)
% trackFeatures Tracks the features on the first image across the rest of
% the images
%
% [rows, cols] = trackFeatures(imgsDoubleGray, dispRow, dispCol,
% SSDkernelSize, thresholdDetecting, thresholdMatching) where imgDoubleGray
% is a set of grayscale images, dispRow is the half of the vertical size of
% the search window, dispCol is the half of the horizontal size of the
% search window, SSDkernelSize is the size of the SSD kernel,
% thresholdDetecting is a threshold for feature detection, and
% thresholdMatching is a threshold for feature matching. Finally, rows and
% cols indicate rows and columns of the matched features.
%
% CSC 262 Final Paper

% Find the size and number of the input images
[numRows, numCols, numImgs] = size(imgsDoubleGray);

% Use the function detectFeatures to detect features on the first image
imgFeature = detectFeatures(imgsDoubleGray(:,:,1), thresholdDetecting);

% Find the rows and columns of the detected features on the first image
[rowsFeature, colsFeature] = find(imgFeature);

% Pad the first image with detected features based on the size of the
% search window
imgFeature = padarray(imgFeature, [dispRow, dispCol], 'both');

% Find the linear index of the padded first image
indexFeature = find(imgFeature);

% Find the number of the detected features on the first image
numFeatures = size(indexFeature, 1);

% Initalize the first rows of rows and cols to the rows and columns of the detected features
% on the first image
rows(1, :) = rowsFeature;
cols(1, :) = colsFeature;

% Initilize the rest of rows of rows and cols to NaN
rows(2:numImgs, :) = NaN(numImgs-1, numFeatures);
cols(2:numImgs, :) = NaN(numImgs-1, numFeatures);

% Initalize the SSD kernel
SSDkernel = ones([1 SSDkernelSize]);

% Pad the first image based on the search window size
firstPadImg = padarray(imgsDoubleGray(:,:,1), [dispRow, dispCol], 'both');

% Traverse the rest of the images to track the features detected on the
% first image
for i=2:numImgs

    % Initalize a vector to store the SSD of each feature
    sumSqrDiffFeaturePrev = gpuArray(inf(numFeatures, 1)); % GPU is used for optimization

    % Move the search window
    for k = -dispRow:dispRow
        for l = -dispCol:dispCol
            % Pad the current image
            compPadImg = gpuArray(zeros(size(firstPadImg))); % GPU is used for optimization
            compPadImg(k+dispRow+1:k+dispRow+numRows,l+dispCol+1:l+dispCol+numCols) = imgsDoubleGray(:,:,i); % This way is used because the function imtranslate does not work for GPU
            
            % Calculate the squard difference between the first image and
            % the current image
            sqrDiff = (firstPadImg - compPadImg) .^2;
            % Use convolution to find the SSD on the SSD window
            sumSqrDiff = conv2(SSDkernel, SSDkernel, sqrDiff, 'same');
            
            % Find the SSD of the detected features
            sumSqrDiffFeature = sumSqrDiff(indexFeature);

            % Find the detected features whose SSD is lower than the
            % previous SSD and the threshold and update them
            indexUpdate = find(sumSqrDiffFeature < sumSqrDiffFeaturePrev & sumSqrDiffFeature < thresholdMatching);
            rows(i,indexUpdate) = k+rowsFeature(indexUpdate);
            cols(i,indexUpdate) = l+colsFeature(indexUpdate);

            % Update the previous SSD of the detected features
            sumSqrDiffFeaturePrev(indexUpdate) = sumSqrDiffFeature(indexUpdate);
        end
    end
end

end

%% Acknowledgements
% The idea for tracking features is from the Stereo Disparity lab.