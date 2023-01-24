function imgResult = detectFeatures(imgDoubleGray, threshold)
% detectFeatures Detects features of an input image
%
% imgResult = detectFeatures(img, threshold) where img is a double and gray
% image and imgResult is a binary image with detected features.
%
% CSC 262 Final Paper

% Initialze necessary kernels
gauss = gkern(1);
gaussD = gkern(1,1);
gaussLarge = gkern(1.5^2);

% Find the horizontal and vertical partial derivative
imgPartialY = conv2(gaussD, gauss, imgDoubleGray,'same');
imgPartialX = conv2(gauss, gaussD, imgDoubleGray,'same');

% Find the elements of the matrix A with the two partial derivatives
imgPartialX2 = imgPartialX.^2;
imgPartialY2 = imgPartialY.^2;
imgPartialXY = imgPartialX.*imgPartialY;

% Blur the elements for a little robustness as described in the Feature
% Detection lab
imgPartialX2Blur = conv2(gaussLarge,gaussLarge, imgPartialX2, 'same');
imgPartialY2Blur = conv2(gaussLarge,gaussLarge, imgPartialY2, 'same');
imgPartialXYBlur = conv2(gaussLarge,gaussLarge, imgPartialXY, 'same');

% Find the determinant and the trace
imgDet = imgPartialX2Blur.*imgPartialY2Blur - imgPartialXYBlur.^2;
imgTrace = imgPartialX2Blur + imgPartialY2Blur;

% Find the ratio of the determinant to the trace
imgRatio = imgDet./imgTrace;

% Filter out the detected features whose ratio is lower than or equal to
% the threshold
imgRatioThres = imgRatio>threshold;

% Find the pixels which are local maxima
imgMaxima = maxima(imgRatio);

% Return the detected features whose ratio are greater than the threshold
% and which are local maxima
imgResult = imgRatioThres & imgMaxima;
end

%% Acknowledgements
% The original code is from the Feature Detection lab and it is slightly
% modified to pass the threshold to the function.