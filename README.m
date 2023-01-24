%% CSC 262: Final Project

%% Project Description
% The goal of this project is to reconstruct a 3D reconstruction from a
% video input. To achieve the goal, we have the following four milestones:
% 1. Feature Detection & Matching
% 2. Matrix Factorization with missing data using the damped Newton method
% 3. Bundle Adjustment
% 4. Integrating all the method with a video input
%
% The milestones are divided by a section below and described with examples
% in each section. You can see what we have done for each milestone and
% replicate our results.
%
% For the first two milestones, we use the Urban3 dataset from the
% Middlebury optical flow dataset. For the last two milestones, we use our
% own pictures and videos.

%% Preprocessing

% Add all the subfolders to the path
addpath('Dataset');
addpath('Dataset/Urban3');
addpath('Dataset/Bottle');
addpath('Dataset/Video');
addpath('First Milestone');
addpath('Second Milestone');
addpath('Third Milestone');
addpath('Fourth Milestone');

%% First Milestone

% Load synthetic images from the Optical Flow Middleburry Dataset

% Number of images to load
numImgs = 8;

% Values for constructing the file names of the images
numStart = 7;
strStart = 'Dataset/Urban3/frame';
strEnd = '.png';
numDigits = 2;

% Load each image in looop
for i = numStart:numStart+numImgs-1
    
    % Find the file names of the current image
    numZeros = numDigits-numel(num2str(i));
    leadingZeros = strcat(num2str(zeros([1 numZeros]),'%d'));
    
    % Read the current image
    imgs(:,:,:,i-numStart+1) = imread(strcat(strStart, leadingZeros, num2str(i), strEnd));
    
    % Convert the current RGB image to the gray and double image
    imgsDoubleGray(:,:,i-numStart+1) = im2double(rgb2gray(imgs(:,:,:,i-numStart+1)));
end

% Detect and match features in consecutive images

% Variables for detecting and matching featueres in consecutive images
dispRow = 30;
dispCol = 30;
SSDkernelSize = 40;
thresholdDetecting = 0.000001;
thresholdMatching = 30;

% Find the rows and columns for the matched features
[rows, cols] = trackFeatures(imgsDoubleGray, dispRow, dispCol, SSDkernelSize, thresholdDetecting, thresholdMatching);

% Initialize a concatenated image to the first image
concatImg = imgsDoubleGray(:,:,1);

% Initialize x and y coordinates for the matched features to the rows and
% columns from the function trackFeatures
rowsPlot = rows;
colsPlot = cols;

% Concatenate the rest of the images and adjust their y-coordinates
% accordingly
for i=2:size(imgsDoubleGray, 3)
    concatImg = [concatImg imgsDoubleGray(:,:,i)];
    colsPlot(i:end,:) = colsPlot(i:end,:) + size(imgsDoubleGray, 2);
end

% Show the concatenated image with the matched features connected by lines
figure;
imshow(concatImg);
hold on;
line(colsPlot(:,1:50:end), rowsPlot(:,1:50:end));
title('Detected and Matched Features in Consecutive Images');
hold off;

%% Second Milestone

% Create an observation matrix with NaN values
Onan = [cols; rows];

% Create an observation matrix without NaN values
O = Onan(:, find(~sum(isnan(Onan))));

% Subtract a centroid from each observation matrix
Onan = Onan-mean(O,2);
O = O-mean(O,2);

% Use SVD to find a shape matrix from the observation matrix without NaN
% values
[A,S,B] = svd(O, 'econ');
shape = (S(1:3,1:3)*B(:,1:3)')';

% Find a weight matrix for the observation matrix with NaN values
W = ones(size(Onan,1), size(Onan,2));
W(find(isnan(Onan))) = 0;

% Initialize matrices to be factorized with random numbers
m = size(Onan,1);
n = size(Onan,2);
r = min(m,n);

A1 = rand(m, r);
B1 = rand(n, r);

Onan1 = Onan;
Onan1(find(W)) = 0;

% Initialize other matrices to be factorized based on the matched features
while sum(isnan(Onan), 'all') ~= 0
    [rowsnan, colsnan] = find(isnan(Onan));
    Onan(sub2ind([m n],rowsnan,colsnan)) = Onan(sub2ind([m n],rowsnan-1,colsnan));
end

[A2,S,B2] = svd(Onan, 'econ');
B2 = (S*B2')';

% Find the factorized matrices with the damped Newton method
iteration = 1000;
lambda = 1;
lambdaStep = 10;
thresholdConvergence = 1e-10;

Onew1 = matrixFactorization(Onan, W, A1, B1, iteration, lambda, lambdaStep, thresholdConvergence);
Onew2 = matrixFactorization(Onan, W, A2, B2, iteration, lambda, lambdaStep, thresholdConvergence);

[A1,S, B1] = svd(Onew1, 'econ');
shape1 = (S(1:3,1:3)*B1(1:3,:))';
[A2,S, B2] = svd(Onew2, 'econ');
shape2 = (S(1:3,1:3)*B2(1:3,1:3))';

% Plot a point cloud for each example
pc = pointCloud(shape);
pc1 = pointCloud(shape1);
pc2 = pointCloud(shape2);

figure;
ax = pcshow(pc);
ax.DataAspectRatio = [1, 1, diff(ax.ZLim)/ diff(ax.YLim)];

figure;
ax = pcshow(pc1);
ax.DataAspectRatio = [1, 1, diff(ax.ZLim)/ diff(ax.YLim)];

figure;
ax = pcshow(pc2);
ax.DataAspectRatio = [1, 1, diff(ax.ZLim)/ diff(ax.YLim)];

%% Third Milestone

imgs(:,:,1) = imread("bottle1.png");
imgs(:,:,2) = imread("bottle1.png");

for i = 1:2
    imgsDoubleGray(:,:,i) = imresize(im2double(rgb2gray(imgs(:,:,:,i))), 0.3);
end

% Load the camera parameters
% The camera parameters are obtained by the MatLab camera calibration app.
load cameraParams.mat

intrinsics = cameraParams.Intrinsics;

% Sep up the variables for feature tracking
dispRow = 30;
dispCol = 30;
SSDkernelSize = 40;
thresholdDetecting = 0.000001;
thresholdMatching = 30;

% Sep up the variables for matrix factorization
iteration = 1000;
lambda = 1;
lambdaStep = 8;
thresholdConvergence = 1e-10;

% Sep up the variables for bundle adjustment
iterationHelper = 5000;
Confidence = 20;
MaxDistance = 50;
MaxNumTrials = 30;

numNextImgs = 1;

% Initialize the first image view
vSet = imageviewset;

[rows, cols] = find(detectFeatures(imgsDoubleGray(:,:,1), thresholdDetecting));
currPoints = [cols, rows];
vSet = addView(vSet, 1, rigidtform3d, Points=currPoints);

numPrevMatchedNotDetected = 0;

% Track features in consecutive frames
[rows, cols] = trackFeatures(imgsDoubleGray(:,:,1:1+numNextImgs), dispRow, dispCol, SSDkernelSize, thresholdDetecting, thresholdMatching);

% Create an observation matrix
Onan = [cols; rows];

% Create a weight matrix
W = ones(size(Onan,1), size(Onan,2));
W(isnan(Onan)) = 0;

% Initalize A and B
while sum(isnan(Onan), 'all') ~= 0
    [rowsnan, colsnan] = find(isnan(Onan));
    Onan(sub2ind([size(Onan,1) size(Onan,2)],rowsnan,colsnan)) = Onan(sub2ind([size(Onan,1) size(Onan,2)],rowsnan-1,colsnan));
end

Onan = Onan-mean(Onan,2);

[A,S,B] = svd(Onan, 'econ');
B = S*B';

% Find the estimated A and B
Onew = matrixFactorization(OnanCentered, W, iteration, lambda, lambdaStep, thresholdConvergence, A, B);

% Update the missing values
Onew = round(Onew+centroid);
Onan(find(W)) = Onew(find(W));

% Store the matched features bewteen the current image and the next
% image
currPoints = [Onan(1,:); Onan(2+numNextImgs,:)]';
[rows, cols] = find(detectFeatures(imgsDoubleGray(:,:,2), thresholdDetecting));
nextPointsMatched = [Onan(2,:); Onan(3+numNextImgs,:)]';
nextPointsDetected = [cols, rows];
[nextPointsMatchedNotDetected, indexM] = setdiff(nextPointsMatched, nextPointsDetected, 'rows', 'stable');

nextPoints = [nextPointsMatchedNotDetected; nextPointsDetected];

% Calculate the relative position of the camera for the next image
[relPose, inlierIdx] = helperEstimateRelativePose(...
    currPoints, nextPointsMatched, intrinsics, iterationHelper, Confidence, MaxDistance, MaxNumTrials);

% Find the poisition of the camera for the current image
currPose = poses(vSet, 1).AbsolutePose;

% Calculate the position of the camera for the next image
nextPose = rigidtform3d(currPose.A*relPose.A);

% Add the next image view
vSet = addView(vSet, 2, nextPose, Points=nextPoints);

% Make connections bewteen the current and next images using the
% matched features
currIndexMatches = find(inlierIdx);

firstNextIndexMatches = inlierIdx;
firstNextIndexMatches(setdiff(1:end, indexM)) = 0;
firstNextIndexMatches = find(firstNextIndexMatches);

secondNextIndexMatches = inlierIdx;
secondNextIndexMatches(indexM) = 0;
secondNextIndexMatches = find(secondNextIndexMatches);

vSet = addConnection(vSet, 1, 2, relPose, Matches=[currIndexMatches, [firstNextIndexMatches; secondNextIndexMatches]]);

% Find a track in the view set
tracks = findTracks(vSet);

% Find the camera positions in the view set
camPoses = poses(vSet);

% Find the feature points in the view set
xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);

% Apply bundle adjustment in the view set
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
    tracks, camPoses, intrinsics, FixedViewId=1, ...
    PointsUndistorted=true);

% Update the refined camera poses
vSet = updateView(vSet, camPoses);

% Plot the position of the cameras
camPoses = poses(vSet);

figure;
plotCamera(camPoses, Size=.2);
hold on

% Plot the feature points
pcshow(xyzPoints, VerticalAxis='y', VerticalAxisDir='down', MarkerSize= 45);
grid on
hold off

% Adjust the figure
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-20, loc1(1)+40]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);

%% Fourth Milestone

% Extract frame images from a video
imgs = extractFramesFromVideo('Dataset/video.MOV', 1);

% Convert the images into double and gray images
numImgs = size(imgs, 4);
for i = 1:numImgs
    imgsDoubleGray(:,:,i) = imresize(im2double(rgb2gray(imgs(:,:,:,i))), 0.3);
end

% Load the camera parameters
% The camera parameters are obtained by the MatLab camera calibration app.
load cameraParams.mat

intrinsics = cameraParams.Intrinsics;

% Sep up the variables for feature tracking
dispRow = 30;
dispCol = 30;
SSDkernelSize = 40;
thresholdDetecting = 0.000001;
thresholdMatching = 30;

% Sep up the variables for matrix factorization
iteration = 1000;
lambda = 1;
lambdaStep = 8;
thresholdConvergence = 1e-10;

% Sep up the variables for bundle adjustment
iterationHelper = 5000;
Confidence = 20;
MaxDistance = 50;
MaxNumTrials = 30;

numNextImgs = 4;

% Initialize the first image view
vSet = imageviewset;

[rows, cols] = find(detectFeatures(imgsDoubleGray(:,:,1), thresholdDetecting));
currPoints = [cols, rows];
vSet = addView(vSet, 1, rigidtform3d, Points=currPoints);

numPrevMatchedNotDetected = 0;

% Loop through the set of images
for i = 1:size(imgsDoubleGray, 3)-1
    
    % Find the last index of the images
    last = i+numNextImgs;
    
    if last >= size(imgsDoubleGray, 3)
       last = size(imgsDoubleGray, 3);
    end

    % Track features in consecutive frames
    [rows, cols] = trackFeatures(imgsDoubleGray(:,:,i:i+numNextImgs), dispRow, dispCol, SSDkernelSize, thresholdDetecting, thresholdMatching);

    % Create an observation matrix
    Onan = [cols; rows];
    
    % Create a weight matrix
    W = ones(size(Onan,1), size(Onan,2));
    W(isnan(Onan)) = 0;

    % Initalize A and B
    while sum(isnan(Onan), 'all') ~= 0
        [rowsnan, colsnan] = find(isnan(Onan));
        Onan(sub2ind([size(Onan,1) size(Onan,2)],rowsnan,colsnan)) = Onan(sub2ind([size(Onan,1) size(Onan,2)],rowsnan-1,colsnan));
    end

    Onan = Onan-mean(Onan,2);

    [A,S,B] = svd(Onan, 'econ');
    B = S*B';

    % Find the estimated A and B
    Onew = matrixFactorization(OnanCentered, W, iteration, lambda, lambdaStep, thresholdConvergence, A, B);
    
    % Update the missing values
    Onew = round(Onew+centroid);
    Onan(find(W)) = Onew(find(W));

    % Store the matched features bewteen the current image and the next
    % image
    currPoints = [Onan(1,:); Onan(2+numNextImgs,:)]';
    [rows, cols] = find(detectFeatures(imgsDoubleGray(:,:,i+1), thresholdDetecting));
    nextPointsMatched = [Onan(2,:); Onan(3+numNextImgs,:)]';
    nextPointsDetected = [cols, rows];
    [nextPointsMatchedNotDetected, indexM] = setdiff(nextPointsMatched, nextPointsDetected, 'rows', 'stable');

    nextPoints = [nextPointsMatchedNotDetected; nextPointsDetected];
    
    % Calculate the relative position of the camera for the next image
    [relPose, inlierIdx] = helperEstimateRelativePose(...
        currPoints, nextPointsMatched, intrinsics, iterationHelper, Confidence, MaxDistance, MaxNumTrials);
    
    % Find the poisition of the camera for the current image
    currPose = poses(vSet, i).AbsolutePose;
        
    % Calculate the position of the camera for the next image
    nextPose = rigidtform3d(currPose.A*relPose.A);
    
    % Add the next image view
    vSet = addView(vSet, i+1, nextPose, Points=nextPoints);

    % Make connections bewteen the current and next images using the
    % matched features
    currIndexMatches = find(inlierIdx);

    firstNextIndexMatches = inlierIdx;
    firstNextIndexMatches(setdiff(1:end, indexM)) = 0;
    firstNextIndexMatches = find(firstNextIndexMatches);

    secondNextIndexMatches = inlierIdx;
    secondNextIndexMatches(indexM) = 0;
    secondNextIndexMatches = find(secondNextIndexMatches);

    vSet = addConnection(vSet, i, i+1, relPose, Matches=[numPrevMatchedNotDetected+currIndexMatches, [firstNextIndexMatches; secondNextIndexMatches]]);
    
    numPrevMatchedNotDetected = size(nextPointsMatchedNotDetected, 1);

    % Find a track in the view set
    tracks = findTracks(vSet);

    % Find the camera positions in the view set
    camPoses = poses(vSet);

    % Find the feature points in the view set
    xyzPoints = triangulateMultiview(tracks, camPoses, intrinsics);
    
    % Apply bundle adjustment in the view set
    [xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(xyzPoints, ...
        tracks, camPoses, intrinsics, FixedViewId=1, ...
        PointsUndistorted=true);

    % Update the refined camera poses
    vSet = updateView(vSet, camPoses);
end

% Plot the position of the cameras
camPoses = poses(vSet);

figure;
plotCamera(camPoses, Size=.2);
hold on

% Plot the feature points
pcshow(xyzPoints, VerticalAxis='y', VerticalAxisDir='down', MarkerSize= 45);
grid on
hold off

% Adjust the figure
loc1 = camPoses.AbsolutePose(1).Translation;
xlim([loc1(1)-20, loc1(1)+40]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);

%% Acknowlegements
% The functions gkern and maxima are obtained from Professor Jerod Weinman.
%
% The function detectFeatures is originally from the Feature Detection lab
% and is slightly modified to pass a threshold.
%
% The idea for feature detection is from the Stereo Disparity lab.
% 
% The idea for matrix factorization with the damped Newton method is from
% the paper "Damped Newton Algorithms for Matrix Factorization with Missing
% Data"
%
% The matrix factorization code is obtained from the author's code on his
% website: https://www.robots.ox.ac.uk/~amb/, and we modified the code on
% our purpose based on our understanding of the paper.
%
% The function helperEstimateRelativePose is obtained from the MatLab example "Structure From Motion From
% Multiple Views". Also, the code for bundle adjustment is obtained from
% the same MatLab example, and we modified the code on our purpose.