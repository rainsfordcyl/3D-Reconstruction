function imgs = extractFramesFromVideo(videoPath, frameRate)
% extractFramesFromVideo Extracts frame images from a video input
%
% imgs = extractFramesFromVideo(videoPath, frameRate) where videoPath is
% a string that indicates the path of a video input, frameRate is the
% frameRate used to sample the frame images, and imgs are the sampled frame
% images
% 
% CSC 262 Final Paper

% Read a video using the VideoReader object
video = VideoReader(videoPath);

% Set the start and end frames
startFrame = 1;
endFrame = video.NumFrames;

% Find the interval based on frameRate
frameInterval = round(video.FrameRate/frameRate);

% Loop through the frames
index = 1;
for i = startFrame:frameInterval:endFrame
    % Read the current frame
    imgs(:,:,:,index) = read(video, i);
    index = index+1;
end

end