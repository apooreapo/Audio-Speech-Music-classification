% Apostolou Orestis
% 10/01/2018
%
% AUTH University, Electrical Engineering Department 
%
% finalized version
% 
%
% inputs must be sampled at 22050 hz, Mono
%
%
%
% mirToolbox is necessary for this executable
% tested on MATLAB R2018a
%
% contact me at orestisapostolou@yahoo.gr if you have any questions
% about the code
% 
% first give the folder of the music files (wavs), and then give the folder
% of the speech files in order to extract the input data for the training
% model





function dataExtraction_v7_5

musicData = [];
tempData= [];
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
melMatrix = melFilter;
cd(myDir);
numBlocks = 16;
for k = 1:length(myFiles)
    baseFileName = {myFiles(k).name};
    music = miraudio(baseFileName);
    frame = mirframe(music, 1024, 'sp', 1024, 'sp');   % divides the full audio into frames
    spectrum = mirspectrum(frame, 'Window', 'hanning');    % constructs the spectrum of each frame
    dataSpec = mirgetdata(spectrum);                      % gets the spectrum
    
    
    mirMusicRMS = mirrms(frame);
    musicRMS = mirgetdata(mirMusicRMS);                 % gets the rms energy of each frame
    
    mirMusicZCR = mirzerocross(frame);                 % gets zero-crossing-rate of each frame
    musicZCR = mirgetdata(mirMusicZCR);  
    
    [~, cols] = size(musicZCR);
    [specRows, ~] = size(dataSpec);
    [~, melCols] = size(melMatrix);
    
    mirMusicRolloff = mirrolloff(spectrum);             % getse the rolloff of each frame (85%)
    musicRolloff = mirgetdata(mirMusicRolloff);
    
    musicBandsFlatness = zeros(melCols,cols);          % gets the spectral flatness of each mel-band for each frame
    helpV = zeros(specRows, melCols);
    for i=1:cols
        for j=1:melCols
            for h=1:specRows
                if (melMatrix(h,j)~=0)
                    helpV(h,j) = melMatrix(h,j)*dataSpec(h,i);      % melMatrix is a mel-filterbank
                end
            end
        end
        for j=1:melCols
            musicBandsFlatness(j,i) = spectralFlatness(helpV(1:specRows,j));
        end
    end
    
    mirMusicSpectralFlatness = mirflatness(spectrum);         % gets full spectral flatness of each frame
    musicSpectralFlatness = mirgetdata(mirMusicSpectralFlatness);
    
    mirMusicMFCC = mirmfcc(spectrum);                 % gets the MFCCs of each frame
    musicMFCC = mirgetdata(mirMusicMFCC);

    tempFloor = floor(cols/numBlocks);
    
    %tempMusicData = musicBandsFlatness;
    tempMusicData = [musicRMS; musicZCR; musicRolloff; musicSpectralFlatness; musicMFCC; musicBandsFlatness];
    [rows,~]=size(tempMusicData);
    meanData = zeros(rows,tempFloor);
    varData = zeros(rows,tempFloor);
    
    for m=1:tempFloor
        for n=1:rows
            t = (m-1)*numBlocks;
            meanData(n,m) = mean(tempMusicData(n, t+1:t+16));        % gets mean and variance of the frames (16 aggregated frames)
            varData(n,m) = var(tempMusicData(n, t+1:t+16));
        end
    end
    
    musicOnes = ones(1, tempFloor);
    tempData = [meanData ; varData; musicOnes]; 
    musicData = [musicData,tempData];
    
    
end
    
myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
cd(myDir);
for k = 1:length(myFiles)
    baseFileName = {myFiles(k).name};
    music = miraudio(baseFileName);
    frame = mirframe(music, 1024, 'sp', 1024, 'sp');
    spectrum = mirspectrum(frame, 'Window', 'hanning');
    dataSpec = mirgetdata(spectrum);
    
    
    mirMusicRMS = mirrms(frame);
    musicRMS = mirgetdata(mirMusicRMS);
    
    mirMusicZCR = mirzerocross(frame);
    musicZCR = mirgetdata(mirMusicZCR);  
    
    [~, cols] = size(musicZCR);
    [specRows, ~] = size(dataSpec);
    [~, melCols] = size(melMatrix);
    
    mirMusicRolloff = mirrolloff(spectrum);
    musicRolloff = mirgetdata(mirMusicRolloff);
    
    musicBandsFlatness = zeros(melCols,cols);
    helpV = zeros(specRows, melCols);
    for i=1:cols
        for j=1:melCols
            for h=1:specRows
                if (melMatrix(h,j)~=0)
                    helpV(h,j) = melMatrix(h,j)*dataSpec(h,i);
                end
            end
        end
        for j=1:melCols
            musicBandsFlatness(j,i) = spectralFlatness(helpV(1:specRows,j));
        end
    end
    
    mirMusicSpectralFlatness = mirflatness(spectrum);
    musicSpectralFlatness = mirgetdata(mirMusicSpectralFlatness);
    
    mirMusicMFCC = mirmfcc(spectrum);
    musicMFCC = mirgetdata(mirMusicMFCC);

    tempFloor = floor(cols/numBlocks);

    
    tempMusicData = [musicRMS; musicZCR; musicRolloff; musicSpectralFlatness; musicMFCC; musicBandsFlatness];
    [rows,~]=size(tempMusicData);
    meanData = zeros(rows,tempFloor);
    varData = zeros(rows,tempFloor);
    
    for m=1:tempFloor
        for n=1:rows
            t = (m-1)*numBlocks;
            meanData(n,m) = mean(tempMusicData(n, t+1:t+16));
            varData(n,m) = var(tempMusicData(n, t+1:t+16));
        end
    end
    
    musicOnes = (-1)*ones(1, tempFloor);  % that's the only difference, as we now talk about speech
    tempData = [meanData ; varData; musicOnes]; 
    musicData = [musicData,tempData];
    
    
end

dataT = transpose(musicData);
cd ..;
xlswrite('data_v7_5.xls',dataT);
end


function B = spectralFlatness(vector)
    
    vv = nonzeros(vector);
    rows = size(vv);
    if rows == 0
        B = 0;
    else
        B = geomean(vv)/mean(vv);
    end
    
    
end