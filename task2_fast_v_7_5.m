% Apostolou Orestis
% 10/01/2018
%
% AUTH University, Electrical Engineering Department 
%
% finalized version
% trained on svm cubic
%
% inputs must be sampled at 22050 hz, Mono
%
% if you have excel installed uncomment the two last lines to get the
% results.xls
%
% mirToolbox is necessary for this executable
% tested on MATLAB R2018a
%
% contact me at orestisapostolou@yahoo.gr if you have any questions
% about the code
% 
% the difference in the fast model is that only the full data matrix is
% used for calculating the novelty vector. This causes minor decrease in
% detection of major changes in the testing sample, but results in a much
% faster implementation
%
% When asked for a directory, give th directory in which cubic_svm_4.mat
% exist, along with the testing wav.
%
% the result is written in resultsraw.csv, or results.xls



function task2_fast_v_7_5


myDir = uigetdir;                       %gets directory
myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
cd(myDir);
load 'cubic_svm_7_4.mat';               %loads the trained model
melMatrix = melFilter;                  %loads the script which constructs melfilternbanks
numBlocks = 16;                         %number of blocks (it is good to be the same as in training
for k = 1:length(myFiles)
    baseFileName = {myFiles(k).name};
    music = miraudio(baseFileName);
    samplingCell = get(music,'Sampling');        %gets audio sampling rate (it must be 22050 in order to have the right results)
    samplingRate = cell2mat(samplingCell);
    samples = 1024;                              % each frame is created by 1024 samples
    frame = mirframe(music, samples, 'sp', samples, 'sp');      %decompose the full audio to frames
    spectrum = mirspectrum(frame, 'Window', 'hanning');         %FFT to get spectrum for each frame, hanning window used
    
    mirMusicRMS = mirrms(frame);                               % RMS energy for each frame
    musicRMS = mirgetdata(mirMusicRMS);
    musicRMS = eliminateNans(musicRMS,8);                      % eliminate false Nan values 
    
    mirMusicZCR = mirzerocross(frame);                        % Zero-crossing-rate for each frame
    musicZCR = mirgetdata(mirMusicZCR);  
    musicZCR = eliminateNans(musicZCR,8);                     % eliminate false Nan values 
    
    dataSpec = mirgetdata(spectrum);
    [~, cols] = size(musicZCR);
    [specRows, ~] = size(dataSpec);
    [~, melCols] = size(melMatrix);
    
    musicBandsFlatness = zeros(melCols,cols);                 % calculate the spectral flatness per band using mel filterbank
    helpV = zeros(specRows, melCols);                         % in this executable, spectrum is decomposed to 19 bands
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
    
    musicSpectralFlatness = zeros(1,cols); 
    for i = 1:cols                                               %gets full spectral flatness
        musicSpectralFlatness(1,i) = spectralFlatness((dataSpec(1:specRows,i)));
    end
    
    clearvars dataspec;                                          %dataspec needs much memory when audio samples are long
    mirMusicRolloff = mirrolloff(spectrum,'MinRMS',0.000001);     % Roll-off of each frame (an upper frequency. 0-frequency has
    musicRolloff = mirgetdata(mirMusicRolloff);                   % 85% of the full energy of the spectrum)
    musicRolloff = eliminateNans(musicRolloff,8);                 % eliminate false Nan values 
    
    
    mirMusicMFCC = mirmfcc(spectrum);                           % gets 13 MFCCs
    musicMFCC = mirgetdata(mirMusicMFCC);
    musicMFCC = eliminateNans(musicMFCC,8);                     % eliminate false Nan values 
    
    
    tempMusicDataFull = [musicRMS; musicZCR; musicRolloff; musicSpectralFlatness; musicMFCC; musicBandsFlatness];    %concentrates all features
    simMatrix4 = similarityMatrix(tempMusicDataFull, 20);          % calculates similarity using width = 20
    checkMatrix = checkerBoardKernel(20, 0.1);                     % creates checkerboard matrix with k = 20, p =0.1
    noveltyVector4 = novelty(simMatrix4, checkMatrix);             % creates novelty vector. This is the most time-costing function of the executable
    
    %tempData = [musicRMS; musicZCR; musicRolloff; musicSpectralFlatness];
    %simMatrix1 = similarityMatrix(tempData, 20);
    %noveltyVector1 = novelty(simMatrix1, checkMatrix);
    
    
    [N,~]=size(noveltyVector4);
    endVal = N*samples/samplingRate;
    t=linspace(0,endVal,N);
    
    %plot(t, noveltyVector4);  %uncomment this line to plot noveltyVector4
    
    %plot(t,noveltyVector1);  %uncomment this line to plot noveltyVector1
    
    %tempData = musicMFCC;
    %simMatrix2 = similarityMatrix(tempData, 20);
    %noveltyVector2 = novelty(simMatrix2, checkMatrix);
    
    %plot(t,noveltyVector2);   %uncomment this line to plot noveltyVector2
   
    %tempData = musicBandsFlatness;
    %simMatrix3 = similarityMatrix(tempData, 20);
    %noveltyVector3 = novelty(simMatrix3, checkMatrix);
    
    %plot(t,noveltyVector3);  %uncomment this line to plot noveltyVector3
    
    %peakMat1 = peakMatrix(noveltyVector1, 11, 1.4, 4); % we don't use it
    %peakMat2 = peakMatrix(noveltyVector2, 11, 11.5, 2); 
    %peakMat3 = peakMatrix(noveltyVector3,11, 1.6, 3);
    peakMat4 = peakMatrix(noveltyVector4,11, 1.5, 1);    
    
    % the above lines get the peaks of novelty. Second argument is peaks' min distance, 3rd argument is a lower threshold
    % and 4th argument is priority, in case of merging many peakMatrices
    
    %peakMatTemp = mergeMat(peakMat2,peakMat3);
    %peakMat = mergeMat(peakMatTemp,peakMat4);   %merging peakmat 2,3 and 4
    %peakMat = finalizeMerged(peakMat,11);       % finalize full peakMatrix
    
    tempData = [musicRMS; musicZCR; musicRolloff; musicSpectralFlatness; musicMFCC; musicBandsFlatness];
    clearvars tempMusicData tempMusicDataFull;
    %classifiedMatrix1 = classificationMethod(peakMat1,tempData,cubicSVM7_4, numBlocks, samples, samplingRate);
    %classifiedMatrix2 = classificationMethod(peakMat2,tempData,cubicSVM7_4, numBlocks, samples, samplingRate);
    %classifiedMatrix3 = classificationMethod(peakMat3,tempData,cubicSVM7_4, numBlocks, samples, samplingRate);
    %classifiedMatrix4 = classificationMethod(peakMat4,tempData,cubicSVM7_4, numBlocks, samples, samplingRate);
    classifiedMatrix = classificationMethod(peakMat4,tempData,cubicSVM7_4, numBlocks, samples, samplingRate);
    
    
    %fin = num2time(classifiedMatrix);
    %xlswrite('results.xls',fin);
    
    % uncomment the 2 lines above to get results.xls ( if you have excel
    % installed)
    
    xlswrite('resultsraw.xls',classifiedMatrix);
    
    
end

end

function A = majorityVoting(prediction, rat)

% uses the prediction model in order to classify a longer "window".
% It uses majority voting of the shorter predictions
% and it uses the regression results instead of the classification results 
%  in order to be more accurate


[cols,~] = size(prediction);
sum = 0;
for i=1:cols-1
    sum=sum+prediction(i,2);
end

sum = sum+prediction(cols,2)*rat;

if (sum>0)
    A = 1;                         %1 is for music, -1 for speech
else
    A = -1;
end

end

function B = similarityMatrix(features, width)

% it calculates the similarity function of a set of features




    [~,cols] = size(features);
    B = sparse(1:cols,1:cols,1,cols,cols,cols*(width+1));
    %B = ones(cols,cols);
    width = width-2;
    uplimit = min(cols, width/2);
    for i=1:cols-1
        for j=i+1:i+uplimit+1
             A = features(:,i);
             C = features(:,j);
             B(i,j) = dot(A,C)/(norm(A)*norm(C));
                %B(i,j) = vectCos(features(:,i), features(:,j));
             if (j==cols)
                 break;
             end
        end
    end
    
    % the second half of the matrix is not needed beacause the matrix is
    % symmetric
    
   % for j=1:cols-1
    %    for i=j+1:j+uplimit+1
     %       B(i,j)=B(j,i);
      %      if (i==cols)
       %          break;
        %    end
       % end
    %end
end

function C = vectCos(vect1, vect2)

% not used

    C = dot(vect1,vect2)/(norm(vect1)*norm(vect2));
    

end

function D = checkerBoardKernel(fullWidth, p)

% creates a checkerboard Kernel of fullWidth, with expotential factor p 

    half = fullWidth/2;
    A4 = ones(half, half);
    
    for i=1:half
        for j=i:half
            A4(i,j) = exp( -p*(j-1)^2);
        end
    end
    for j=1:half-1
        for i=j+1:half
            A4(i,j) = A4(j,i);
        end
    end
    A1 = rot90(A4,2);
    D = [A1, (-1)*fliplr(A1); (-1)*fliplr(A4), A4];    
    
        

end

function E = novelty(similarityMatrix, kernel)

% creates the novelty vector using the similarity matrix and the
% checkerboard kernel

    [krows, ~] = size(kernel);
    [srows, ~] = size(similarityMatrix);
    E=zeros(srows,1);
    half = krows/2;
    for index=5:srows-4            % first 5 and last 5 rows are not used
        for i=1:krows
            for j=i:krows
                indi = i+index-half;
                indj = j+index-half;
                if (indj>0 && indi>0 && indj<=srows && indi<=srows)
                    if i==j
                        multFactor = 1;
                    else
                        multFactor = 2;
                    end
                    E(index,1)=E(index,1)+similarityMatrix(indi,indj)*kernel(i,j)*multFactor;
                end
            end
        end
    end
    %maximum = max(E);      % uncomment this to normalize vector between
                            % min and max
    %minimum = min(E);
    %E(:,1) = (E(:,1) - minimum)/(maximum - minimum);
end

function F = peakMatrix(noveltyVector, minPeaks, peakFactor, priority)

% it finds the local peaks of the novelty vector at a minPeaks distance,
% which are over peakFactor threshold


    [peaks, locs] = findpeaks(noveltyVector, 'MinPeakDistance', minPeaks);
    peakRawMatrix = [peaks, locs];
    [peakRawSize,~] = size(peakRawMatrix);
    peakMatrix = zeros(peakRawSize,3);
    counter=1;
    for i=1:peakRawSize
        if peakRawMatrix(i,1)>peakFactor
            peakMatrix(counter,1:2)=peakRawMatrix(i,1:2);
            counter = counter+1;
        end
    end
    peakMatrix( ~any(peakMatrix,2), : ) = [];
    [rows,~] = size(peakMatrix);
    peakMatrix(1:rows,1) = priority;
    F = peakMatrix;
    
end

function G = classificationMethod(peakM,data,classificationModel, numBlocks, numSamples, samplingFrequency)
    
% it classifies all of the windows between the peaks, using
% classificatioModel and majority voting. 


    [peakMSize,~] = size(peakM);
    [dataRows,dataCols] = size(data);
    peakM(peakMSize+1,1:3) = [0, dataCols, 0];
    peakMat = peakM;
    [peakMatSize,~] = size(peakMat);
    for i=1:peakMatSize
       if (i==1)
           startPoint = 1;
           endPoint = peakMat(1,2);
       else
           startPoint = peakMat(i-1,2)+1;
           endPoint = peakMat(i,2);
       end
       numSteps = endPoint - startPoint + 1;
       tempCeil = ceil(numSteps/numBlocks);
       tempFloor = floor(numSteps/numBlocks);
       
       % if a window is smaller than 16 frames, it must have a smaller
       % percentage in majority voting
       % if tempfloor = tempceil, we have a perfect division and the last
       % window is not smaller. In any other case the last window is
       % smaller
       
       
       if (tempFloor == tempCeil)
          meanData = zeros(dataRows,tempFloor);
          varData = zeros(dataRows,tempFloor);
          for m=1:tempFloor
              tPoint= (m-1)*numBlocks+startPoint-1;
              for n =1:dataRows
                  vector = data(n,tPoint+1:tPoint+numBlocks);
                  meanData(n,m) = mean(vector);
                  varData(n,m) = var(vector);
              end
          end
          
          musicOnes = ones(1, tempFloor);
          fullData = [meanData ; varData; musicOnes];
          ratio = 1;   % voting ration in majority voting
       else
          meanData = zeros(dataRows,tempCeil);
          varData = zeros(dataRows,tempCeil);
          for m=1:tempFloor
              tPoint= (m-1)*numBlocks+startPoint-1;
              for n =1:dataRows
                  vector = data(n,tPoint+1:tPoint+numBlocks);
                  meanData(n,m) = mean(vector);
                  varData(n,m) = var(vector);
              end
          end
          tPoint= (tempCeil-1)*numBlocks+startPoint-1;
          ratio = (endPoint - tPoint)/numBlocks;
          for n=1:dataRows
              vector = data(n,tPoint+1:endPoint);
              meanData(n,tempCeil) = mean(vector);
              varData(n, tempCeil) = var(vector);
          end
          musicOnes = ones(1, tempCeil);
          fullData = [meanData ; varData; musicOnes]; 
       end
       
       dataT = transpose(fullData);
       t = array2table(dataT);
       
       % the line below is necessary if you use matlab's classification
       % learner. The names must be the same as in training
       
       t.Properties.VariableNames(1:73)={'VarName1','VarName2','VarName3','VarName4','VarName5','VarName6','VarName7','VarName8','VarName9','VarName10','VarName11','VarName12','VarName13','VarName14','VarName15','VarName16','VarName17','VarName18','VarName19','VarName20','VarName21','VarName22','VarName23','VarName24','VarName25','VarName26','VarName27','VarName28','VarName29','VarName30','VarName31','VarName32','VarName33','VarName34','VarName35','VarName36','VarName37','VarName38','VarName39','VarName40','VarName41','VarName42','VarName43','VarName44','VarName45','VarName46','VarName47','VarName48','VarName49','VarName50','VarName51','VarName52','VarName53','VarName54','VarName55','VarName56','VarName57','VarName58','VarName59','VarName60','VarName61','VarName62','VarName63','VarName64','VarName65','VarName66','VarName67','VarName68','VarName69','VarName70','VarName71','VarName72','VarName73'};
       [pred1,score] = classificationModel.predictFcn(t);
       [predSize,~] =size(score);
       
       % for any reason, if we have a nan value, the prediction predicts
       % zero
       
       for iii=1:predSize
           if isnan(score(iii,2))
               score(iii,2) = 0;
               score(iii,1) = 0;
           end
       end
       pred2 = majorityVoting(score, ratio);
       peakMat(i,3) = pred2;
    end
    
    % peakmat now has in its 3rd column the results of the prediction
    % the rest of the part finalizes the vector in order to have a correct
    % "appearance"
    
    % finalization
    finalizeVector=zeros(peakMatSize+1,1);
    finalizeVector(1,1)=1;
    finalizeVector(peakMatSize+1,1)=1;
    for i=2:peakMatSize
        if (peakMat(i,3) == peakMat(i-1,3))
            finalizeVector(i,1) = finalizeVector(i-1,1)+1;
        else
            finalizeVector(i,1) = 1;
        end
    end
    newVector = zeros(peakMatSize,5);
    fCounter = 1;
    for i=2:peakMatSize+1
        if finalizeVector(i) == 1
            newVector(fCounter,2:3)=peakMat(i-1,2:3);
            fCounter = fCounter+1;
        end
            
    end
    newVector( ~any(newVector,2), : ) = [];
    newVector(1,1)=1;
    [newVectorSize,~] = size(newVector);
    for i=2:newVectorSize
        newVector(i,1)=newVector(i-1,2)+1;
    end
    newVector(:,4:5)=(newVector(:,1:2)-1)*numSamples/samplingFrequency;
    G = newVector;
end
    

function A = eliminateNans(matrix, factor)

% it eliminates nan values using the mean of the values that are left and
% right of the selcted cell


    [rows, cols] = size(matrix);
    for i=1:rows
        for j = 1:cols
            if isnan(matrix(i,j))
                lower = max(j-factor,1);
                upper = min(j+factor, cols);
                matrix(i,j) = nanmean(matrix(i, lower:upper));
            end
        end
    end
    A = matrix;

end

function B = spectralFlatness(vector)
    
    % calculates spectralFlatness of a vector


    vv = nonzeros(vector);
    rows = size(vv);
    if rows == 0
        B = 0;
    else
        B = geomean(vv)/mean(vv);
    end
    
end

function A = mergeMat(mat1, mat2)

% it implements merge sort, ignoring same values
    
    [size1,~] = size(mat1);
    [size2,~] = size(mat2);
    A = zeros(size1+size2,3);
    mat1(size1+1,1:3) = inf;
    mat2(size2+1,1:3) = inf;
    k=1;
    l=1;
    upper = size1+size2;
    i=1;
    while i<=upper
        if mat1(k,2)<mat2(l,2)
            A(i,1:3) = mat1(k,1:3);
            k = k+1;
        elseif mat1(k,2)>mat2(l,2)
            A(i,1:3) = mat2(l,1:3);
            l = l+1;
        else
            A(i,1:3) = mat1(k,1:3);
            k = k+1;
            l = l+1;
            upper = upper-1;
        end
        i = i+1;
    end
    
    A( ~any(A,2), : ) = [];


end


function D = finalizeMerged(mat,peaks)
    
% it merges 2 peak matrices, under the limitation that min dist is "peaks".
% Priority is taken into consideration in order to keep a peak or not. 1 is
% for the best priority and 4 is for the worst

    m = 1;
    [rows,~] = size(mat);
    D = zeros(rows,3);
    for i = 1:rows
        if mat(i,1) == 1
            D(i,1:3) = mat(i,1:3);
        end
    end
    for j = 2:4
        for i =1:rows
            if mat(i,1) ==j
                next = 1;
                prev = 1;
                m=1;
                while(i>m)
                    if D(i-m,1)==0
                        m = m+1;
                    elseif (mat(i-m,1)<j && (mat(i,2)-mat(i-m,2)<=peaks)) 
                        prev = 0;
                        m=i;
                    else
                        m=i;
                    end
                    
                end
                m =1;
                while(i+m<=rows)
                    if D(i+m,1)==0
                        m = m+1;
                    elseif (mat(i+m,1)<j && (mat(i+m,2)-mat(i,2)<=peaks)) 
                        next = 0;
                        m=rows;
                    else
                        m=rows;
                    end
                    
                end
                if (next==1 && prev ==1)
                    D(i,1:3) = mat(i,1:3);
                end
            end
        end
    end
    d =111;
    D( ~any(D,2), : ) = [];

end

function num =num2time(mat)

% converts seconds to time and -1 1 to speech music
    
    [rows,~]=size(mat);
    num = strings([rows,3]);
    for i =1:rows
        
        hstart = fix(mat(i,4)/3600);
        hend = fix(mat(i,5)/3600);
        mstart = fix(mat(i,4)/60);
        mend = fix(mat(i,5)/60);
        sstart = fix(mat(i,4)/1);
        send = fix(mat(i,5)/1);
        milstart = rem(mat(i,4),1);
        milend = rem(mat(i,5),1);
        mmilstart = fix(milstart/0.1);
        mmilend = fix(milend/0.1);
        shstart = num2str(hstart);
        shend = num2str(hend);
        smstart = num2str(mstart);
        smend = num2str(mend);
        ssstart = num2str(sstart);
        ssend = num2str(send);
        smilstart = num2str(mmilstart);
        smilend = num2str(mmilend);
        num(i,1)=strcat(shstart,"::",smstart,"::",ssstart,".",smilstart);
        num(i,2)=strcat(shend,"::",smend,"::",ssend,".",smilend);
        
        if mat(i,3) == -1
            num(i,3) = "Speech";
        else
            num(i,3) = "Music";
        end
    end

end








