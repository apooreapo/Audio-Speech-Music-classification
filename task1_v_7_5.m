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
% first give the folder where cubic_svm_7_4.mat is, then the folder where
% inputPrep_v7_5 is, then the folder where the wav which is about to be
% classified. Optionally put these 3 files in the same folder.
% 
% The output prints in the console, and you can uncomment the plot in line
% 49 to see the regression plot for each aggregated frame




function task1_v_7_5
myDir = uigetdir; %gets directory where model is
cd(myDir);
load 'cubic_svm_7_4.mat';

myDir2 = uigetdir;
cd(myDir2);
%gets directory where inputPrep is

sample1 = inputPrep_v7_5;
[~,score] = cubicSVM7_4.predictFcn(sample1);       % we take the regression model
[predSize,~] =size(score);
for i=1:predSize
    if isnan(score(i,2))
        score(i,2) = 0;
        score(i,1) = 0;
    end
end
pred2 = majorityVoting(score);

%plot(score(:,2), '.r');                            % plots the sample's regression for each aggregated frame
formatSpecMusic="Your file is a Music File";
formatSpecSpeech="Your file is a Speech File";

if (pred2 == 1)
    fprintf(formatSpecMusic);
else
    fprintf(formatSpecSpeech);
end

end

function A = majorityVoting(prediction)
cols = size(prediction);
sum = 0;
for i=1:cols
    sum=sum+prediction(i,2);
end

if (sum>0)
    A = 1;
else
    A = -1;
end

end


