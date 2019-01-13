Audio2Speech-Music-classification,
Apostolou Orestis, 13/01/2019

This project is implemented in MATLAB. The MIRtoolbox must be installed in order to execute this.

The project consists of 2 tasks. The first is to classify an audio track as music or speech.
The second is to divide a mixed audio file (both music and speech) to the music segments and the speech segments.

The first task uses a cubic svm trained model. The below features were used for classifying:
RMS energy
Zero-crossing-rate
Rolloff ( the frequency under which exists the 85% of the full energy)
Spectral flatness
1-13 MFCCs
1-19 Spectral Flatness

As training set, we used the MUSAN dataset, GTZAN dataset and some other random music files from a personal library.

A first step to the feature extraction was to divide the full audio tracks to frames of 1024 samples, and then to statistically aggregate 16 frames each time, in order to get the 16-frame variance and mean. Thus, the full extracted features were 72 (36 mean and 36 variable features).

The first task divides the testing files in frames, then in 16-frames segments, then uses the regression of the training model and then a simple majority-voting algorithm to classify the full audio file.

The second task uses a novelty detection algorithm to detect moments that are possible to have a music to speech or speech to music transition, then divides the full audio file to segments based on the possible transition points, and then uses the first task to classify each of the segments. The result is a csv file with the segments of music and speech.

Note that testing audio must be mono, sampled at 22050 Hz.

Files in the repo:

cubic_svm_7_4.mat: the trained model

dataExtraction_v7_5.m: the executable for extracting the above features from a music and a speech dataset. In order to execute, you must give first the music directory, and then the speech directory.

data_v7_32.csv: the csv with the extracted features

inputPrep_v7_5.m: don't execute this, it is an assisting executable for bringing the audio file in the finalized form

melFilter.m: don't run this, it is an assisting executable to implement the mel filter bank

task1_v_7_5.m: the first task. Run this and give the directory in which you have inputPrep_v7_5.m, melFilter.m, cubic_svm_7_4.mat and the testing audio file.

task2_v_7_5.m: the second task. Run this and give the directory in which you have melFilter.m, cubic_svm_7_4.mat and the testing audio file.

task2_fast_v_7_5.m: the second task, in fast mode. Run this and give the directory in which you have melFilter.m, cubic_svm_7_4.mat and the testing audio file.

test2mono.wav: an audio test file

wavscript.sh: shell script to convert audio files in 22050 Hz mono. Copy it in a directory with wav files and run it as . ./wavscript.sh to get the copyfiles in the desired form (Ubuntu terminal)

For questions contact me at orestisapostolou@yahoo.gr
