% Read and play first two channels of a 12-channel WAV file

% Read the WAV file
[audio, Fs] = audioread('synthesized_scenarios/SDM_Sollazzo_Brou_church_direct_MM5.wav');

% Extract first two channels
channels_1_2 = audio(:, 1:2);

% Play the audio
sound(channels_1_2, Fs);

% Optional: Display information about the audio
fprintf('Sampling rate: %d Hz\n', Fs);
fprintf('Number of samples: %d\n', size(audio, 1));
fprintf('Number of channels in original file: %d\n', size(audio, 2));
fprintf('Playing channels 1 and 2\n'); 