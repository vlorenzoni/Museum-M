clear;
clc;

%% Load data
fs = 48000;

% Load X
load("srirs/SDM_Brou_pos_MM3.mat", 'ref_p');

% Scale h so that the filter has gain = 1 at maximum passband
h = ref_p / max(max(abs(fft(ref_p))));

%% Scale audio and calibrate loudspeakers
start_amplitude = 0.01; % -40 dB

calibration_table = readtable("loudspeaker_calibration/calibrations.xlsx");
calibration_amplitudes = table2array(calibration_table(:, 4));

% I CAN PLAY HERE WITH THE NUMBER OF SPEAKERS

total_amplitudes = start_amplitude  * calibration_amplitudes;

%% Convolute audio with dry audio signals

singer1 = audioread("dry_excitations/Sollazzo_04_02_2025_LOV_mix_v1.wav");
singer1 = singer1(:, 1);

fft_len = size(singer1, 1) + size(h, 1);

SINGER1 = fft(singer1, fft_len);

H = fft(h, fft_len);

out = ifft(SINGER1 .* H) .* total_amplitudes.';

%% Write to file
disp("Synthesis complete. Writing to file...");
audiowrite("synthesized_scenarios/mono_Sollazzo_Brou.wav", out, fs, ...
           'BitsPerSample', 32);

