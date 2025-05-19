clear;
clc;

%% Load data
fs = 48000;

% Load X2 and X4 - positioned on the balcony
left = load("./nassau_LoV_synthesis/h_lsp_X2.mat");
right = load("./nassau_LoV_synthesis/h_lsp_X4.mat");

% Scale h so that the filter has gain = 1 at maximum passband
h_l = left.h_lsp / max(max(abs(fft(left.h_lsp))));
h_r = right.h_lsp / max(max(abs(fft(right.h_lsp))));

%% Scale audio and calibrate loudspeakers
start_amplitude = 0.01; % -40 dB
calibration_table = readtable("loudspeaker_calibration/calibrations.xlsx");
calibration_amplitudes = table2array(calibration_table(:, 4));

total_amplitudes = start_amplitude * calibration_amplitudes;

%% Convolute audio with dry audio signals

singer1 = audioread("Sollazzo_04_02_2025_LOV_mix_v1.wav");

fft_len = size(singer1, 1) + size(h_l, 1);

SINGER1_l = fft(singer1(:, 1), fft_len);
SINGER1_r = fft(singer1(:, 2), fft_len);

H_l = fft(h_l, fft_len);
H_r = fft(h_r, fft_len);

out_l = ifft(SINGER1_l .* H_l) .* total_amplitudes.';
out_r = ifft(SINGER1_r .* H_r) .* total_amplitudes.';

out = out_l + out_r;

%% Write to file
audiowrite("synthesized_scenarios/SDM_CappellaMariana_Nassau_balcony_test.wav", out, fs, ...
           'BitsPerSample', 32);
