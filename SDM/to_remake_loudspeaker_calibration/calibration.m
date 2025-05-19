clear;
clc;

%% Get correct amplitude level of 60 dB SPL for all loudspeakers

start_amplitude = 0.01; % -40 dB

calibration_table = readtable("calibrations.xlsx");
calibration_amplitudes = table2array(calibration_table(:, 4));

total_amplitudes = start_amplitude * calibration_amplitudes;

%% Create noise signals for each loudspeaker of 60 dB SPL
fs = 48000;
noise = randn(120 * fs, 1);
noise = (noise / max(abs(noise))) * total_amplitudes.';

%% Write calibration noise to file
audiowrite("calibration_test_noise.wav", noise, fs);
