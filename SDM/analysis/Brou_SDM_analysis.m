clear all;

clc


IR_folder = '../rirs/brou_church';

d = dir(IR_folder);
dfolders = d([d(:).isdir]) ;
dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));

for iFold =  1:length(dfolders) % loop over positions

    % Define the .mat file name based on the folder name
    matFileName = [fullfile('mat_files', dfolders(iFold).name), '.mat'];
    dfolders(iFold).name

    % Define the path to the current folder
    IR_folder = fullfile(dfolders(iFold).folder, dfolders(iFold).name);

    % Get list of all audio files in the current folder
    Files = dir(fullfile(IR_folder, '*.wav')); % Assumes audio files are in .wav format
    numFiles = length(Files);

    % Initialize cell array to store audio data
    ir_left = cell(1, numFiles);

    % Read each audio file and store in ir_left
    for k = 1:numFiles
        FileNames = Files(k).name;
        ir_left{k} = audioread(fullfile(IR_folder, FileNames)); % Store audio data in cell array
    end

    %% Read the data
    % Read impulse response
    ir_left = cell2mat(ir_left);


    % HACK put right channel same as left channel
    ir_right = ir_left;
    % Read stereo signal



    fs = 48e3;
    a = createSDMStruct('DefaultArray','GRASVI25','fs',fs);

    %% Calculate the SDM coefficients
    % Solve the DOA of each time window assuming wide band reflections, white
    % noise in the sensors and far-field (plane wave propagation model inside the array)
    DOA{1} = SDMPar(ir_left, a);

    % Here we are using the top-most microphone as the estimate for the
    % pressure in the center of the array
    P{1} = ir_left(:,5);

    % Same for right channel
    DOA{2} = SDMPar(ir_right, a);
    P{2} = ir_right(:,5);


    v = createVisualizationStruct('DefaultRoom','VeryLarge',...
        'name','Brou church','fs',fs);
    % For visualization purposes, set the text interpreter to latex
    set(0,'DefaultTextInterpreter','latex')
    
    %% Draw the spatio temporal visualization for each section plane
    v.plane = 'lateral';
    spatioTemporalVisualization(P, DOA, v)


    ref_p = P;
    save("../srirs/SDM_Brou_" + dfolders(iFold).name + ".mat", "DOA", "ref_p")

end