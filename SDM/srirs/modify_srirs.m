clear all
close all

srirsFolder = "./";
srirs_files = dir(fullfile(srirsFolder, "*.mat"));
fs = 48000;
split_samples = 1000;

for i = 1:length(srirs_files)

    file = srirs_files(i).name;
    srir = load(fullfile(srirsFolder, file));  
    plot(srir.ref_p{1});hold on;
    plot(srir.ref_p{1}(1:split_samples));hold off

    DOA = srir.DOA{1}(1:split_samples,:);
    ref_p = srir.ref_p{1}(1:split_samples);

    save("split_rirs/" + srirs_files(i).name(1:end-4) + "_direct.mat", "DOA", "ref_p")

    file = srirs_files(i).name;
    srir = load(fullfile(srirsFolder, file));  

    DOA = srir.DOA{1}(split_samples:end,:);
    ref_p = srir.ref_p{1}(split_samples:end);

    save("split_rirs/" + srirs_files(i).name(1:end-4) + "_reverb.mat", "DOA", "ref_p")


end