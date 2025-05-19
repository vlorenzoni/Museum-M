%% Set up variables
clear; clc;


RIRpart = "direct"; % other option "reverb"

lsp_prefixes = ["MM1", "MM2","MM3","MM4","MM5"];


fs = 48000;

loudspeaker_cart = table2array(readtable('../layout_8loudspeaker_LOV_GENELEC.csv'));
[azi, ele, r] = cart2sph(loudspeaker_cart(:, 1), ...
                         loudspeaker_cart(:, 2), ...
                         loudspeaker_cart(:, 3));
azi = azi * 180 / pi;
ele = ele * 180 / pi;

%% Perform synthesis for all measurement positions
L = 6.5 * fs;
s = createSynthesisStruct('lspLocs', [azi, ele, r], 'snfft', L, ...
                          'fs', fs, 'c', 343);
for lsp = lsp_prefixes
        disp("SDM synthesis for measurement position: " + lsp + "_" + RIRpart )
        srir = load("../srirs/split_rirs/SDM_Brou_pos_" + lsp  + "_" + RIRpart + ".mat");
        h_lsp = synthesizeSDMCoeffs(srir.ref_p, srir.DOA, s);
        save("h_lsp_" + lsp + ".mat", "h_lsp");
end
