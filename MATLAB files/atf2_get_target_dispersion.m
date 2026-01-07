function state = atf2_get_target_dispersion(deltafreq)
% deltafreq [kHz]


[names,Z,disp] = textread('bpmdisp.txt', '%s %f %f');

DR_freq = 714e3; % 714 MHz in kHz
DR_momentum_compaction = 2.1e-3;

dP_P = -deltafreq / DR_freq / DR_momentum_compaction;

nBpms = numel(names);

state.bpms.name = names;
state.bpms.X = 1e6 * disp * dP_P;
state.bpms.Y = zeros(nBpms, 1);
state.bpms.Z = Z;
state.bpms.TMIT = ones(nBpms, 1);

[Z,I] = sort(state.bpms.Z);
  
state.bpms.name = state.bpms.name(I);
state.bpms.X = state.bpms.X(I,:);
state.bpms.Y = state.bpms.Y(I,:);
state.bpms.Z = state.bpms.Z(I);
state.bpms.TMIT = state.bpms.TMIT(I,:);

