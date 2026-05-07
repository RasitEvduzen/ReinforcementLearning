clc; clear all; close all;
% Reaction Wheel Inverted Pendulum  SAC Agent Test
% Written By: Rasit
% Date: 06-May-2026
 
%% Load Agent
load('reactionwheel_agent.mat', 'agent')
fprintf('Agent loaded.\n')
 
%% Run Simulink
sim_ts   = 1e-2;
sim_time = 5;
 
sim('RlReactionWheel.slx');
