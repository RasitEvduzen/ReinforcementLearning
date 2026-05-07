clc; clear; close all;
% Reaction Wheel Inverted Pendulum SAC Training
% Observations: [roll, pitch, roll_rate, pitch_rate, w_wheel1, w_wheel2]
% Actions:      [u_roll, u_pitch]
% Written By: Rasit
% Date: 06-May-2026

%% Simulation Parameters
sim_ts   = 1e-2;
sim_time = 3;

%% Observation Info
obsInfo = rlNumericSpec([6 1], ...
    'LowerLimit', [-pi; -pi; -50; -50; -200; -200], ...
    'UpperLimit', [ pi;  pi;  50;  50;  200;  200]);
obsInfo.Name = 'Observations';

%% Action Info
actInfo = rlNumericSpec([2 1], ...
    'LowerLimit', [-5; -5], ...
    'UpperLimit', [ 5;  5]);
actInfo.Name = 'Actions';

%% Environment
env = rlSimulinkEnv('RLReactionWheel', 'RLReactionWheel/RL Agent', obsInfo, actInfo);

%% Actor Network
obsSize = 6;
actSize = 2;

lgA = layerGraph([
    featureInputLayer(obsSize, 'Name', 'obs')
    fullyConnectedLayer(128,   'Name', 'fc1')
    reluLayer(                 'Name', 'r1')
    fullyConnectedLayer(128,   'Name', 'fc2')
    reluLayer(                 'Name', 'r2')]);
lgA = addLayers(lgA, [fullyConnectedLayer(actSize, 'Name', 'fc_m'); tanhLayer('Name', 'mean')]);
lgA = addLayers(lgA, [fullyConnectedLayer(actSize, 'Name', 'fc_s'); softplusLayer('Name', 'std')]);
lgA = connectLayers(lgA, 'r2', 'fc_m');
lgA = connectLayers(lgA, 'r2', 'fc_s');

actor = rlContinuousGaussianActor(dlnetwork(lgA), obsInfo, actInfo, ...
    'ActionMeanOutputNames',              'mean', ...
    'ActionStandardDeviationOutputNames', 'std');

%% Critic 1
obsP1 = [featureInputLayer(obsSize, 'Name', 'obs_in')
         fullyConnectedLayer(128,   'Name', 'obs_fc')
         reluLayer(                 'Name', 'obs_rl')];
actP1 = [featureInputLayer(actSize, 'Name', 'act_in')
         fullyConnectedLayer(128,   'Name', 'act_fc')
         reluLayer(                 'Name', 'act_rl')];
jnP1  = [concatenationLayer(1, 2,  'Name', 'cat')
         fullyConnectedLayer(128,  'Name', 'jn_fc')
         reluLayer(               'Name', 'jn_rl')
         fullyConnectedLayer(1,   'Name', 'qval')];
lg1 = addLayers(layerGraph(obsP1), actP1);
lg1 = addLayers(lg1, jnP1);
lg1 = connectLayers(lg1, 'obs_rl', 'cat/in1');
lg1 = connectLayers(lg1, 'act_rl', 'cat/in2');
critic1 = rlQValueFunction(dlnetwork(lg1), obsInfo, actInfo, ...
    'ObservationInputNames', 'obs_in', 'ActionInputNames', 'act_in');

%% Critic 2
obsP2 = [featureInputLayer(obsSize, 'Name', 'obs_in')
         fullyConnectedLayer(128,   'Name', 'obs_fc')
         reluLayer(                 'Name', 'obs_rl')];
actP2 = [featureInputLayer(actSize, 'Name', 'act_in')
         fullyConnectedLayer(128,   'Name', 'act_fc')
         reluLayer(                 'Name', 'act_rl')];
jnP2  = [concatenationLayer(1, 2,  'Name', 'cat')
         fullyConnectedLayer(128,  'Name', 'jn_fc')
         reluLayer(               'Name', 'jn_rl')
         fullyConnectedLayer(1,   'Name', 'qval')];
lg2 = addLayers(layerGraph(obsP2), actP2);
lg2 = addLayers(lg2, jnP2);
lg2 = connectLayers(lg2, 'obs_rl', 'cat/in1');
lg2 = connectLayers(lg2, 'act_rl', 'cat/in2');
critic2 = rlQValueFunction(dlnetwork(lg2), obsInfo, actInfo, ...
    'ObservationInputNames', 'obs_in', 'ActionInputNames', 'act_in');

fprintf('Critic Q-test: %.4f\n', ...
    getValue(critic1, {rand(obsSize,1)}, {rand(actSize,1)}))

%% SAC Agent
opt = rlSACAgentOptions( ...
    'SampleTime',             sim_ts, ...
    'ExperienceBufferLength', 1e5, ...
    'MiniBatchSize',          256, ...
    'DiscountFactor',         0.99, ...
    'TargetSmoothFactor',     5e-3);
opt.ActorOptimizerOptions.LearnRate     = 3e-4;
opt.CriticOptimizerOptions(1).LearnRate = 3e-4;
opt.CriticOptimizerOptions(2).LearnRate = 3e-4;

agent = rlSACAgent(actor, [critic1 critic2], opt);

%% Training Options
% Max reward per episode = 0 (perfect balance entire episode)
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                5000, ...
    'MaxStepsPerEpisode',         sim_time / sim_ts, ...
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          -3, ...
    'SaveAgentCriteria',          'EpisodeReward', ...
    'SaveAgentValue',             -80, ...
    'SaveAgentDirectory',         'saved_agents', ...
    'Verbose',                    true, ...
    'Plots',                      'training-progress');

fprintf('Training started...\n')
result = train(agent, env, trainOpts);
save('reactionwheel_agent.mat', 'agent', 'result')
fprintf('Agent saved.\n')