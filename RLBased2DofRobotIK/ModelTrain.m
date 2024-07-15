clc,clear,close all,warning off
% Written By: Rasit
% Date: 15-Jul-2024
% Reinforcement Learning Based 2DOF Robotic Arm Inverse Kinematic Test
%% Parameters
alpha = 0.2;    % Learning Rate
gamma = 0.9;    % Discount
epsilon = 0.5;  % Exploration Rate
episodes = 1e5;
max_steps = 100;

l1 = 1.0; % Robot Link lenght
l2 = 1.0;

% State and Action Space
theta1_space = linspace(0, pi, 50);
theta2_space = linspace(0, pi, 50);
actions = [-0.05, 0, 0.05];

% Q Table Init Val
Q = zeros(length(theta1_space), length(theta2_space), length(actions), length(actions));

% Target Pose and Tolerance
target_x = 0.5;
target_y = 1.0;
target_tolerance = 0.01;

for episode = 1:episodes
    theta1 = rand * pi;
    theta2 = rand * pi;

    for step = 1:max_steps
        state1_idx = get_state_index(theta1, theta1_space);
        state2_idx = get_state_index(theta2, theta2_space);

        if rand < epsilon
            action1_idx = randi(length(actions));
            action2_idx = randi(length(actions));
        else
            [~, max_idx] = max(Q(state1_idx, state2_idx, :, :), [], 'all', 'linear');
            [action1_idx, action2_idx] = ind2sub(size(squeeze(Q(state1_idx, state2_idx, :, :))), max_idx);
        end

        action1 = actions(action1_idx);
        action2 = actions(action2_idx);

        new_theta1 = theta1 + action1;
        new_theta2 = theta2 + action2;

        new_state1_idx = get_state_index(new_theta1, theta1_space);
        new_state2_idx = get_state_index(new_theta2, theta2_space);

        [x, y] = forward_kinematics(new_theta1, new_theta2, l1, l2);
        distance_to_target = sqrt((x - target_x)^2 + (y - target_y)^2);
        
        if distance_to_target < target_tolerance
            reward = 100;
        else
            reward = -distance_to_target;
        end

        Q(state1_idx, state2_idx, action1_idx, action2_idx) = ...
            Q(state1_idx, state2_idx, action1_idx, action2_idx) + ...
            alpha * (reward + gamma * max(Q(new_state1_idx, new_state2_idx, :, :), [], 'all') - ...
            Q(state1_idx, state2_idx, action1_idx, action2_idx));

        theta1 = new_theta1;
        theta2 = new_theta2;

        if distance_to_target < target_tolerance
            break;
        end
    end

end
return
% save RLTrainedModel Q max_steps theta1_space theta2_space actions target_x target_y target_tolerance
