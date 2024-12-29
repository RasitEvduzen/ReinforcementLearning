clc, clear, close all;
% Q-Learning Based Pendulum Control
% Written By: Rasit
% Date: 29-Dec-2024
%%
% Dynamic System Parameters
m = 0.5; M = 1.0; L = 1.0; g = 9.81; d = 0.1; 
dt = 0.01; 

% Q-Learning Parameters
num_theta = 100; % Theta discretization
num_omega = 150; % Angular velocity discretization
num_force = 10;  % Force levels [-15, -10, -5, 0, 5, 10, 15]
theta_space = linspace(-pi, pi, num_theta); 
omega_space = linspace(-8, 8, num_omega);  
forces = linspace(-10, 10, num_force); 

Q = zeros(num_theta, num_omega, num_force); % Q-table
alpha = 0.9;    % Learning rate
gamma = 0.8;    % Discount factor
epsilon = 0.99; % Exploration rate
epsilon_decay = .99; % Exploration rate decay
min_epsilon = 0.01;  % Minimum exploration rate

% Training Parameters
episodes = 5e3;
max_steps = 5e3;       
epsilon_theta = 0.001; 
rewards_per_episode = zeros(episodes, 1);

for episode = 1:episodes
    
    theta = pi + rand() * epsilon_theta - epsilon_theta / 2;
    omega = epsilon_theta * (rand() - 1);  
    x = [0, 0, theta, omega]; 
    total_reward = 0; 

    for step = 1:max_steps
        [theta_idx, omega_idx] = discretize_state([x(3), x(4)], theta_space, omega_space);
        % Action selection (epsilon-greedy)
        if rand() < epsilon
            action_idx = randi(num_force); % Exploration
        else
            [~, action_idx] = max(Q(theta_idx, omega_idx, :)); % Exploitation
        end
        action = forces(action_idx); % Select force

        % Propagate dynamics
        dx = PendStateSpace(x, m, M, L, g, d, action);
        x = x + dx' * dt; 
        x(3) = wrapToPi(x(3));

        % Reward function
        reward = -abs(wrapToPi(x(3) - pi))^2 ... % Angle reward
            - 0.4 * abs(x(4)) ...                % Angular velocity reward
            - 0.3 * abs(x(1))^2 ...              % Position reward
            - 0.1 * abs(x(2))^2;                 % Velocity reward

        % Extra reward for balancing
        if abs(wrapToPi(x(3) - pi)) < 0.02 && abs(x(4)) < 0.01 && abs(x(1)) < 0.01 && abs(x(2)) < 0.01
            reward = reward + 20; 
        end
        total_reward = total_reward + reward;
        [new_theta_idx, new_omega_idx] = discretize_state([x(3), x(4)], theta_space, omega_space);

        % Update Q-table
        best_future_q = max(Q(new_theta_idx, new_omega_idx, :));
        Q(theta_idx, omega_idx, action_idx) = ...
            Q(theta_idx, omega_idx, action_idx) + alpha * (reward + gamma * best_future_q - Q(theta_idx, omega_idx, action_idx));

        % Stop Condition
        if abs(wrapToPi(x(3) - pi)) < 0.1 && abs(x(4)) < 0.1 && abs(x(1)) < 0.1 && abs(x(2)) < 0.1
            break;
        end
    end

    rewards_per_episode(episode) = total_reward;
    epsilon = max(min_epsilon, epsilon * epsilon_decay);
    if mod(episode, 500) == 0
        disp(['Episode: ', num2str(episode), ', Epsilon: ', num2str(epsilon), ', Total Reward: ', num2str(total_reward)]);
    end
end

figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
plot(rewards_per_episode, 'LineWidth', 2);
xlabel('Episode');
ylabel('Total Reward');
title('Learning Progress (Total Reward per Episode)');
grid on;


%% Test Phase
test_steps = 3e2; 
theta = pi + deg2rad(.5); 
omega = 0; 
x = [0, 0, theta, omega];
state_history = zeros(test_steps, 4); 
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
for step = 1:test_steps
    state_history(step, :) = x;
    [theta_idx, omega_idx] = discretize_state([x(3), x(4)], theta_space, omega_space);
    
    [~, action_idx] = max(Q(theta_idx, omega_idx, :));
    action = forces(action_idx);

    % Propagate Dynamics
    dx = PendStateSpace(x, m, M, L, g, d, action);
    x = x + dx' * dt;
    x(3) = wrapToPi(x(3)); 
    theta_unwrapped = unwrap(state_history(1:step, 3)); 

    if mod(step, 1) == 0 || mod(step, test_steps) == 0
        clf;
        
        subplot(2, 2, [1, 2]);
        hold on;
        yline(0 - 0.5, 'k--', 'LineWidth', 2);
        W = 1.5; H = 0.5; % Cart dimensions
        wr = 0.5; % Wheel radius
        mr = 0.5; % Pendulum mass radius
        cart_pos = x(1); % Cart position
        pendx = cart_pos + L * sin(x(3)); % Pendulum tip x position
        pendy = -L * cos(x(3)); % Pendulum tip y position

        rectangle('Position', [cart_pos - W / 2, -H / 2, W, H], 'Curvature', 0.1, ...
            'FaceColor', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);
        rectangle('Position', [cart_pos - W / 2, -H, wr, wr], 'Curvature', 1, ...
            'FaceColor', [1 1 0], 'LineWidth', 1.5);
        rectangle('Position', [cart_pos + W / 2 - wr, -H, wr, wr], 'Curvature', 1, ...
            'FaceColor', [1 1 0], 'LineWidth', 1.5);
        plot([cart_pos pendx], [0 pendy], 'k', 'LineWidth', 2); % Pendulum line
        rectangle('Position', [pendx - mr / 2, pendy - mr / 2, mr, mr], ...
            'Curvature', 1, 'FaceColor', [1 0.1 0.1], 'LineWidth', 1.5);
        axis equal; axis([-12 12 -3 3]); grid on;
        title('Inverted Pendulum RL-Based Control');

        subplot(2, 1, 2);
        plot(theta_unwrapped * (180 / pi), state_history(1:step, 4), 'b', 'LineWidth', 1.5);
        xlabel('\theta [rad]');
        ylabel('\omega [rad/s]');
        grid on;
        title('Phase Space (\theta ~ \omega)');
        drawnow; 
       
    end
end

%% Helper functions
function [theta_idx, omega_idx] = discretize_state(state, theta_space, omega_space)
[~, theta_idx] = min(abs(theta_space - state(1)));
[~, omega_idx] = min(abs(omega_space - state(2)));
end

function dx = PendStateSpace(x, m, M, L, g, d, u)
D = m * L^2 * (M + m * (1 - cos(x(3))^2));
dx(1, 1) = x(2); 
dx(2, 1) = (1 / D) * (-m^2 * L^2 * g * cos(x(3)) * sin(x(3)) + m * L^2 * (m * L * x(4)^2 * sin(x(3)) - d * x(2))) + m * L^2 * (1 / D) * u;
dx(3, 1) = x(4); 
dx(4, 1) = (1 / D) * ((M + m) * m * g * L * sin(x(3)) - m * L * cos(x(3)) * (m * L * x(4)^2 * sin(x(3)) - d * x(2))) - m * L * cos(x(3)) * (1 / D) * u;
end
