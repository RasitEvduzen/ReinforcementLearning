clc,clear,close all,warning off
% Written By: Rasit
% Date: 15-Jul-2024
% Reinforcement Learning Based 2DOF Robotic Arm Inverse Kinematic Test
%% Test Robot
load RLTrainedModel.mat
% Robot DH Param
l1 = 1.0;
l2 = 1.0;

a     = [0,l1,l2];
alp   = [0,0,0];
d     = [0,0,0];

theta1 = rand*pi;
theta2 = rand*pi;

trajectory_x = [];
trajectory_y = [];

figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
for step = 1:max_steps
    state1_idx = get_state_index(theta1, theta1_space);
    state2_idx = get_state_index(theta2, theta2_space);

    [~, max_idx] = max(Q(state1_idx, state2_idx, :, :), [], 'all', 'linear');
    [action1_idx, action2_idx] = ind2sub(size(squeeze(Q(state1_idx, state2_idx, :, :))), max_idx);

    action1 = actions(action1_idx);
    action2 = actions(action2_idx);

    theta1 = theta1 + action1;
    theta2 = theta2 + action2;

    [x, y] = forward_kinematics(theta1,theta2,l1,l2);
    trajectory_x = [trajectory_x, x];
    trajectory_y = [trajectory_y, y];

    clf
    Tee = eye(4,4);       
    trplot(Tee,'frame',0,'thick',.1,'rgb','length',.25),hold on,axis([-2 2 -2 2])
    theta = [theta1, theta2,0];
    for i=1:size(theta,2)
        temp = Tee;
        T(:,:,i) = DHMatrixModify(alp(i),a(i),d(i),theta(i));
        Tee = Tee * T(:,:,i);
        plot([temp(1,4) Tee(1,4)],[temp(2,4) Tee(2,4)],'k','LineWidth',1.5)
        trplot(Tee,'frame',num2str(i),'thick',.1,'rgb','length',.25)
        xlabel('X-axis'),ylabel('Y-axis')
    end
    plot(trajectory_x, trajectory_y, 'b-',LineWidth=2);
    scatter(target_x, target_y, 'ro',"filled");
    title(["2Dof Robotic Arm Inverse Kinematic Solution Via Reinforcement Learning"; ...
        "||error||: "+num2str(sqrt((x - target_x)^2 + (y - target_y)^2))]);
    xlabel('X'),ylabel('Y'),grid on,xline(0),yline(0);
    drawnow

    if sqrt((x - target_x)^2 + (y - target_y)^2) < target_tolerance
        break;
    end
end
