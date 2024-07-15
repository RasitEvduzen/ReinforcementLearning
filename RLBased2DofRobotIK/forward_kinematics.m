function [x, y] = forward_kinematics(theta1, theta2, l1, l2)
x = l1 * cos(theta1) + l2 * cos(theta1 + theta2);
y = l1 * sin(theta1) + l2 * sin(theta1 + theta2);
end