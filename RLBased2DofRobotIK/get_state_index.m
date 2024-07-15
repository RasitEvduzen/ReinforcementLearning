function state_index = get_state_index(theta, theta_space)
[~, state_index] = min(abs(theta_space - theta));
end