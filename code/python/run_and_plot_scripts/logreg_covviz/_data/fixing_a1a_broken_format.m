% The format of the .mat data files for the A1A dataset do not match 
% the other files.
% After extraction of the original .zip file, 

load('exact_covariance_results\a1a\a1a_exact_M_1_L_0_restart_1.mat')

Sigma = Sigma{1}(:,:,end);
mu = mu{1}(:,:,end);

save('exact_covariance_results\a1a\a1a_fixed_exact_M_1_L_0_restart_1.mat')

load('mf_exact_covariance_results\a1a\a1a_mf-exact_M_1_L_0_restart_1.mat')

Sigma = Sigma{1}(:,:,end);
mu = mu{1}(:,:,end);

save('exact_covariance_results\a1a\a1a_fixed_mf-exact_M_1_L_0_restart_1.mat')

load('exact_covariance_results\a1a\a1a_exact_M_1_L_0_restart_1.mat')

Sigma = Sigma{1}(:,:,end);
mu = mu{1}(:,:,end);

save('exact_covariance_results\a1a\a1a_fixed_exact_M_1_L_0_restart_1.mat')

