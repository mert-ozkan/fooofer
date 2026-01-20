% Test Code

x = .1:.1:120;
true_parameters = struct( ...
    periodic = [1, 10, 3; ...
    2.6, 24, 2; ...
    .9, 48, 5; ...
    .7, 80, 2.5], ... % amp, center, sd, no baseline
    aperiodic = [10, 3, 72]);
gaussian_sigmas = [0.1, 0.5, 1.0];
gamma_shapes    = [100, 20, 5]; % High shape = low noise (CV = 1/sqrt(k))

% simulate data across varying levels of gaussian and gamma noise

% f_gauss = FOOOFer();
f_gamma = FOOOFer(log_scale = false, error_distribution='gamma');


%% fit using gaussian minimizer
paramsN = true_parameters.aperiodic;
paramsN(1) = log10(paramsN(1));
[y_gauss, y] = f_gauss.simulate(x, paramsN,[true_parameters.periodic(:)', 0], .1, 'gaussian');
plot(x,y);
hold on;
plot(x,y_gauss)
yscale('log')
%% fit using gamma minimizer
[y_gamma, y] = f_gamma.simulate(x, true_parameters.aperiodic,true_parameters.periodic(:)', 10, 'gamma');
figure;
plot(x,y);
hold on;
plot(x,y_gamma)
yscale('log')

%%
f_gamma = FOOOFer(log_scale = false, error_distribution='gamma');

[results, ap_fitter, p_fitter] = f_gamma.fit(x, y_gamma',plot=true);