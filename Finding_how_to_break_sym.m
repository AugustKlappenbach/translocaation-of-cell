% reaction_diffusion_sim.m
% Full simulation of φ-based reaction-diffusion model for cf and cb

clear; clc; close all;

%% --- Parameters ---
Nx = 200;
Ny = 100;
Lx = 20;  % domain size x
Ly = 10;  % domain size y
dx = Lx / Nx;
dy = Ly / Ny;
x = linspace(-Lx/2, Lx/2, Nx);
y = linspace(-Ly/2, Ly/2, Ny);
[X, Y] = meshgrid(x, y);

dt = 1e-3;
T_final = 2;
nSteps = round(T_final / dt);

kon = (2/(T_final));
Df = .002;

%% --- Phase Field φ ---
R = 3;
r = sqrt(X.^2 + Y.^2);
phi = 0.5 * (1 - tanh((r - R) / .2));  % tanh transition around radius R

%% --- Initial Conditions for cf and cb ---
wphi = phi .* (1 - phi);
cf = 1 *wphi;  % small random initial free concentration
cb = zeros(Ny, Nx);       % bound initially zero
save_rate = 50;  % save every 50 steps
Nsamp = floor(nSteps/save_rate);
cf_tot = zeros(1, Nsamp, 'double');
cb_tot = zeros(1, Nsamp, 'double');
t_axis = (0:Nsamp) * save_rate * dt;   % physical time for x-axis
sums_cf = zeros(nSteps, 1);
sums_cb = zeros(nSteps, 1);
sample_idx = 0;
%% --- Visualization Setup ---
figure; 
colormap(spring);
tiledlayout(1,3, 'Padding', 'compact', 'TileSpacing', 'compact');

for step = 1:nSteps
    % --- Smooth boundary mask w(phi) ---
    wphi = 18 .* phi.^2 .* (1 - phi).^2;
   

    % --- Compute dφ/dy and koff ---
    [~, dphidy] = gradient(phi, dx, dy);
    abs_dphidy = abs(dphidy);
    epsilon = 1e-6;
    denom = max(abs_dphidy - dphidy, epsilon)+ epsilon;
    koff = .05*exp(2 ./ denom);
    koff = min(koff, 1e3);  % or some safe max like 1e3


    % --- Gradient and diffusion ---
    [cf_x, cf_y] = gradient(cf, dx, dy);
    div_diff_cf = divergence(wphi .* cf_x, wphi .* cf_y);

    % --- Reaction terms ---
    react_cf = wphi .* (-kon .* cf + koff .* cb);
    react_cb = wphi .* ( kon .* cf - koff .* cb);

    % --- Time update ---
    cf = cf + dt * (Df * div_diff_cf + react_cf);
    cb = cb + dt * react_cb;
    % --- Plot every 50 steps ---
    if mod(step, save_rate) == 0 || step == 1
        sample_idx = sample_idx + 1;
        idx = step / save_rate;
        % weight localised to the interface
        w = 6 .* phi.^2 .* (1-phi).^2;          % same as in the solver
        % integrate along surface :  ∑ w * c * dx * dy
        cb_tot(sample_idx) = sum( w(:) .* cb(:) ) * dx * dy;
        cf_tot(sample_idx) = sum( w(:) .* cf(:) ) * dx * dy;
        nexttile(1); imagesc(x, y, phi); axis equal tight; title('\phi');
        nexttile(2); imagesc(x, y, cf); colorbar; axis equal tight; title('c_f');
        nexttile(3); imagesc(x, y, cb); colorbar; axis equal tight; title('c_b');
        sgtitle(sprintf('Step %d / %d /n cf = %d, cd = %d', step, nSteps));
        drawnow;
    end
    
end
disp(['length(t_axis): ', num2str(length(t_axis))]);
disp(['length(cb_tot): ', num2str(length(cb_tot))]);


figure;
plot(t_axis, cb_tot , 'LineWidth',1.6); hold on;
plot(t_axis, cf_tot, '--', 'LineWidth',1.6);
xlabel('time (s)'); ylabel('surface-integrated density');
legend({'c_b  (bound)', 'c_f  (free)'}, 'Location','best');
title('Receptor populations vs. time');
grid on;
