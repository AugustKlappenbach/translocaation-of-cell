close all; clear; clc;
theme_of_plots = 'parula'; % 'hsv', 'parula', 'jet', 'gray'
force=1;
gap_size_um = 15; % [µm] gap size between pillars
    %saving
    gifFile = fullfile(getenv('HOME'), 'gifs_for_RD', ...
        sprintf('RD.gif'));

%% ----------------  Paper PHYSICAL INPUT  ---------------- %%
phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
phys.Rpillar_um = 13.5;       % [µm]
phys.Rcell_um   = 10;         % [µm]
conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
W      = 1;  
%% -----------------Paper -> Unitless! --------------------------------- %%
dx     = 0.4;  dy = dx;     % dx/W = 0.4              % tune if stable
R_pillar = phys.Rpillar_um * conv; % 67.5
R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
gap_size = gap_size_um*conv;
lambda=2;
v=force;
Df = .002*conv; % diffusion coefficient in PF units
dt     = 1e-2;
nSteps = 20/dt;
save_interval = 1;
kon = .2;
save_rate = 10; % save every 50 steps
%% --- domain size --- %%
Lx = 3*(R_cell);   
Ly = 3*R_cell;  

Nx = ceil(Lx/dx);
Ny = ceil(Ly/dy);
Nt=nSteps
x  = (0:Nx-1)*dx;
y  = (0:Ny-1)*dy;
[X,Y] = meshgrid(x,y);
%% --- Cell --- %%
cy_cell = round(.5*Ny);
cx_cell = round(.5*Nx);
cx_cell_center = x(cx_cell);
cy_cell_center = y(cy_cell);
phi= phase_field(X, Y, cx_cell_center, cy_cell_center, R_cell, W);
%% --- Pillars --- %%
centerx = round(Nx/2);
centery = round(Ny/2);
x_center=x(centerx);
y_center = y(centery);
gap_distance = 2 * R_pillar + gap_size;
center_offset = gap_distance / 2;
x_left  = x(centerx) - center_offset;
x_right = x(centerx) + center_offset;
psi_left = phase_field(X, Y, x_left, y_center, R_pillar, W);
psi_right = phase_field(X, Y, x_right, y_center, R_pillar, W);
psi = psi_left + psi_right;
imagesc(psi, [0 1]); axis equal tight;
colormap(theme_of_plots); colorbar;
title('\phi (Cell Shape)');
%% ------------------- fields for reaction diffussion ------------------ %%
x_center = max(x) / 2;  % or x(end)/2
gap_size = round(gap_size/dx)*dx;

x_left_end = x_center - gap_size/2;
x_right_end = x_center + gap_size/2;

my_field = zeros(size(X));
% Left half of gap: 0.2 → 0.067
left_mask = (X > x_left_end) & (X <= x_center);
my_field(left_mask) = 0.2 - (X(left_mask) - x_left_end) / (gap_size/2) * (0.2 - 0.067);
% Right half of gap: 0.067 → 0.2
right_mask = (X > x_center) & (X <= x_right_end);
my_field(right_mask) = 0.067 + (X(right_mask) - x_center) / (gap_size/2) * (0.2 - 0.067);
% imagesc(my_field, [0 .3]); axis equal tight;
% colormap(theme_of_plots); colorbar;
%title('my field');

%% ------------------ PDE functions ------------------ %%
g = @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
w = @(phi) 4*phi.*(1-phi); 

[~, dphidy] = my_gradient(phi, dx, dy);
mask_front =  (dphidy <  0);
front_of_cell = w(phi).*double(mask_front > 0);
% un_binding_rate =front_of_cell .* my_field;
unbinding_rate = 0.2 * w(phi) .* double(~mask_front)+0.2 * front_of_cell .* double(my_field == 0)+ my_field .* front_of_cell;
binding_rate =.2*w(phi);
count = 0;
cf= .5.*w(phi);  % initial free concentration
cb = .5.*w(phi);  % initial bound concentration
init_cf=sum(cf(:))*dx*dy;
init_cb=sum(cb(:))*dx*dy;
total_c_b = zeros(1, round(nSteps));
total_c_f = zeros(1, round(nSteps));
t = zeros(1, round(nSteps));
w_variance = zeros(1, round(nSteps));
figure(1); clf;

        subplot(2,2,1);
        imagesc(cb); colorbar;
        title(['c_b at t = ', num2str(0*dt, '%.2f'), ' s']);

        subplot(2,2,2);
        imagesc(cf); colorbar;
        title(['c_f at t = ', num2str(0*dt, '%.2f'), ' s']);

        subplot(2,2,3);
        imagesc(unbinding_rate , [0 max(unbinding_rate(:))]); colorbar;
        title('unbinding rate');

        subplot(2,2,4);
        imagesc(binding_rate , [0 max(binding_rate(:))]); colorbar;
        title('binding rate');

        
        % Save to GIF
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        imwrite(imind, cm, 'cf_cb_movie.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
       

%% ------------------ PDE functions ------------------ %%
for step = 1:nSteps
    % Inside time loop (at each step)
    wphi = w(phi);
    g_prime_phi=g_prime(phi);
    % ---------- forces -------------
    lap_phi   = 4*del2(phi,dx,dy)/(dx*dy);
    lap_psi   = 4*del2(psi,dx,dy)/(dx*dy);
    
    % --- Gradient and diffusion ---
    [cf_x, cf_y] = my_gradient(cf, dx, dy);
    div_diff_cf = wphi.*my_laplacian(cf,dx,dy) + my_gradient(wphi,dx,dy).*my_gradient(cf,dx,dy);
    front_of_cell = wphi.*double(mask_front > 0);
    % --- inside the time-stepping loop, after you recompute wphi ----------
    binding_rate  = 0.2 * wphi;                           % ← update if φ changes
    unbinding_rate = 0.2*wphi .* double(~mask_front) ...
                + 0.2*front_of_cell.*double(my_field==0) ...
                + my_field .* front_of_cell;            % same formula, new wphi

    react_cf = (-binding_rate .* cf + unbinding_rate .* cb);
    react_cb = ( binding_rate .* cf - unbinding_rate .* cb);

    cf = cf + dt * (Df * div_diff_cf + react_cf);
    cb = cb + dt *  react_cb;
    cf_no_w= cf./(wphi+1e-6); % avoid division by zero
    cb_no_w= cb./(wphi+1e-6); % avoid division by zero
    total_c_f(step)= sum(cf_no_w(:))*dx*dy;
    total_c_b(step)= sum(cb_no_w(:))*dx*dy;
    t(step) = step * dt;
    w_variance(step) = sum(wphi(:)) * dx * dy; % variance of w(phi)
    if mod(step, save_rate) == 0 
        
        idx = step / save_rate;
        both = psi + phi; % combined field
        % weight localised to the interface

        subplot(2,2,1);
        imagesc(cb); colorbar;
        title(['c_b at t = ', num2str(step*dt, '%.2f'), ' s']);

        subplot(2,2,2);
        imagesc(cf); colorbar;
        title(['c_f at t = ', num2str(step*dt, '%.2f'), ' s']);

        subplot(2,2,3);
        imagesc(unbinding_rate , [0 max(unbinding_rate(:))]); colorbar;
        title('unbinding rate');

        subplot(2,2,4);
        imagesc(binding_rate , [0 max(binding_rate(:))]); colorbar;
        title('binding rate');

        
        % Save to GIF
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        imwrite(imind, cm, 'cf_cb_movie.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);

    end
end
clc;
figure(2); clf;
total_cs= total_c_f(1) + total_c_b(1);
plot(t, total_c_b/total_cs, 'b', 'DisplayName', 'Total c_b');
hold on;
plot(t, total_c_f/total_cs, 'r', 'DisplayName', 'Total c_f');
hold on;
plot(t, (total_c_f + total_c_b)/(total_cs), 'g', 'DisplayName', 'Total c');
xlabel('Time [s]');
ylabel('Total Concentration');
legend;
title('Total c_b and c_f over time');
saveas(gcf, 'total_cb_cf_plot.png');
clc;
figure(3); clf;
plot(t, w_variance/w_variance(1), 'k', 'DisplayName', 'w variance');
xlabel('Time [s]');
ylabel('Variance of w(phi)');
title('Variance of w(phi) over time');
saveas(gcf, 'w_variance_plot.png');
clc;
