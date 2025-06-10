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
W      = 1.3;  
%% -----------------Paper -> Unitless! --------------------------------- %%
dx     = 0.8;  dy = dx;     % dx/W = 0.4              % tune if stable
R_pillar = phys.Rpillar_um * conv; % 67.5
R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
gap_size = gap_size_um*conv;
lambda=2;
v=force;
Df = .002*conv; % diffusion coefficient in PF units
dt     = 1e-2; % time step in PF units
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
x_center = x(centerx);
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
grad_field=gradient_field(.2, .06, gap_size, x, X);
% figure(1);
% imagesc(grad_field, [0 0.3]);
% axis equal tight;
% colormap(theme_of_plots); colorbar;

% figure(2);
% imagesc(my_field, [0 0.3]);
% axis equal tight;
% colormap(theme_of_plots); colorbar;

%title('my field');

%% ------------------ PDE functions ------------------ %%
g = @(phi) ph`i.^3.*(10 + 3*phi.*(2*phi-5));
g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
w = @(phi) (16*phi.^2.*(1-phi).^2); 

[~, dphidy] = my_gradient(phi, dx, dy);
mask_front =  (dphidy <  0);
front_of_cell = w(phi).*double(mask_front > 0);
% un_binding_rate =front_of_cell .* my_field;
unbinding_rates = 0.2 * w(phi) .* double(~mask_front)+0.2 * front_of_cell .* double(my_field == 0)+ my_field .* front_of_cell;
binding_rate =.2*w(phi);
k_off = .2;
phi_mask = phi > 0.5;
y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
ubr=unbinding_rate(my_field, k_off, phi, dphidy, psi, w(phi), y_coms, y_center);
% figure(911)
% imagesc(unbinding_rates);
% figure(912)
% imagesc(ubr);
% drawnow;
count = 0;
cf_w= .5.*w(phi);  % initial free concentration
cb_w= .5.*w(phi);  % initial bound concentration
cf = cf_w./(w(phi)+1e-6); % avoid division by zero
cb = cb_w./(w(phi)+1e-6); % avoid division by zero
total_c_b = zeros(1, round(nSteps));
total_c_f = zeros(1, round(nSteps));
t = zeros(1, round(nSteps));
figure(1); clf;

        subplot(2,2,1);
        imagesc(cf, [0, max(cf(:))]); colorbar;
        title(['c_b at t = ', num2str(0*dt, '%.2f'), ' s']);
        subplot(2,2,2);
        imagesc(cf, [0, max(cf(:))]); colorbar;
        title(['c_f at t = ', num2str(0*dt, '%.2f'), ' s']);

        subplot(2,2,3);
        imagesc(w(phi) , [0 max(w(phi(:)))]); colorbar;
        title('w(\phi)');

        subplot(2,2,4);
        imagesc(binding_rate , [0 max(binding_rate(:))]); colorbar;
        title('binding rate');


        % % Save to GIF
        % frame = getframe(cf);
        % im = frame2im(frame);
        % [imind, cm] = rgb2ind(im, 256);
        % imwrite(imind, cm, 'cf_cb_movie.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.1);


%% ------------------ PDE functions ------------------ %%
total_c_record = zeros(1, round(nSteps));
for step = 1:nSteps
    % Inside time loop (at each step)
    wphi = w(phi);
    g_prime_phi=g_prime(phi);
    % ---------- forces -------------
    lap_phi   = 4*del2(phi,dx,dy)/(dx*dy);
    lap_psi   = 4*del2(psi,dx,dy)/(dx*dy);
    unbinding_rate_on_edge = unbinding_rate(grad_field, k_off, phi, dphidy, psi, wphi, y_coms, y_center);
    binding_rate_on_edge = k_off .* wphi; % binding rate on edge
    [cf, cb] = update_RD(Df,phi, cf, cb, binding_rate_on_edge, unbinding_rate_on_edge, wphi, dx, dy, dt);
    % ---------- precompute spatial stuff ----------
    total_cf = sum(wphi(:) .* cf(:)) * dx * dy;
    total_cb = sum(wphi(:) .* cb(:)) * dx * dy;
    total_c  = total_cf + total_cb;
    total_c_record(step) = total_c; % record total concentration

    if mod(step, save_rate) == 0 
        cfw=cf.*wphi; % weighted free concentration
        cbw=cb.*wphi; % weighted bound concentration
        figure(33);
        idx = step / save_rate;
        both = psi + phi; % combined field
        % weight localised to the interface

        subplot(2,2,1);
        imagesc(cbw, [0, max(cfw(:))]); colorbar;
        title(['c_b at t = ', num2str(step*dt, '%.2f'), ' s']);

        subplot(2,2,2);
        imagesc(cfw, [0, max(cfw(:))]); colorbar;
        title(['c_f at t = ', num2str(step*dt, '%.2f'), ' s']);

        subplot(2,2,3);
        imagesc(unbinding_rate_on_edge , [0 max(unbinding_rate_on_edge(:))]); colorbar;
        title('unbinding rate');

        subplot(2,2,4);
        imagesc(binding_rate , [0 max(binding_rate(:))]); colorbar;
        title('binding rate');
        drawnow;

        % % Save to GIF
        % frame = getframe(gcf);
        % im = frame2im(frame);
        % [imind, cm] = rgb2ind(im, 256);
        % imwrite(imind, cm, 'cf_cb_movie.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);

    end
end
plot((1:Nt)*dt, total_c_record / total_c_record(1)); % Normalize to initial value
xlabel('Time'); ylabel('Total c / Initial c');
title('Check for conservation');

function gradient_field = gradient_field(k_fast, k_slow, gap_size, x, X)
    x_center = max(x) / 2;
    dx = x(2) - x(1);  % Assuming uniform spacing!
    gap_size = round(gap_size/dx)*dx;

    x_left_end = x_center - gap_size/2;
    x_right_end = x_center + gap_size/2;

    my_field = zeros(size(X));
    % Left half of gap: k_fast → k_slow
    left_mask = (X > x_left_end) & (X <= x_center);
    my_field(left_mask) = k_fast - (X(left_mask) - x_left_end) / (gap_size/2) * (k_fast - k_slow);
    % Right half of gap: k_slow → k_fast
    right_mask = (X > x_center) & (X <= x_right_end);
    my_field(right_mask) = k_slow + (X(right_mask) - x_center) / (gap_size/2) * (k_fast - k_slow);
    gradient_field = my_field; % Return the gradient field
end

function [cb, cf] = update_RD(Df,phi, cf, cb, binding_rate_on_edge, unbinding_rate_on_edge, wphi, dx, dy, dt)
    lap_cf         = my_laplacian(cf, dx, dy);
    div_term = weighted_diffusion(cf, wphi, dx, dy);


    % ---------- reactive terms ----------
    react_cf = -binding_rate_on_edge .* cf + unbinding_rate_on_edge .* cb;
    react_cb =  binding_rate_on_edge .* cf - unbinding_rate_on_edge .* cb;

    % ---------- evolve the *weighted* variables ----------
    wc  = wphi .* cf;
    wb  = wphi .* cb;

    wc  = wc + dt * (Df * div_term  + react_cf);  % note: multiply by wphi
    wb  = wb + dt * (               react_cb);

    % ---------- recover plain concentrations (for plotting / reaction only) ----------
    cf  = wc ./ max(wphi,1e-6);   % safer ε
    cb  = wb ./ max(wphi,1e-6);

    % ---------- clean-up outside cell ----------
    cf(phi < 1e-3) = 0;
    cb(phi < 1e-3) = 0;

end
function div_diff_cf = weighted_diffusion(cf, wphi, dx, dy)
    % Get gradients of cf
    [dcfdx, dcfdy] = gradient(cf, dx, dy);
    
    % Multiply gradients by the weight
    jx = wphi .* dcfdx;
    jy = wphi .* dcfdy;

    % Divergence of the flux
    [djxdx, ~] = gradient(jx, dx, dy);
    [~, djydy] = gradient(jy, dx, dy);

    % Total diffusion term
    div_diff_cf = djxdx + djydy;
end

function unbinding_rate_on_edge = unbinding_rate(gradient_field, k_off, phi, dphidy, psi, w_phi, y_coms, y_center)
    % Calculate the unbinding rate based on the reaction kinetics
    % k_on: binding rate constant (int)
    % k_off: unbinding rate constant (int)
    % phi: phase field  (matrix for position and time)
    % dphidy: gradient of phase field (matrix for position)
    % psi: Pillars (matrix for position and time)
    % y_coms: center of mass for the cell (scalar)
    % dx, dy: spatial discretization steps
    mask_front =  (dphidy <  0);  % mask for the front of the cell
    % unbinding_rate_on_edge = k_off * w_phi;  % default unbinding rate
    % imagesc(gather(front_mask), [0 1]); axis image; axis off; hold on;
    % Ensure inputs are valid
    if nargin < 8
        error('Not enough input arguments.');
    end
    if y_coms <= y_center || any(psi(:) .* phi(:) ~= 0)
        front_of_cell = w_phi.*double(mask_front > 0);
        % figure(777);
        % imagesc(front_of_cell)
        drawnow;
        %0.2 * w(phi) .* double(~mask_front)+0.2 * front_of_cell .* double(my_field == 0)+ my_field .* front_of_cell;
        unbinding_rate_on_edge = k_off * w_phi .* double(~mask_front) + k_off* front_of_cell .* double(gradient_field == 0)+ gradient_field .* front_of_cell;
    end
    % Calculate the unbinding rate
    end

function lap = my_laplacian(phi, dx, dy)
%MY_LAPLACIAN Fast 2D Laplacian for uniform grid spacing
%   lap = my_laplacian(phi, dx, dy)
    if nargin < 2
        dx = 1;
    end
    if nargin < 3
        dy = 1;
    end
    lap= del2(phi, dx, dy) * 4 / (dx * dy);
end

function [dphidx, dphidy] = my_gradient(phi, dx, dy)
%MY_GRADIENT Fast 2D gradient for uniform grid spacing
%   [dphidx, dphidy] = my_gradient(phi, dx, dy)
%
%   Inputs:
%     phi - 2D array
%     dx  - spacing in x-direction (columns)
%     dy  - spacing in y-direction (rows)
%
%   Outputs:
%     dphidx - partial derivative ∂φ/∂x
%     dphidy - partial derivative ∂φ/∂y

if nargin < 2
    dx = 1;
end
if nargin < 3
    dy = 1;
end

[Ny, Nx] = size(phi);

dphidx = zeros(Ny, Nx);
dphidy = zeros(Ny, Nx);

% ∂φ/∂x: finite difference across columns
dphidx(:,2:Nx-1) = (phi(:,3:Nx) - phi(:,1:Nx-2)) / (2*dx);
dphidx(:,1)      = (phi(:,2) - phi(:,1)) / dx;
dphidx(:,end)    = (phi(:,end) - phi(:,end-1)) / dx;

% ∂φ/∂y: finite difference across rows
dphidy(2:Ny-1,:) = (phi(3:Ny,:) - phi(1:Ny-2,:)) / (2*dy);
dphidy(1,:)      = (phi(2,:) - phi(1,:)) / dy;
dphidy(end,:)    = (phi(end,:) - phi(end-1,:)) / dy;
end