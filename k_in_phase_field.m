close all; clear; clc;
theme_of_plots = 'parula'; % 'hsv', 'parula', 'jet', 'gray'
force=1;
gap_size_um = 15; % [µm] gap size between pillars
    %saving
    gifFile = fullfile(getenv('HOME'), 'video', ...
    sprintf('R_D.gif'));

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
Df = .2*conv; % diffusion coefficient in PF units
dt     = 1e-3/3; % time step in PF units
nSteps = 10/dt;
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
g = @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);

[dphidx, dphidy] = my_gradient(phi, dx, dy);
mask_front =  (dphidy <  0);
% un_binding_rate =front_of_cell .* my_field;
k_off = .2;
phi_mask = phi > 0.5;
y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
ubr=unbinding_rate(my_field, k_off, phi, dphidy, psi, y_coms, y_center);
boundary_of_cell= dphidx.^2 + dphidy.^2;
epsilon = 1./max(boundary_of_cell(:)); 
w = epsilon*( dphidx.^2 + dphidy.^2);
binding_rate = .2;
cf_boundary= .5.*boundary_of_cell;  % initial free concentration
cb_boundary= .5.*boundary_of_cell;  % initial bound concentration
c_f = cf_boundary./(boundary_of_cell+1e-3); % avoid division by zero
c_b = cb_boundary./(boundary_of_cell+1e-3); % avoid division by zero
% Normal vector components
nx = -dphidx ./ sqrt(dphidx.^2 + dphidy.^2 + 1e-6);
ny = -dphidy ./ sqrt(dphidx.^2 + dphidy.^2 + 1e-6);

total_cbs = zeros(1, round(nSteps));
total_cfs = zeros(1, round(nSteps));
total_cs = zeros(1,round(nSteps));

t = zeros(1, round(nSteps));
fig = figure('Visible', 'on');
        subplot(2,3,1);
        imagesc(c_f, [0, max(c_f(:))]); colorbar;
        title(['c_b at t = ', num2str(0*dt, '%.2f'), ' s']);
        subplot(2,3,2);

        imagesc(c_b, [0, max(c_b(:))]); colorbar;
        title(['c_f at t = ', num2str(0*dt, '%.2f'), ' s']);

        subplot(2,3,3);
        imagesc(ubr , [0 max(ubr(:))]); colorbar;
        title('unbinding rate');

        subplot(2,3,4);
        imagesc(ubr.*boundary_of_cell , [0 .2]); colorbar;
        title('binding rate');

        subplot(2,3,5);
        imagesc(.2.*boundary_of_cell , [0 .2]); colorbar;
        title('boundary of cell'); 

        subplot(2,3,6);
        imagesc(boundary_of_cell , [0 max(boundary_of_cell(:))]); colorbar;
        title('|\nabla \phi |^2');

        % Save to GIF
        frame = getframe(fig);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        imwrite(imind, cm, 'cf_cb_movie.gif', 'gif', 'Loopcount', inf, 'DelayTime', 0.1);

lap = @(u) my_laplacian(u, dx, dy);

for step = 1:Nt
    % Compute diffusion of free species
    g_phi = epsilon*(dphidx.^2 + dphidy.^2);
    diff_cf = Df * weighted_divergence(g_phi, c_f, dx, dy);
    % Reaction terms
    bind = binding_rate * g_phi .* c_f;
    ubr=unbinding_rate(my_field, k_off, phi, dphidy, psi, y_coms, y_center);
    unbind = ubr .* g_phi .* c_b;
    
    % Time update
    gcf = g_phi .* c_f;
    gcb = g_phi .* c_b;
    
    gcf_new = gcf + dt * (diff_cf - bind + unbind);
    gcb_new = gcb + dt * (bind - unbind);

    % Avoid divide-by-zero: enforce positivity of g_phi
    g_phi_safe = g_phi + 1e-6;

    % Solve for updated concentrations
    c_f = gcf_new ./ g_phi_safe;
    c_b = gcb_new ./ g_phi_safe;

    % Track totals if needed
    total_cfs(step)  = sum(c_f(:) .* g_phi(:))*dx*dx;
    total_cbs(step) = sum(c_b(:) .* g_phi(:))*dx*dx;
    total_cs(step) = total_cbs(step) + total_cfs(step);
    t(step)         = step * dt;

    % Visualization (every few steps)
    if mod(step, save_rate) == 0
        subplot(2,2,1); imagesc(c_f.*g_phi, [0, max(c_f(:))]); title(sprintf('c_f, t=%.2f', step*dt)); axis equal tight; colorbar;
        subplot(2,2,2); imagesc(c_b.*g_phi, [0, max(c_b(:))]); title(sprintf('c_b, t=%.2f', step*dt)); axis equal tight; colorbar;
        subplot(2,2,3); imagesc(ubr .* g_phi, [0, .2]);colorbar;
        subplot(2,2,4); imagesc(.2.*g_phi, [0, .2]);colorbar;
        drawnow;
    end
end

%% ------------------ PDE functions ------------------ %%

figure(33); clf;
plot((1:Nt)*dt, total_cs/ total_cs(1)); % Normalize to initial value
hold on;
plot((1:Nt)*dt, total_cbs / total_cs(1));
hold on;
plot((1:Nt)*dt, total_cfs / total_cs(1));
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
function div = weighted_divergence(gphi, c, dx, dy)
    % Compute ∇·(g ∇c) using finite differences

    [Ny, Nx] = size(c);
    div = zeros(Ny, Nx);

    % X direction
    gpx = (gphi(:, [2:Nx, Nx]) + gphi(:, 1:Nx)) / 2;
    dcdx = (c(:, [2:Nx, Nx]) - c(:, [1, 1:Nx-1])) / dx;
    flux_x = gpx .* dcdx;
    dfluxdx = (flux_x(:, [2:Nx, Nx]) - flux_x(:, 1:Nx)) / dx;

    % Y direction
    gpy = (gphi([2:Ny, Ny], :) + gphi(1:Ny, :)) / 2;
    dcdy = (c([2:Ny, Ny], :) - c([1, 1:Ny-1], :)) / dy;
    flux_y = gpy .* dcdy;
    dfluxdy = (flux_y([2:Ny, Ny], :) - flux_y(1:Ny, :)) / dy;

    % Total divergence
    div = dfluxdx + dfluxdy;
end


function unbinding_rate_on_edge = unbinding_rate(gradient_field, k_off, phi, dphidy, psi, y_coms, y_center)
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
    if y_coms <= y_center || any(psi(:) .* phi(:) ~= 0)
        front_of_cell = double(mask_front > 0);
        % figure(777);
        % imagesc(front_of_cell)
        drawnow;
        %0.2 * w(phi) .* double(~mask_front)+0.2 * front_of_cell .* double(my_field == 0)+ my_field .* front_of_cell;
        unbinding_rate_on_edge = k_off * double(~mask_front) + k_off* front_of_cell .* double(gradient_field == 0)+ gradient_field .* front_of_cell;
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