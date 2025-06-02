force=1;
gap_size_um = 10; % [µm] gap size between pillars
%% ----------------  Paper PHYSICAL INPUT  ---------------- %%
phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
phys.Rpillar_um = 13.5;       % [µm]
phys.Rcell_um   = 10;         % [µm]
conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
W      = 1;  
%% -----------------Paper -> Unitless! --------------------------------- %%
dx     = 0.4;  dy = dx;     % dx/W = 0.4
dt     = 1e-3;                % tune if stable
nSteps = 600/dt;
save_interval = round(.2/ dt);
R_pillar = phys.Rpillar_um * conv; % 67.5
R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
gap_size = gap_size_um*conv;
lambda=2;
v=force;
Df = .007;
kon = .2;
save_rate = 50; % save every 50 steps
%% --- domain size --- %%
Lx = 3*(R_cell);   
Ly = 9*R_cell;  

Nx = ceil(Lx/dx);
Ny = ceil(Ly/dy);

x  = (0:Nx-1)*dx;
y  = (0:Ny-1)*dy;
[X,Y] = meshgrid(x,y);
%% --- Cell --- %%
cy_cell = round(.2*Ny);
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
imagesc(psi + phi, [0 1]); axis equal tight;
colormap(spring); colorbar;
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
% colormap(spring); colorbar;
%title('my field');

%% ------------------ PDE functions ------------------ %%
g = @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
w = @(phi) 4*phi.*(1-phi); 

[~, dphidy] = my_gradient(phi, dx, dy);
mask_front =  (dphidy <  0);
front_of_cell = w(phi).*double(mask_front > 0);
un_binding_rate =front_of_cell .* my_field;
unbinding_rate = 0.2 * w(phi) .* double(~mask_front)+0.2 * front_of_cell .* double(my_field == 0)+ my_field .* front_of_cell;
binding_rate =.2*w(phi);
imagesc(binding_rate, [0 max(binding_rate(:))]); axis equal tight;
colormap(spring); colorbar;
sample_idx = 0
cf=.5.*w(phi);  % initial free concentration
cb = cf;
%% ------------------ PDE functions ------------------ %%
for step = 1:nSteps
    % Inside time loop (at each step)
    w_phi = w(phi);
    g_prime_phi=g_prime(phi);
    % ---------- forces -------------
    lap_phi   = 4*del2(phi,dx,dy)/(dx*dy);
    lap_psi   = 4*del2(psi,dx,dy)/(dx*dy);
    
    % --- Gradient and diffusion ---
    [cf_x, cf_y] = gradient(cf, dx, dy);
    div_diff_cf = divergence(w_phi .* cf_x, wphi .* cf_y);

    % --- Reaction terms ---
    react_cf = (-binding_rate .* cf + unbinding_rate .* cb);
    react_cb = (unbinding_rate .*  kon .* cf - binding_rate .* cb);

    % --- Time update ---
    cf = cf + dt * (Df * div_diff_cf + react_cf);
    cb = cb + dt * react_cb;
    if mod(step, save_rate) == 0 || step == 1
        sample_idx = sample_idx + 1;
        idx = step / save_rate;
        % weight localised to the interface
        nexttile(1); imagesc(un_binding_rate, [0, .2]); axis equal tight; title('\phi');
        nexttile(2); imagesc(cf, [0,.5]); colorbar; axis equal tight; title('c_f');
        nexttile(3); imagesc(cb,  [0 , .5])/,; colorbar; axis equal tight; title('c_b');
        sgtitle(sprintf('Step %d / %d /n cf = %d, cd = %d', step, nSteps));
        drawnow;
    end
end

