clc; clear; close all;
gap_size_um = 10; % [µm]
force = 1; % [pN]
%% ----------------  Paper PHYSICAL INPUT  ---------------- %%
    phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
    phys.Rpillar_um = 13.5;       % [µm]
    phys.Rcell_um   = 10;         % [µm]
    conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
    W      = 1;  
    %% -----------------Paper -> Unitless! --------------------------------- %%
    dx     = 0.4;  dy = dx;     % dx/W = 0.4
    dt     = 1e-3/2;                % tune if stable
    nSteps = 200/dt;
    save_interval = round(.2/ dt);
    R_pillar = phys.Rpillar_um * conv; % 67.5
    R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
    gap_size = gap_size_um*conv;
    lambda=2;
    v=force;
    %% --- domain size --- %%
    Lx = 3*(R_cell);   
    Ly = 7*R_cell;  
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);

    %saving
    gifFile = fullfile(getenv('HOME'), 'gifs', ...
        sprintf('trans_gap%d_%d.gif', gap_size,v));
    %% Cell init:
    %Where we starting at?
    cy_cell = round(.2*Ny);
    cx_cell = round(.5*Nx);
    %soft cell:
    r= sqrt((X-x(cx_cell)).^2 + (Y-y(cy_cell)).^2);
    phi = 0.5 * (1 - tanh((r - R_cell)/1));
   
    %% Pillars:
    % Distance fields from each center
    centerx = round(Nx/2);
    centery = round(Ny/2);
    x_center=x(centerx);
    y_center = y(centery);
    gap_distance = 2 * R_pillar + gap_size;
    center_offset = gap_distance / 2;
    x_left  = x(centerx) - center_offset;
    x_right = x(centerx) + center_offset;
    r_left  = sqrt((X - x_left).^2  + (Y- y_center).^2);
    r_right = sqrt((X - x_right).^2 + (Y - y_center).^2);
    x_center = max(x) / 2;  % or x(end)/2
    gap_size = round(gap_size/dx)*dx;

    x_left = x_center - gap_size/2;
    x_right = x_center + gap_size/2;

    my_field = zeros(size(X));
  % Left half of gap: 1 → 2
    left_mask = (X >= x_left) & (X <= x_center);
    my_field(left_mask) = 2 - (X(left_mask) - x_left) / (gap_size/2);
    % Right half of gap: 2 → 1
    right_mask = (X > x_center) & (X <= x_right);
    my_field(right_mask) = 1 + (X(right_mask) - x_center) / (gap_size/2);
    % Smooth tanh profiles for each pillar
    psi_left  = 0.5 * (1 - tanh((r_left  - R_pillar) / 1));
    psi_right = 0.5 * (1 - tanh((r_right - R_pillar) / 1));
    g = @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
    w = @(phi) phi.*(1-phi);
    % Combine into psi field
    psi = psi_left + psi_right;
    both= psi + phi;
    [dphix, dphidy] = my_gradient(phi, dx, dy);
    mask_front =  (dphidy <  0);
    front_of_cell= zeros(Ny, Nx);
    front_of_cell(mask_front) = 4*w(phi(mask_front));
    max(w(phi(:))*4)
    imagesc( front_of_cell.*my_field, [0 2]); axis equal tight;
    colormap(spring); colorbar;
    title('\phi (Cell Shape)');
