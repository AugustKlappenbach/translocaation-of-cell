plotting = true;
gap_sizes_um=[5,3]; 
forces = [.09,.08,.07];
for i= 1:length(gap_sizes_um)
    for j = 1:length(forces)
        pog(gap_sizes_um(i),forces(j))
    end
end
function pog(gap_size_um,force)
    %reset(gpuDevice); % if you want a full GPU clear (may slow repeated calls)
%% ----------------  Paper PHYSICAL INPUT  ---------------- %%
    phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
    phys.Rpillar_um = 13.5;       % [µm]
    phys.Rcell_um   = 6.5;         % [µm]
    conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
    W      = 1.4;  
    start_point_offset_um=13;
    %% -----------------Paper -> Unitless! --------------------------------- %%
    dx     = 0.8;  dy = dx;     % dx/W = 0.4
    dt     = 1e-3;                % tune if stable
    nSteps = 800/dt;
    save_interval = round(.5/ dt);
    R_pillar = phys.Rpillar_um * conv; % 67.5
    R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
    gap_size = gap_size_um*conv;
    start_point_offset= start_point_offset_um * conv; % offset from center in PF units
    volume_at_start = R_cell^2*pi;
    v=force;
    k_fast =.067;
    k_slow=.2;
    %% --- domain size --- %%
    Lx = 2.5*(R_cell);   
    Ly = 2*R_cell+2*start_point_offset+20;  
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);

    %saving
    gifFile = fullfile(getenv('HOME'), 'gifs', ...
    sprintf('new_grad_gap_size_%d_F_affect_%d.gif', gap_size, force));

    %% Cell init:
    %Where we starting at?
    cy_cell = round((.5-start_point_offset/Ly)*Ny);
    cx_cell = round(.5*Nx);
    %soft cell:
    r= sqrt((X-x(cx_cell)).^2 + (Y-y(cy_cell)).^2);
    phi = 0.5 * (1 - tanh((r - R_cell)/W));
    %% Pillars:
    % Distance fields from each center
    M_a = 1/(conv^3);
    centerx = round(Nx/2);
    centery = round(Ny/2);
    y_center = y(centery);
    gap_distance = 2 * R_pillar + gap_size;
    center_offset = gap_distance / 2;
    x_left  = x(centerx) - center_offset;
    x_right = x(centerx) + center_offset;
    r_left  = sqrt((X - x_left).^2  + (Y- y_center).^2);
    r_right = sqrt((X - x_right).^2 + (Y - y_center).^2);
    
    % Smooth tanh profiles for each pillar
    psi_left  = 0.5 * (1 - tanh((r_left  - R_pillar) / W));
    psi_right = 0.5 * (1 - tanh((r_right - R_pillar) / W));
    
    % Combine into psi field
    threshold = 0.5; % or whatever works for your cell's body
    phi_mask = phi > threshold;
    psi = psi_left + psi_right;
    both= psi + phi;
    both = gpuArray(both);
    phi=gpuArray(phi);
    psi=gpuArray(psi);
    Y=gpuArray(Y);
    X=gpuArray(X);
    % x=gpuArray(x);
    % y=gpuArray(y);
    my_field = zeros(size(X), 'like', X);   % allocate on the GPU
    %% ------------------ PDE functions ------------------ %%
    g= @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
    g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
    f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
    %functions that will be used:
    %volumes=zeros(1, nSteps-1);
    y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
    velocities = zeros(1, nSteps-1); % store velocity
    time_array = (0:nSteps-1) * dt;
    [d_phi_dx, d_phi_dy] = my_gradient(phi,dx,dy);
    w_phi=(16*phi.^2.*(1-phi).^2);
    front_of_cell = (d_phi_dy < 0);
    velocities = zeros(1, nSteps); % store velocity
    volumes= zeros(1, nSteps);
    y_coms_array = zeros(1, nSteps);  % center of mass vs time
    max_saves = floor(nSteps / save_interval) + 1;
    y_coms_array = zeros(1, max_saves);
    volume_array = zeros(1, max_saves);
    save_counter = 1;  % index for storing save-specific quantities

%% ------------------ getting repulsion force constant \lambda ------------------ %%
    lambda=(15/4) * W * v;
    grad_field = gradient_field(k_fast, k_slow,gap_size, x, X);
    figure(33)
    imagesc(grad_field, [min(grad_field(:)), max(grad_field(:))]);
    colorbar;
%% ------------------- Main loop ------------------ %%
    % Loop over time steps
    for step = 1:nSteps
        [d_phi_dx, d_phi_dy] = my_gradient(phi,dx,dy);
        front_of_cell = (d_phi_dy < 0);
        y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
        lap_phi   = my_laplacian(phi, dx, dy);
        lap_psi   = my_laplacian(psi, dx, dy);
        norm_dphi = sqrt(d_phi_dx.^2 + d_phi_dy.^2) + 1e-12;
        n_x = d_phi_dx ./ norm_dphi;
        n_y = d_phi_dy ./ norm_dphi;
        n = n_x + n_y;
        pull= unbinding_rate(phi,d_phi_dy, grad_field, front_of_cell, y_coms, y_center);
        imagesc(-pull.*n); colorbar;
        drawnow;
        break;
        tension      = 2*lap_phi - f_prime(phi);   
        interaction  = -lambda*lap_psi;
        vol=sum(phi(:),'all')*dx*dy;
        Vomule = -M_a*(vol - volume_at_start).*norm_dphi;
        pulling = 1*force*pull;
        phi = phi + dt*(tension + interaction + Vomule + pulling);
        y_coms_old=y_coms;
        y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
        velocities(step) =  y_coms_old + y_coms*dt;
        volume_array(step) = vol;
%% -- updating gif ----------------------------
        if mod(step, save_interval) == 0 || step == 2
        % --- Compute volume and COM ---
        

        % --- Plotting ---
        fig = figure('Visible', 'off');  % toggle visibility with flag
        tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');

        %-- Plot φ --
        nexttile;
        imagesc(gather(phi), [0 1]); axis image; axis off; hold on;
        colormap(spring); colorbar;
        contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  
        title('\phi (Cell Shape)');

        %-- Plot pulling field --
        nexttile;
        imagesc(gather(pulling), [0 max(gather(pulling(:)))]); axis image; axis off; hold on;
        colormap(spring); colorbar;
        contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  
        title('Pulling Field');
        nexttile;
        velocity_plot = plot(NaN, NaN, 'LineWidth', 2); 
        xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
        xlim([0, step*dt]);
        ylim([0, max(velocities)+1]);
        grid on;
        hold on;
        set(velocity_plot, 'XData', time_array(2:step), 'YData', velocities(1:step-1));
        nexttile;
        vol_plot = plot(NaN, NaN, 'LineWidth', 2); 
        xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
        xlim([0, step*dt]);
        ylim([0, 1.2]);
        grid on;
        hold on;
        set(vol_plot, 'XData', time_array(2:step), 'YData', volume_array(1:step-1)/volume_array(1));
            frame = getframe(fig);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            
            if step == save_interval || step == 2
                imwrite(imind, cm, gifFile, 'gif', 'Loopcount', inf, 'DelayTime', 0.2);
            else
                imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
            end
            close(fig); % Close the figure to avoid memory issues
        end
    end
end

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

function pulling = unbinding_rate(phi,dphidy, gradient_field, front_of_cell, y_coms, y_center)
    % Calculate the unbinding rate based on the reaction kinetics
    % k_on: binding rate constant (int)
    % k_off: unbinding rate constant (int)
    % phi: phase field  (matrix for position and time)
    % dphidy: gradient of phase field (matrix for position)
    % psi: Pillars (matrix for position and time)
    % y_coms: center of mass for the cell (scalar)
    % dx, dy: spatial discretization steps
    % unbinding_rate_on_edge = k_off * w_phi;  % default unbinding rate
    % imagesc(gather(front_mask), [0 1]); axis image; axis off; hold on;
    % Ensure inputs are valid
    mask_front =  (dphidy <  0);
    figure(420);
    imagesc(mask_front);
    w_phi=(16*phi.^2.*(1-phi).^2);
    pulling = .2*front_of_cell.*w_phi.*double(gradient_field~=0);
    if y_coms <= y_center
        pulling = front_of_cell.*w_phi.*gradient_field;
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