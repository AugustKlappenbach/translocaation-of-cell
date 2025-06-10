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
    v=force;
    %% --- domain size --- %%
    Lx = 2.5*(R_cell);   
    Ly = 2*R_cell+2*start_point_offset;  
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);

    % %saving
    % videoFile = fullfile(getenv('HOME'), 'videos', ...
    % sprintf('trans_gap%d_%d.mp4', gap_size, v));

    % vWriter = VideoWriter(videoFile, 'MPEG-4');
    % vWriter.FrameRate = 10;  % or whatever frame rate you want
    % open(vWriter);

    %% Cell init:
    %Where we starting at?
    cy_cell = round((.5-start_point_offset/Ly)*Ny);
    cx_cell = round(.5*Nx);
    %soft cell:
    r= sqrt((X-x(cx_cell)).^2 + (Y-y(cy_cell)).^2);
    phi = 0.5 * (1 - tanh((r - R_cell)/W));
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
    
    % Smooth tanh profiles for each pillar
    psi_left  = 0.5 * (1 - tanh((r_left  - R_pillar) / W));
    psi_right = 0.5 * (1 - tanh((r_right - R_pillar) / W));
    
    % Combine into psi field
    psi = psi_left + psi_right;
    both= psi + phi;
    % both = gpuArray(both);
    % phi=gpuArray(phi);
    % psi=gpuArray(psi);
    % Y=gpuArray(Y);
    % X=gpuArray(X);
    % x=gpuArray(x);
    % y=gpuArray(y);










    %% ------------------ PDE functions ------------------ %%
    g= @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
    g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
    f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
    w = @(phi) 4*phi.*(1-phi); 
    %functions that will be used:
    %volumes=zeros(1, nSteps-1);
    y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
    velocities = zeros(1, nSteps-1); % store velocity
    time_array = (0:nSteps-1) * dt;
%% ------------------ getting repulsion force constant \lambda ------------------ %%
    lambda=-(15/4) * W * v;
    
%% ------------------- Main loop ------------------ %%
    % Loop over time steps
    for step = 1:nSteps
        % Inside time loop (at each step)
        w_phi = w(phi);
        g_prime_phi=g_prime(phi);
        % ---------- forces -------------
        lap_phi   = 4*del2(phi,dx,dy)/(dx*dy);
        lap_psi   = 4*del2(psi,dx,dy)/(dx*dy);
    
        tension      = 2*lap_phi - f_prime(phi);   
        interaction  = -lambda*lap_psi;
        % Apply mask in x-direction near the center
        % Find lowest y-position with phi > threshold
        threshold = 0.5; % or whatever works for your cell's body
        phi_mask = phi > threshold;
        y_indices = find(any(phi_mask, 2));  % rows where phi > threshold
        if ~isempty(y_indices)
            bottom_idx = max(y_indices);
            y_bottom = y(bottom_idx);
            
            % Create band mask just above that region
            y_band = (Y > y_bottom - gap_size*0.5) & (Y < y_bottom + gap_size*0.5);
            x_band = abs(X - x_center) < gap_size / 2;
        
            mu = double(x_band & y_band);
        else
            mu = zeros(size(phi));  % fallback if phi is gone
        end
    
        % Only apply frontal force where dphix is positive (elementwise)
        front = v * mu .* g_prime_phi;
        
        F = tension + interaction + front;               
        % volume projection
        numerator   = sum(g_prime_phi.*F,'all');
        denominator = sum(g_prime_phi.^2,'all');
        p = numerator / (denominator);
        
        dphi_dt = F - p*g_prime_phi;
        phi     = phi + dt*dphi_dt;  
        y_coms_old=y_coms;
        y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
        velocities(step) =  y_coms_old + y_coms*dt;
%% -- updating gif ----------------------------
        if mod(step, save_interval) == 0 || step == 2
            fig = figure('Visible', 'off');
            tiledlayout(1,2, 'Padding', 'compact', 'TileSpacing', 'compact');

            %-- Plot φ --
            nexttile;
            imagesc(gather(phi), [0 1]); axis image; axis off; hold on;
            colormap(spring); colorbar;
            contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  % Black contour at phi = 0.5
            title('\phi (Cell Shape)');

            %-- Velocity plot --
            nexttile;
            velocity_plot = plot(NaN, NaN, 'LineWidth', 2); 
            xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
            xlim([0, step*dt]);
            ylim([0, max(velocities)+1]);
            grid on;
            hold on;
            set(velocity_plot, 'XData', time_array(2:step), 'YData', velocities(1:step-1));

            % %-- Write video frame --
            % drawnow;
            % frame = getframe(fig);
            % writeVideo(vWriter, frame);
        end
    end
end