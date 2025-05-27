plotting = false;
gap_sizes_um=[5,15,3]; 
forces = [.2,.3,.6];
for i= 1:length(gap_sizes_um)
    for j = 1:length(forces)
        pog(gap_sizes_um(i),forces(j))
    end
end
function pog(gap_size_um,force)
reset(gpuDevice); % if you want a full GPU clear (may slow repeated calls)
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
    cy_cell = round(.3*Ny);
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
    
    % Smooth tanh profiles for each pillar
    psi_left  = 0.5 * (1 - tanh((r_left  - R_pillar) / 1));
    psi_right = 0.5 * (1 - tanh((r_right - R_pillar) / 1));
    
    % Combine into psi field
    psi = psi_left + psi_right;
    both= psi + phi;
    phi=gpuArray(phi);
    psi=gpuArray(psi);
    Y=gpuArray(Y);
    X=gpuArray(X);
    x=gpuArray(x);
    y=gpuArray(y);
%% ------------------ PDE functions ------------------ %%
    g= @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
    g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
    f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
     %functions that will be used:
     %volumes=zeros(1, nSteps-1);
    
    % Cell volume weight
    phi_mask = phi > 0.5; % binary mask

    y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
    velocities = zeros(1, nSteps-1); % store velocity
    time_array = (0:nSteps-1) * dt;

%% ------------------- Main loop ------------------ %%
    % Loop over time steps
    for step = 1:nSteps
        % Inside time loop (at each step)
        
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
            % fprintf('step %d | max phi: %.2f | max lap_phi: %.2f | max gprime: %.2f | max F: %.2f\n', ...
            % step, max(phi(:)), max(lap_phi(:)), max(g_prime_phi(:)), max(F(:)));
            both = phi + psi;
            %Plotting initial cell.
            % subplot(1,3,1); imagesc(g_prime_phi); title("g'(\phi)");
            % subplot(1,3,2); imagesc(mu); title("μ band");
            % subplot(1,3,3); imagesc(mu .* g_prime_phi); title("μ × g'");

            fig = figure('Visible', 'on');
            tiledlayout(1,2, 'Padding', 'compact', 'TileSpacing', 'compact');

            %-- Plot φ --
            nexttile;
            imagesc(both, [0 1]); axis equal tight;
            colormap(spring); colorbar;
            title('\phi (Cell Shape)');
            nexttile;
            velocity_plot = plot(NaN, NaN, 'LineWidth', 2); 
            xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
            xlim([0, step*dt]);
            ylim([0, max(velocities)+1]);  % adjust this based on expected velocity range
            grid on;
            hold on;
            set(velocity_plot, 'XData', time_array(2:step), 'YData', velocities(1:step-1));
            drawnow limitrate;  % prevents lag by throttling redraw frequency 

            % ► GIF: grab frame and append
            drawnow;
            frame = getframe(fig);
            [im, map] = rgb2ind(frame.cdata, 256);

            if step == 2                            % first time → create file
                imwrite(im, map, gifFile, 'gif', ...
                    'LoopCount', inf, 'DelayTime', 0);  % DelayTime ~ seconds per frame
            else                                    % later → append
                imwrite(im, map, gifFile, 'gif', ...
                    'WriteMode', 'append', 'DelayTime', 0);
            end
        end
       
    
    
    end
end