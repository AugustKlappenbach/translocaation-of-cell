plotting = true;
% Set a nice theme once, up top in your script:
gap_sizes_um=[5,3]; 
forces = [.1,.2,.3];
for i= 1:length(gap_sizes_um)
    for j = 1:length(forces)
        pog(gap_sizes_um(i),forces(j))
    end
end
function pog(gap_size_um,force)
    % Set a nice theme once, up top in your script:
    theme_of_plots = 'spring';  % options: 'parula', 'hot', 'jet', 'gray', 'turbo', etc.

    reset(gpuDevice); % if you want a full GPU clear (may slow repeated calls)
%% ----------------  Paper PHYSICAL INPUT  ---------------- %%
    phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
    phys.Rpillar_um = 13.5;       % 
    phys.Rcell_um   = 6.5;         % 
    conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
    W      = 1.4;  
    start_point_offset_um=13;
    diffusion_constant_cf_um = 0.002; % diffusion coefficient in PF units
    k_fast = 0.2; % fast binding rate constant
    k_slow = 0.067; %slow binding rate constant
    k_off= k_fast; %unbinding rate constant
    %% ----------------- Implamentation in PF (phase-field Co-ordinates) --------------------------------- %%
    dx     = 0.8;  dy = dx;     % dx/W = 0.4
    dt     = 1e-3;                % tune if stable
    nSteps = 800/dt;
    save_interval = round(1/ dt);
    R_pillar = phys.Rpillar_um * conv; % 67.5
    R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
    gap_size = gap_size_um*conv;
    diffusion_constant_cf = diffusion_constant_cf_um * conv; % convert to PF units
    start_point_offset= start_point_offset_um * conv+2*conv; % offset from center in PF units
    velocity_start= 22.5e-6 * conv; % 2.4 um/tstep in PF units
    %% --- domain size --- %%
    Lx = 3*(R_cell);   
    Ly = 2*R_cell+2*start_point_offset + W*30;  
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);

    %saving
    gifFile = fullfile(getenv('HOME'), 'gifs', ...
    sprintf('R_D_through_gap_%d_C_affect_%d_starting_v_%d.gif', gap_size, force, velocity_start));

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
        y_center = y(centery);
        x_center = Lx/2;
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
        % both= psi + phi;
        % both = gpuArray(both);
        phi=gpuArray(phi);
        psi=gpuArray(psi);
        Y=gpuArray(Y);
        X=gpuArray(X);
        x=gpuArray(x);
        y=gpuArray(y);
        figure;
        % imagesc(both, [0 1]); axis equal tight;
        % colormap(theme_of_plots); colorbar;
%% ------------------ PDE functions ------------------ %%
    g= @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
    g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
    f_prime = @(phi) 8*phi.*(1-phi).*(1-2*phi);
    w = @(phi) (16*phi.^2.*(1-phi).^2);     
    % Cell volume weight
    
    phi_mask = phi > 0.5; % binary mask
%% ------------------ Initial conditions for RD ------------------ %%
    cf_w = .5.*w(phi);  % initial free concentration
    cb_w = .5.*w(phi);  % initial bound concentration
    y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));
    velocities = zeros(1, nSteps); % store velocity
    total_cs = zeros(1, nSteps); % store velocity
    total_cfs= zeros(1, nSteps); % store free concentration
    total_cbs= zeros(1, nSteps); % store bound concentration
    time_array = (0:nSteps-1) * dt;
    field=gradient_field(k_fast, k_slow, gap_size, x,X); % gradient field for reaction-diffusion
    %imagesc(gather(field), [0 .2]); axis image; axis off; hold on;
%% ------------------ getting repulsion force constant \lambda ------------------ %%
    lambda=(15/4) * W * force*5;
    v= velocity_start % initial velocity in PF units
    cb = cb_w ./ (w(phi)+1e-6); % bound concentration in PF units
    cf = cf_w ./ (w(phi)+1e-6); % free concentration in PF units
    %imagesc(gather(w(phi)), [0 1]); axis image; axis off; hold on;
%% ------------------- Main loop ------------------ %%
    % Loop over time steps
    for step = 1:nSteps
        % Stuff needed for each time step
        g_prime_phi=g_prime(phi);
        w_phi= w(phi);
        % ---------- forces -------------
        lap_phi   = my_laplacian(phi, dx, dy);
        lap_psi   = my_laplacian(psi, dx, dy);
        [dphiy, dphix] = my_gradient(phi, dx, dy);

%   %-- Force terms --%%
        tension      = 2*lap_phi - f_prime(phi);   
        interaction  = -lambda*lap_psi;
        
        F_advection= - 1*v*dphix; % advection in y-direction only v in y direction
        
        
        % Reaction-Diffusion force
        cb_w = w_phi .* cb; % bound concentration in PF units
        cf_w = w_phi .* cf; % free concentration in PF units
        F_RD= 10*cb_w.*force;
        %imagesc(gather(phi), [0 1]); axis image; axis off; hold on;
        
        F = tension + interaction + F_advection + F_RD;              
        % volume projection
        numerator   = sum(g_prime_phi.*F,'all');
        denominator = sum(g_prime_phi.^2,'all');
        p = numerator / (denominator);
        
        dphi_dt = F - p*g_prime_phi;
        phi     = phi + dt*dphi_dt; 
        %-- Reaction-Diffusion --%
        unbinding_rate_on_edge = unbinding_rate(field, k_off, phi, dphix, psi, w_phi, y_coms, y_center);
        binding_rate_on_edge = k_fast.* w_phi; % binding rate
        % Update concentrations
        [cb, cf] = update_RD(diffusion_constant_cf,phi,dphi_dt, cf, cb, binding_rate_on_edge, unbinding_rate_on_edge, w_phi, dx, dy, dt); 
        % Measure overlap
        % ----- pull translational velocity from w-field --------------------------
        v0           = v;
        if step < 300
            v = v0;
        else
            % --- projection-based drift, as before ---
            dwdt       = (32*phi .* (1-phi) .* (1-2*phi)) .* dphi_dt;
            [dwy, ~] = gradient(w_phi, dx, dy);
            num  = sum(dwdt .* dwy,'all');
            denom= sum(dwy.^2,'all');
            v    = -num / max(denom,eps);
        end
        v= (v + v0)/2; % average with previous velocity
        F_advection = - v .* dphiy;                 % use the fresh v
        velocities(step) = gather(v/conv);          % store in μm/s if you like
        phi_mask = phi > 0.5; 
        y_coms = sum(Y(phi_mask)) / sum(phi_mask(:));


        total_cf = sum(cf_w(:)) * dx * dy;
        total_cb = sum(cb_w(:)) * dx * dy;
        total_c  = total_cf + total_cb;
        total_cs(step)= total_c;
        total_cfs(step)= total_cf;
        total_cbs(step)= total_cb;
        velocities(step) = v; % store velocity
    %% -- updating gif ----------------------------%%

        if mod(step, save_interval) == 0 || step == 2
            if mod(step, save_interval*3) == 0 
                fprintf('%5d  v_proj = %.3e μm/s   COM = %.3e μm\n', step, gather(v/conv), gather(y_coms/conv));
            end
            %%-Plotting---%%%
            fig = figure('Visible', 'off');
            tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'compact');

            %-- Plot bound concentration --
            nexttile;
            imagesc(gather(cb_w), [0, max(cb_w(:))]); axis image; axis off; hold on;
            colormap(theme_of_plots); colorbar;
            contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  % Black contour at phi = 0.5
            title('bound concentration (C_b\cdot w(\phi))');
            %-- Plot free concentration --%%

            nexttile;
            imagesc(gather(cf_w), [0, max(cb_w(:))]); axis image; axis off; hold on;
            colormap(theme_of_plots); colorbar;
            contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  % Black contour at phi = 0.5
            title('free concentration (C_f\cdot w(\phi))');
            %%--Plot total concentration--%%

            nexttile;
            con_cf  = plot(NaN, NaN, 'b-', 'LineWidth', 2); hold on;
            con_cb  = plot(NaN, NaN, 'r-', 'LineWidth', 2);
            con_tot = plot(NaN, NaN, 'k--', 'LineWidth', 2);  % total

            xlabel('Time (s)'); ylabel('Concentration');
            title('Total, Free, and Bound Concentrations');
            legend({'c_f', 'c_b', 'total'}, 'Location', 'best');
            xlim([0, step*dt]);
            ylim([0, 1.2]);
            grid on;

            % Then update all three:
            set(con_cf,  'XData', time_array(2:step), 'YData', total_cfs(1:step-1)/total_cs(1));
            set(con_cb,  'XData', time_array(2:step), 'YData', total_cbs(1:step-1)/total_cs(1));
            set(con_tot, 'XData', time_array(2:step), 'YData', total_cs(1:step-1)/total_cs(1));

            %%-- Plot of the field --%%

            nexttile;
            imagesc(gather(unbinding_rate_on_edge), [0, max(unbinding_rate_on_edge(:))]); axis image; axis off; hold on;
            colormap(theme_of_plots); colorbar;
            contour(gather(psi), [0.5 0.5], 'k', 'LineWidth', 2);  % Black contour at phi = 0.5
            title('Unbinding rate on edge (\kappa_{off})');

            %%-- Plot ψ --%%
            nexttile;
            imagesc(x, y, gather(phi+psi), [0 1]); axis image; axis off; hold on;
            colormap(theme_of_plots); colorbar;
            plot(x_center, y_coms, 'r+', 'MarkerSize', 12, 'LineWidth', 2);
            title('\phi (Cell Shape)');

            %-- Velocity plot -- %%
            nexttile;
            velocity_plot = plot(NaN, NaN, 'LineWidth', 2); 
            xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
            xlim([0, step*dt]);
            ylim([0, max(max(velocities(1:step)), eps)]);
            title('Cell Velocity');
            grid on;
            hold on;
            set(velocity_plot, 'XData', time_array(2:step), 'YData', velocities(1:step-1));

            % %-- Write video frame --
            drawnow;
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

function [cb, cf] = update_RD(Df,phi,dphi_dt, cf, cb, binding_rate_on_edge, unbinding_rate_on_edge, wphi, dx, dy, dt)
    div_term = weighted_diffusion(cf, wphi, dx, dy);


    % ---------- reactive terms ----------
    react_cf = -binding_rate_on_edge .* cf + unbinding_rate_on_edge .* cb;
    react_cb =  binding_rate_on_edge .* cf - unbinding_rate_on_edge .* cb;

    % ---------- evolve the *weighted* variables ----------
    wc  = wphi .* cf;
    wb  = wphi .* cb;

    dwdt       = (32*phi .* (1-phi) .* (1-2*phi)) .* dphi_dt;   % derivative of w wrt φ times φ̇
    wc         = wc + dt*(Df*div_term + react_cf - cf.*dwdt);
    wb         = wb + dt*(             react_cb - cb.*dwdt);

    % ---------- recover plain concentrations (for plotting / reaction only) ----------
    cf  = wc ./ max(wphi,1e-6);   % safer ε
    cb  = wb ./ max(wphi,1e-6);

    % ---------- clean-up outside cell ----------


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
    unbinding_rate_on_edge= k_off * w_phi;  % default unbinding rate
    if y_coms <= y_center && any(psi(:) .* phi(:) > .01)
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