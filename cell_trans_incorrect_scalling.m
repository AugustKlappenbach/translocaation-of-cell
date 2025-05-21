% cell transportation through a restricted thing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Physical parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma = 2;
h = sigma;
v=1;
lambda = 2;


%Grid spacings 
Nx = 200; Ny = 500;
dx = 0.1; dy = 0.1;
x = linspace(0, (Nx-1)*dx, Nx);
y = linspace(0, (Ny-1)*dy, Ny);
[X, Y] = meshgrid(x, y); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%time spacing
dt = 1e-4;
nSteps = 100/dt;
save_interval = round(.1 / dt); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%saving
gifFile = fullfile(getenv('HOME'), 'trans.gif');  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 9 point kernal
global kernel;
kernel =[1 4 1; 4 -20 4; 1 4 1] / 6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Smooth tanh initial phase field (soft cell):
%cell starting point
cy_cell = round(.2 * Ny);
cx_cell = round(Nx/2);

radius = 6.52;                  
transition_width = .5;  % adjust this for sharpness/smoothness

%defigning cell
r = sqrt((X - x(cx_cell)).^2 + (Y - y(cy_cell)).^2);
phi = 0.5 * (1 - tanh((r - radius) / transition_width));
phi_prev = phi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%pillars:
% Center coordinates (index space)
centerx = round(Nx/2);
centery = round(Ny/2);
% Parameters (µm)
pillar_radius   = 8;        % µm Same
gap_size        = 7;       % µm between circle edges  (CHANGE)
wall_thickness  = .5;       % µm tanh transition zone


% Derived horizontal center-to-center distance
gap_distance = 2 * pillar_radius + gap_size;
center_offset = gap_distance / 2;

% Horizontal circle centers (in µm)
x_left  = x(centerx) - center_offset;
x_right = x(centerx) + center_offset;
y_center = y(centery);
x_center = x(centerx);

% Distance fields from each center
r_left  = sqrt((X - x_left).^2  + (Y- y_center).^2);
r_right = sqrt((X - x_right).^2 + (Y - y_center).^2);

% Smooth tanh profiles for each pillar
psi_left  = 0.5 * (1 - tanh((r_left  - pillar_radius) / wall_thickness));
psi_right = 0.5 * (1 - tanh((r_right - pillar_radius) / wall_thickness));

% Combine into psi field
psi = psi_left + psi_right;
psi(psi > 1) = 1;  % clip to avoid values > 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 volumes=zeros(1, nSteps-1);
g= @(phi) phi.^3.*(10 + 3*phi.*(2*phi-5));
g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
f = @(phi) 18*phi.^2.*(1- phi).^2;
f_prime = @(phi) 8*(1 - phi).^2.*phi - 8*(1 - phi).*phi.^2;

%Plotting initial cell.
fig = figure('Visible', 'on');
    imagesc(phi, [0 1]); axis equal tight; hold on;
    contour(psi, [0.5 0.5], 'k', 'LineWidth', 2);
    title('\phi with wall overlay'); colorbar;
figure;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for step = 1:nSteps
    % ---------- forces -------------
    volumes(step)=sum(g(phi(:)));
    lap_phi   = laplacian9(phi,dx,dy);
    lap_psi   = laplacian9(psi,dx,dy);
    
    tension      = 2*sigma*lap_phi - h*f_prime(phi);   
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
        x_band = abs(X - x_center) < gap_size / 4;
    
        mu = double(x_band & y_band);
    else
        mu = zeros(size(phi));  % fallback if phi is gone
    end


    
    
    % Only apply frontal force where dphix is positive (elementwise)
    front = v * mu .* g_prime(phi);
    
    
    F = tension + interaction + front;               
    
    % volume projection
    phi(phi>1)=1;  phi(phi<0)=0;                       % clip before g′
    numerator   = sum(g_prime(phi).*F,'all');
    denominator = sum(g_prime(phi).^2,'all') + 1e-8;
    p = numerator / (h*denominator);
    
    dphi_dt = F - p*h*g_prime(phi);
    phi     = phi + dt*dphi_dt;
    phi(phi>1)=1;  phi(phi<0)=0;                       % clip after update
       
  
if max(abs(F(:))) > 1e3
    disp("Max F:" + max(abs(F(:))))
    disp("time step:"+step)
    break   
end

 % -- updating gif ----------------------------
if mod(step, save_interval) == 0 || step == 2
    both = phi + psi;
    V0 = volumes(1);  % or mean(volumes) if that makes more sense
            vol_range = max(volumes(1:step)) - min(volumes(1:step));
            percent_variation = 100 * vol_range / V0;
            disp(['% volume variation: ', num2str(percent_variation, '%.10f'), '%'])
    %Plotting initial cell.
    fig = figure('Visible', 'on');
    tiledlayout(1,2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % -- Plot φ --
    nexttile;
    imagesc(both, [0 1]); axis equal tight;
    colormap(parula); colorbar;
    title('\phi (Cell Shape)');
    
    % -- Plot front force --
    nexttile;
    imagesc(front); axis equal tight;
    colormap(parula); colorbar;
    title('Frontal Force');


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


function lap = laplacian9(phi, dx, dy)
    kernel =[1 4 1; 4 -20 4; 1 4 1] / 6;
    lap = conv2(phi, kernel, 'same') / dx^2;
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
function div = my_divergence(vx, vy, dx, dy)
%MY_DIVERGENCE Fast 2D divergence for uniform grid spacing
%   div = my_divergence(vx, vy, dx, dy)
%
%   Inputs:
%     vx - x-component of vector field (same size as vy)
%     vy - y-component of vector field
%     dx - spacing in x-direction
%     dy - spacing in y-direction
%
%   Output:
%     div - scalar divergence field ∇·v = ∂vx/∂x + ∂vy/∂y

    if nargin < 3
        dx = 1;
    end
    if nargin < 4
        dy = 1;
    end

    [Ny, Nx] = size(vx);
    div = zeros(Ny, Nx);

    % ∂vx/∂x
    dvxdx = zeros(Ny, Nx);
    dvxdx(:,2:Nx-1) = (vx(:,3:Nx) - vx(:,1:Nx-2)) / (2*dx);
    dvxdx(:,1)      = (vx(:,2) - vx(:,1)) / dx;
    dvxdx(:,end)    = (vx(:,end) - vx(:,end-1)) / dx;

    % ∂vy/∂y
    dvydy = zeros(Ny, Nx);
    dvydy(2:Ny-1,:) = (vy(3:Ny,:) - vy(1:Ny-2,:)) / (2*dy);
    dvydy(1,:)      = (vy(2,:) - vy(1,:)) / dy;
    dvydy(end,:)    = (vy(end,:) - vy(end-1,:)) / dy;

    % Total divergence
    div = dvxdx + dvydy;
end
