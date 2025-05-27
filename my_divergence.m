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
