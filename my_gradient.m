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