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