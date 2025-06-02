function phi = phase_field(X, Y, x_center, y_center, R_cell, W)
    %PHASE_FIELD Generate a phase field for a circular cell
    %   phi = phase_field(X, Y, x_center, y_center, R_cell, W)
    %
    %   Inputs:
    %     X - 2D grid of x-coordinates
    %     Y - 2D grid of y-coordinates
    %     x_center - x-coordinate of the cell center
    %     y_center - y-coordinate of the cell center
    %     R_cell - radius of the cell
    %     W - width of the transition region
    %
    %   Output:
    %     phi - phase field representing the cell

    r = sqrt((X - x_center).^2 + (Y - y_center).^2);
    phi = 0.5 * (1 - tanh((r - R_cell) / W));
end
