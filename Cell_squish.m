% --------------------------
% Parameters
% --------------------------
N = 600;            % Grid size
Nsteps = 100000;     % Total time steps
Nplot = 1000;       % Plot every Nplot steps

% PDE coefficients
k = 1;
sigma = 2.5;
h = 1;
dx = 0.8;
dt = 0.01;
dx2 = dx^2;
lambda = 1;
vx = -0.25;

% --------------------------
% Helper functions
% --------------------------
fprime = @(phi) 8.0 .* phi .* (2 .* phi.^2 - 3 .* phi + 1);
gprime = @(phi) 30.0 .* (phi - 1).^2 .* phi.^2;

laplacian = @(A) (circshift(A, [1 0]) + circshift(A, [-1 0]) + ...
                  circshift(A, [0 1]) + circshift(A, [0 -1]) - 4 * A) / dx2;

% --------------------------
% Initial conditions
% --------------------------
phi = zeros(N); %#ok<PREALL>
phidot = zeros(N);
psi = zeros(N); %#ok<PREALL>
psi_left = zeros(N); %#ok<PREALL>
psi_right = zeros(N); %#ok<PREALL>
psi_left_dot = zeros(N);
psi_right_dot = zeros(N);

[X, Y] = meshgrid(1:N, 1:N);
X = X'; Y = Y'; % To match (i,j) indexing like Python

R = 75;
width = 3;
cx = N/2; cy = N/2;

dist = sqrt((X - cx).^2 + (Y - cy).^2);
phi = 0.5 * (1 - tanh((dist - R) / width));

dist_left = sqrt((X - cx).^2 + (Y - (cy - 300)).^2);
dist_right = sqrt((X - cx).^2 + (Y - (cy + 300)).^2);
psi_left = 0.5 * (1 - tanh((dist_left - 2*R) / width));
psi_right = 0.5 * (1 - tanh((dist_right - 2*R) / width));
psi = psi_left + psi_right;

% --------------------------
% Visualization init
% --------------------------
figure;
imagesc(phi + psi, [0 1]); colormap cool; axis equal tight;
colorbar; title('Initial Condition: φ + ψ');

% --------------------------
% Main loop
% --------------------------
frames = {};

for step_count = 1:Nsteps
     % Calculate p (numerator and denominator)
    lap_phi = laplacian(phi);
    gp_phi = gprime(phi);
    psi_total = psi_left + psi_right;

    numerator = sum(gp_phi .* k .* (sigma * lap_phi - fprime(phi) - lambda * psi_total .* phi), 'all');
    denominator = sum(h * gp_phi.^2, 'all');
    if abs(denominator) > 1e-14
        p = numerator / denominator;
    else
        p = 0.0;
    end

    % Update phidot
    phidot = k * (sigma * lap_phi - fprime(phi) - p * h * gp_phi) - lambda * psi_total .* phi;

    % Approximate drift term
    dpsiL_dx = (circshift(psi_left, [0 -1]) - circshift(psi_left, [0 1])) / (2 * dx);
    dpsiR_dx = (circshift(psi_right, [0 -1]) - circshift(psi_right, [0 1])) / (2 * dx);
    psi_left_dot = vx * dpsiL_dx;
    psi_right_dot = -vx * dpsiR_dx;

    % Euler update
    phi = phi + dt * phidot;
    psi_left = psi_left + dt * psi_left_dot;
    psi_right = psi_right + dt * psi_right_dot;
    psi = psi_left + psi_right;

    % Save frame
    if mod(step_count, Nplot) == 0
        fig = figure('Visible', 'off');
        imagesc(phi + psi, [0 1]); axis equal tight;
        colormap(parula);
        title(sprintf('\\phi + \\psi (step %d)', step_count));
        drawnow;

        % Capture frame
        frame = getframe(fig);
        frames{end+1} = frame.cdata; %#ok<SAGROW>
        close(fig);
    end
end

% --------------------------
% Save GIF
% --------------------------
filename = fullfile(getenv('HOME'), 'phasefield_sim.gif');

for idx = 1:length(frames)
    [A, map] = rgb2ind(frames{idx}, 256); % Convert RGB to indexed
    if idx == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', inf, 'DelayTime', 1/5);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/5);
    end
end

disp(['✨ GIF saved as ', filename, ' ✨']);

