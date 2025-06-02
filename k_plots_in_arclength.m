% Parameters
xi = linspace(-1, 1, 1000); % nucleus arc coordinate
k_off_fast = 0.2;           % s^-1
k_off_slow = 0.067;         % s^-1
k_on_prime = k_off_fast;    % by model assumption
xi_c = 0.6;                 % example value where force starts

% Compute k_off profile
k_off = k_off_fast * ones(size(xi));
idx = xi >= xi_c;
k_off(idx) = k_off_fast + (k_off_slow - k_off_fast) / (1 - xi_c) .* (xi(idx) - xi_c);

% Plot
figure;
plot(xi, k_off, 'b-', 'LineWidth', 2); hold on;
yline(k_on_prime, 'r--', 'LineWidth', 2);
xlabel('\xi (arc coordinate)');
ylabel('Rate (s^{-1})');
legend('k_{off}(\xi)', 'k_{on}''', 'Location', 'best');
title('Spatial profile of k_{off} vs constant k_{on}''');

% Highlight where they match
match_tol = 1e-3;
match_idx = abs(k_off - k_on_prime) < match_tol;
plot(xi(match_idx), k_off(match_idx), 'ko', 'MarkerSize', 4, 'DisplayName', 'match points');

legend show;
