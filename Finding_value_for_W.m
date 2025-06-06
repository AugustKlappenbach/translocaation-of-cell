dx = 0.8;
x = -5:dx:5;
W = 1.4;
phi = 0.5 * (1 - tanh(x / W));
plot(x, phi, 'LineWidth', 2);
xlabel('x'); ylabel('\phi'); title(sprintf('Phase Field Interface at W = %d dx= %d',W, dx));
%seems like 1.4 is a good value for W to have a nice transition. Shouldn't have to change my other code much, just the W value in my phi and psi field.