    %% ----------------  Paper PHYSICAL INPUT  ---------------- %%
    phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
    phys.Rpillar_um = 13.5;       % [µm]
    phys.Rcell_um   = 10;         % [µm]
    conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
    W      = 1;  
    %% -----------------Paper -> Unitless! --------------------------------- %%
    dx     = 0.4*W;  dy = dx;     % dx/W = 0.4
    dt     = 1e-3/5;                % tune if stable
    nSteps = 1000/dt;
    save_interval = round(.2/ dt);
    R_pillar = phys.Rpillar_um * conv; % 67.5
    R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
    radius   = R_cell;                 % keep a short alias
    gap_size = gap_size_um*conv;
        %% --- domain size --- %%
    Lx = 3*(R_cell);   
    Ly = 9*R_cell;  
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);