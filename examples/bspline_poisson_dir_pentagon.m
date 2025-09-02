clear
pkg load nurbs
pkg load geopdes

%% Make geometry
r = 1;
n = 5;

% Vertex positions
phi = 2*pi / n;
coords = zeros(2*n+1,2);
for i = 0:n-1
    phi_i = phi * i;
    phi_j = phi * (i + 1);
    p_i = [r * cos(phi_i), r * sin(phi_i)];
    p_j = [r * cos(phi_j), r * sin(phi_j)];
    coords(2*i+2,:) = p_i;
    coords(2*i+3,:) = (p_i + p_j)/2;
end

% Nurbs squares
npatch = 5;
c = [
    0, 10, 1, 2;
    0, 2, 3, 4;
    0, 4, 5, 6;
    0, 6, 7, 8;
    0, 8, 9, 10;
];

hold on
for i = 1:npatch
    idx = c(i,:) + 1;
    patches(i) = nrb4surf(coords(idx(1),:), coords(idx(2),:), coords(idx(4),:), coords(idx(3),:));
    %nrbplot(patches(i), [10, 10]);
end

[geo, bnd, gamma, ~, bnd_gamma] = mp_geo_load(patches);
bnd_gamma_idx = 1:length(bnd_gamma);

%% Define problem
f = @(x,y) ones(size(x));
g = @(x,y,idx) zeros(size(x));
c = @(x,y) ones(size(x));

%% Define discretization
nsub  = 1;
p     = 3;
k     = 2;
nquad = p+1;

msh   = cell(1, npatch);
sp    = cell(1, npatch);

for iptc = 1:npatch
    [knots, zeta] = kntrefine(geo(iptc).nurbs.knots, [nsub, nsub]-1, [p, p], [k, k]);

    % Construct msh structure
    rule      = msh_gauss_nodes([nquad, nquad]);
    [qn, qw]  = msh_set_quad_nodes(zeta, rule);
    msh{iptc} = msh_cartesian(zeta, qn, qw, geo(iptc));

    % Construct space
    sp{iptc} = sp_bspline(knots, [p, p], msh{iptc});
end

%% Assembly
msh = msh_multipatch(msh, bnd);
space = sp_multipatch(sp, msh, gamma, bnd_gamma);
clear sp

stiff_mat = op_gradu_gradv_mp(space, space, msh, c);
rhs = op_f_v_mp(space, msh, f);

%% Solve
% Apply Dirichlet boundary conditions
u = zeros(space.ndof, 1);
[u_drchlt, drchlt_dofs] = sp_drchlt_l2_proj(space, msh, g, bnd_gamma_idx);
u(drchlt_dofs) = u_drchlt;
int_dofs = setdiff(1:space.ndof, drchlt_dofs);

% Solve the linear system
rhs(int_dofs) = rhs(int_dofs) - stiff_mat(int_dofs, drchlt_dofs)*u_drchlt;
u(int_dofs) = stiff_mat(int_dofs, int_dofs) \ rhs(int_dofs);

% Plot
sp_plot_solution(u, space, geo, [10, 10])
