clear
pkg load nurbs
pkg load geopdes

%% Define solution functions
function coeffs = calc_coeffs(coords)
    % Get x and y coordinates of corner points
    coords = coords(2:2:end, :);
    xs = coords(:, 1);
    ys = coords(:, 2);

    % Calculate coefficients
    a = -diff([ys; ys(1)])'; % Circular difference
    b = -diff([xs; xs(1)])'; % Circular difference
    c = arrayfun(@(i) coords(i, 1) * coords(mod(i, size(coords, 1)) + 1, 2) - ...
                 coords(mod(i, size(coords, 1)) + 1, 1) * coords(i, 2), ...
                 1:size(coords, 1))';
    coeffs = {a, b, c};
end

function result = eval_factor(coeffs, i, x, y)
    a = coeffs{1};
    b = coeffs{2};
    c = coeffs{3};
    result = a(i) * x + b(i) * y + c(i);
end

function result = eval_product(coeffs, x, y)
    result = 1;
    for i = 1:5
        result = result .* eval_factor(coeffs, i, x, y);
    end
end

function result = eval_deriv_summand(coeffs, j, x, y)
    result = 1;
    for i = 1:5
        if i ~= j
            result = result .* eval_factor(coeffs, i, x, y);
        end
    end
end

function result = eval_deriv(deriv_coeffs, coeffs, x, y)
    result = 0;
    for j = 1:5
        result = result + deriv_coeffs(j) .* eval_deriv_summand(coeffs, j, x, y);
    end
end

function result = eval_second_deriv_summand(coeffs, k, j, x, y)
    result = 1;
    for i = 1:5
        if i ~= k && i ~= j
            result = result .* eval_factor(coeffs, i, x, y);
        end
    end
end

function result = eval_second_deriv(deriv_coeffs, coeffs, x, y)
    result = 0;
    for k = 1:5
        for j = 1:5
            if k ~= j
                result = result + eval_second_deriv_summand(coeffs, k, j, x, y) .* deriv_coeffs(k) .* deriv_coeffs(j);
            end
        end
    end
end

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
coeffs = calc_coeffs(coords);
u = @(x,y) eval_product(coeffs, x, y);

u_dx = @(x,y) eval_deriv(coeffs{1}, coeffs, x, y);
u_dy = @(x,y) eval_deriv(coeffs{2}, coeffs, x, y);
u_grad = @(x,y) [u_dx(x,y), u_dy(x,y)];

u_dxx = @(x,y) eval_second_deriv(coeffs{1}, coeffs, x, y);
u_dyy = @(x,y) eval_second_deriv(coeffs{2}, coeffs, x, y);
f = @(x,y) -u_dxx(x,y) - u_dyy(x,y);

g = @(x,y,idx) zeros(size(x));
c = @(x,y) ones(size(x));

%% Define discretization
nsub  = 8;
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
uh = zeros(space.ndof, 1);
[uh_drchlt, drchlt_dofs] = sp_drchlt_l2_proj(space, msh, g, bnd_gamma_idx);
uh(drchlt_dofs) = uh_drchlt;
int_dofs = setdiff(1:space.ndof, drchlt_dofs);

% Solve the linear system
rhs(int_dofs) = rhs(int_dofs) - stiff_mat(int_dofs, drchlt_dofs)*uh_drchlt;
uh(int_dofs) = stiff_mat(int_dofs, int_dofs) \ rhs(int_dofs);

%% Post processing
% Plot
sp_plot_solution(uh, space, geo, [10, 10])

% Error
err_l2 = sp_l2_error(space, msh, uh, u);
err_h1 = sp_h1_error(space, msh, uh, u, u_grad);
fprintf("The error in L2 is ||u - u_h||_L2 = %.10f \n", err_l2)
fprintf("The error in H1 is ||u - u_h||_H1 = %.10f \n", err_h1)
