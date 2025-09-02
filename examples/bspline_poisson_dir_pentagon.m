clear
pkg load nurbs
pkg load geopdes

%% Make geometry
r = 1;
n = 5;

% Angle between segments
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
knots = [0 0 1 1];
patch_1 = nrbmak([coords(1), coords(11), coords(2), coords(3)], {knots, knots});
