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
num_patches = 5;
c = [
    0, 10, 1, 2;
    0, 2, 3, 4;
    0, 4, 5, 6;
    0, 6, 7, 8;
    0, 8, 9, 10;
];

hold on
for i = 1:num_patches
    idx = c(i,:) + 1;
    patches{i} = nrb4surf(coords(idx(1),:), coords(idx(2),:), coords(idx(4),:), coords(idx(3),:));
    nrbplot(patches{i}, [10, 10]);
end

geo = mp_geo_load(patches);
