use iter_num_tools::lin_space;
use nalgebra::{matrix, Point2};
use num_traits::real::Real;
use subd::cells::geo::Cell;
use subd::cells::quad::QuadNodes;
use subd::diffgeo::chart::Chart;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::Mesh;
use subd::plot::plot_fn_msh;

fn main() {
    let coords = matrix![
        0.0, 0.0;
        0.19304569050536266, -0.1402559040003067;
        0.3205815769824112, -0.00000000000000004163336342344337;
        0.19304569050536266, 0.14025590400030666;
        0.09906515537108554, 0.3048911977932986;
        -0.07373689239135567, 0.2269388197953386;
        -0.2593559438622911, 0.18843312310692553;
        -0.238617596228014, 0.00000000000000003469446951953614;
        -0.25935594386229116, -0.18843312310692542;
        -0.07373689239135572, -0.22693881979533853;
        0.09906515537108546, -0.30489119779329865
    ];

    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    let mut quad_msh = QuadVertexMesh::from_matrix(coords, faces);
    quad_msh = quad_msh.lin_subd().unpack();

    let f = |p: Point2<f64>| {
        let arg = p.coords.norm_squared();
        -arg.exp() * arg.sin()
    };
    plot_fn_msh(&quad_msh, &|elem, uv| f(quad_msh.geo_elem(elem).geo_map().eval(uv)), 10, |elem, num| {
        (lin_space(0.0..=1.0, num).collect(), lin_space(0.0..=1.0, num).collect())
    }).show();
}