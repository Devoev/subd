use iter_num_tools::lin_space;
use itertools::Itertools;
use nalgebra::{matrix, Point2};
use std::fs::File;
use subd::cells::geo::Cell;
use subd::cells::quad::QuadNodes;
use subd::diffgeo::chart::Chart;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::mesh::traits::Mesh;
use subd::plot::{plot_faces, plot_fn_msh, write_connectivity, write_coords_with_fn};
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::quadrature::traits::Quadrature;

fn main() {
    // Define mesh
    let coords = matrix![
        0.0, 0.0;
        0.3205815769824112, -0.00000000000000004163336342344337;
        0.19304569050536266, 0.14025590400030666;
        0.09906515537108554, 0.3048911977932986;
        -0.07373689239135567, 0.2269388197953386;
        -0.2593559438622911, 0.18843312310692553;
        -0.238617596228014, 0.00000000000000003469446951953614;
        -0.25935594386229116, -0.18843312310692542;
        -0.07373689239135572, -0.22693881979533853;
        0.09906515537108546, -0.30489119779329865;
        0.19304569050536266, -0.1402559040003067;
    ];

    let faces = vec![
        QuadNodes::from_indices(0, 10, 1, 2),
        QuadNodes::from_indices(0, 2, 3, 4),
        QuadNodes::from_indices(0, 4, 5, 6),
        QuadNodes::from_indices(0, 6, 7, 8),
        QuadNodes::from_indices(0, 8, 9, 10),
    ];
    let mut msh = QuadVertexMesh::from_matrix(coords, faces);
    msh = msh.lin_subd().lin_subd().unpack();
    plot_faces(&msh, msh.elems.iter().copied()).show();

    // Find quadrature points
    let quad = PullbackQuad::new(GaussLegendreBi::with_degrees(2, 2));
    let nodes = quad.nodes_elem(&msh.geo_elem(&&msh.elems[0])).collect_vec();
    dbg!(&nodes);

    // Define function
    let f = |p: Point2<f64>| {
        (p.coords / 5.0).norm().cos()
    };

    // Plot
    plot_fn_msh(&msh, &|elem, uv| f(msh.geo_elem(elem).geo_map().eval(uv)), 10, |elem, num| {
        (lin_space(0.0..=1.0, num).collect(), lin_space(0.0..=1.0, num).collect())
    }).show();

    // Write
    let z = msh.coords.iter().map(|&p| f(p));
    write_connectivity(msh.elems.iter().copied(), &mut File::create("examples/surf_conn.dat").unwrap()).unwrap();
    write_coords_with_fn(msh.coords.iter().copied(), z, &mut File::create("examples/surf.dat").unwrap()).unwrap();
    write_connectivity(msh.edges().filter(|edge| msh.is_boundary_node(edge.start()) && msh.is_boundary_node(edge.end())), &mut File::create("examples/surf_bnd_conn.dat").unwrap()).unwrap();
}