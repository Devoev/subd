use iter_num_tools::lin_space;
use itertools::Itertools;
use nalgebra::{matrix, RowDVector};
use subd::cells::quad::QuadNodes;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::plot::plot_fn_msh;
use subd::subd::catmull_clark::basis::CatmarkBasis;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::patch::CatmarkPatchNodes;
use subd::subd::catmull_clark::space::CatmarkSpace;

fn main() {
    // Define mesh
    let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut quad_msh = QuadVertexMesh::from_coords_matrix(coords_square, quads);

    // Refine and construct Catmark mesh
    quad_msh = quad_msh.catmark_subd().catmark_subd().catmark_subd().catmark_subd().unpack();
    let msh = CatmarkMesh::from(quad_msh.clone());

    // Define space
    let basis = CatmarkBasis(&msh);
    let space = CatmarkSpace::new(basis);

    // Plot boundary basis functions
    let bnd_basis = |elem: &CatmarkPatchNodes, x: (f64, f64)| -> f64 {
        // space.eval_on_elem(elem, x)[1]
        let mut global = RowDVector::zeros(space.dim());
        space.populate_global_on_elem(&mut global, elem, x);
        global[0]
    };

    plot_fn_msh(&msh, &bnd_basis, 10, |_, num| {
        (lin_space(0.0..=1.0, num).collect_vec(), lin_space(0.0..=1.0, num).collect_vec())
    }).show();
}