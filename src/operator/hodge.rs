use crate::basis::local::LocalBasis;
use crate::bspline::local_basis::LocalBsplineBasis;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use crate::mesh::bezier::BezierMesh;
use crate::mesh::topo::Mesh;
use crate::quadrature::tensor_prod_gauss_legendre::TensorProdGaussLegendre;
use itertools::Itertools;
use nalgebra::{DMatrix, RealField, RowDVector};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};
// todo: this function is only a temporary implementation for just 1D bezier meshes.
//  make this generic over the space and mesh type!

/// Assembles the discrete hodge operator (mass matrix).
pub fn assemble_hodge<T: RealField + Copy + Product<T> + Sum<T>, const M: usize>(
    msh: &BezierMesh<T, 1, M>,
    knots: &KnotVec<T>,
    n: usize,
    p: usize,
    quad: TensorProdGaussLegendre<T>
) -> CooMatrix<T> {
    let topo = &msh.ref_mesh.topology;
    let mut mij = CooMatrix::<T>::zeros(topo.num_nodes(), topo.num_nodes());

    for elem in msh.elems() {
        let span = KnotSpan::find(knots, n, T::zero()).unwrap(); // todo
        let sp_local = LocalBsplineBasis::new(knots, p, span);
        let mij_local: DMatrix<T> = assemble_hodge_local(&sp_local, &quad);
        let indices = sp_local.global_indices().enumerate();
        for ((i_local, i), (j_local, j)) in indices.clone().cartesian_product(indices) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    mij
}

/// Assembles the local discrete Hodge operator.
pub fn assemble_hodge_local<T: RealField + Copy + Product<T> + Sum<T>>(
    sp_local: &LocalBsplineBasis<T>,
    quad: &TensorProdGaussLegendre<T>
) -> DMatrix<T> {
    // Evaluate basis at each quadrature point and store in buffer
    let nodes = quad.nodes::<1>();
    let buf: Vec<RowDVector<T>> = nodes.map(|n| {
        sp_local.eval(n[0])
    }).collect();
    
    // Calculate pullback of product uv
    let uv_pullback = |b: &RowDVector<T>, i: usize, j: usize| {
        // Eval basis
        let bi = b[i];
        let bj = b[j];

        // Calculate integrand
        bi * bj
    };
    
    // Integrate over all combinations of b[i] * b[j] and integrate
    let num_basis = sp_local.num_basis();
    let mij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            let integrand = buf.iter().map(|b| uv_pullback(b, i, j));
            quad.integrate::<1>(integrand)
        });
    
    // Assemble matrix
    DMatrix::from_iterator(num_basis, num_basis, mij)
}