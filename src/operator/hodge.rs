use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::cells::bezier_elem::BezierElem;
use crate::cells::geo::Cell;
use crate::cells::hyper_rectangle::HyperRectangle;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::bezier::BezierMesh;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::{Const, DMatrix, RealField, RowDVector};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};
// todo: this function is only a temporary implementation.
//  make this generic over the space and mesh type!

/// Assembles the discrete hodge operator (mass matrix).
pub fn assemble_hodge<'a, T: RealField + Copy + Product<T> + Sum<T>, const D: usize>(
    msh: &'a BezierMesh<'a, T, D, D>,
    space: &impl LocalBasis<T, [T; D], 1, Elem = HyperRectangle<T, D>>,
    quad: impl Quadrature<T, D, Elem=BezierElem<'a, T, D, D>>,
) -> CooMatrix<T> {
    let mut mij = CooMatrix::<T>::zeros(space.num_basis(), space.num_basis());

    for elem in msh.elems() {
        let sp_local = space.elem_basis(&elem.ref_elem);
        let mij_local = assemble_hodge_local(&elem, &sp_local, &quad);
        let indices = space.global_indices(&sp_local).enumerate();
        for ((i_local, i), (j_local, j)) in indices.clone().cartesian_product(indices) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    mij
}

/// Assembles the local discrete Hodge operator.
pub fn assemble_hodge_local<T, X, E, const D: usize>(
    elem: &E,
    sp_local: &impl Basis<T, [T; D], 1>,
    quad: &impl Quadrature<T, D, Elem=E>,
) -> DMatrix<T> 
    where T: RealField + Copy + Product<T> + Sum<T>, 
          E: Cell<T, X, Const<D>, D>
{
    // Evaluate basis at each quadrature point and store in buffer
    let nodes = quad.nodes_elem(elem);
    let buf: Vec<RowDVector<T>> = nodes.map(|p| sp_local.eval(p.into_arr())).collect();

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
            quad.integrate_elem(elem, integrand)
        });
    
    // Assemble matrix
    DMatrix::from_iterator(num_basis, num_basis, mij)
}