use crate::basis::local::LocalBasis;
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use crate::cells::bezier_elem::BezierElem;
use crate::cells::geo::Cell;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::bezier::BezierMesh;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrix, DefaultAllocator, OMatrix, RealField};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};
// todo: this function is only a temporary implementation.
//  make this generic over the space and mesh type!

/// Assembles the discrete hodge operator (mass matrix).
pub fn assemble_hodge<'a, T, B, const D: usize>(
    msh: &'a BezierMesh<'a, T, D, D>,
    space: &Space<T, [T; D], B>,
    quad: impl Quadrature<T, D, Node=[T; D], Elem=BezierElem<'a, T, D, D>>,
) -> CooMatrix<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          B: LocalBasis<T, [T; D]>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis<T, [T; D]>>::NumComponents, <B::ElemBasis as Basis<T, [T; D]>>::NumBasis>
{
    let mut mij = CooMatrix::<T>::zeros(space.dim(), space.dim());

    for elem in msh.elems() {
        // todo: the find_elem call should be removed. 
        //  This can be maybe done, if the bound on BezierMesh is loosened.
        //  Can the bezier elem be directly passed to elem_basis,
        //  or is a find_span call required?
        let span = space.basis.find_elem(elem.ref_elem.a.into_arr());
        let sp_local = space.basis.elem_basis(&span);
        let mij_local = assemble_hodge_local(&elem, &sp_local, &quad);
        let indices = space.basis.global_indices(&sp_local).enumerate();
        for ((i_local, i), (j_local, j)) in indices.clone().cartesian_product(indices) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    mij
}

/// Assembles the local discrete Hodge operator.
pub fn assemble_hodge_local<T, X, E, B, const D: usize>(
    elem: &E,
    sp_local: &B,
    quad: &impl Quadrature<T, D, Node=X, Elem=E>,
) -> DMatrix<T> 
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D>,
          E: Cell<T, X, D, D>,
          B: Basis<T, [T; D]>,
          DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>
{
    // Evaluate basis at each quadrature point and store in buffer
    let nodes = quad.nodes_elem(elem);
    let buf: Vec<OMatrix<T, B::NumComponents, B::NumBasis>> = nodes.map(|p| sp_local.eval(p.into_arr())).collect();

    // Calculate pullback of product uv
    let uv_pullback = |b: &OMatrix<T, B::NumComponents, B::NumBasis>, i: usize, j: usize| {
        // Eval basis
        let bi = b.column(i);
        let bj = b.column(j);

        // Calculate integrand
        bi.dot(&bj)
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