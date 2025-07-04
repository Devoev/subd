use crate::basis::eval::EvalBasis;
use crate::basis::local::LocalBasis;
use crate::basis::space::Space;
use crate::basis::traits::Basis;
use crate::cells::geo::Cell;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DMatrix, DefaultAllocator, DimMin, OMatrix, RealField};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};

/// Assembles the discrete hodge operator (mass matrix).
pub fn assemble_hodge<'a, T, X, E, B, M, Q, const D: usize>(
    msh: &'a M,
    space: &Space<T, X, B, D>,
    quad: PullbackQuad<T, X, E, Q, D>,
    elem_to_sp_elem: impl Fn(&M::Elem) -> B::Elem // todo: this function is just temporary
) -> CooMatrix<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D>,
          E: Cell<T, X, D, D>,
          M: Mesh<'a, T, X, D, D, GeoElem = E>,
          B: LocalBasis<T, X>, // todo: add Elem = E::RefCell
          Q: Quadrature<T, X, E::RefCell>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    // Create empty matrix
    let mut mij = CooMatrix::<T>::zeros(space.dim(), space.dim());

    // Iteration over all mesh elements
    for elem in msh.elem_iter() {
        let sp_elem = elem_to_sp_elem(&elem);
        let geo_elem = msh.geo_elem(elem);

        // Build local space and local mass matrix
        let (sp_local, idx) = space.local_space_with_idx(&sp_elem);
        let mij_local = assemble_hodge_local(&geo_elem, &sp_local, &quad);

        // Fill global mass matrix with local entries
        let idx_local_global = idx.enumerate();
        for ((i_local, i), (j_local, j)) in idx_local_global.clone().cartesian_product(idx_local_global) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    mij
}

/// Assembles the local discrete Hodge operator.
pub fn assemble_hodge_local<T, X, E, B, Q, const D: usize>(
    elem: &E,
    sp_local: &Space<T, X, B, D>,
    quad: &PullbackQuad<T, X, E, Q, D>,
) -> DMatrix<T> 
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D>,
          E: Cell<T, X, D, D>,
          B: EvalBasis<T, X>,
          Q: Quadrature<T, X, E::RefCell>,
          DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    // Evaluate all basis functions and store in 'buf'
    let ref_elem = elem.ref_cell();
    let buf: Vec<OMatrix<T, B::NumComponents, B::NumBasis>> = quad.nodes_ref(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product uv
    let uv_pullback = |b: &OMatrix<T, B::NumComponents, B::NumBasis>, i: usize, j: usize| {
        // Eval basis
        let bi = b.column(i);
        let bj = b.column(j);

        // Calculate integrand
        bi.dot(&bj)
    };
    
    // Integrate over all combinations of b[i] * b[j] and integrate
    let num_basis = sp_local.dim();
    let mij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            let integrand = buf.iter().map(|b| uv_pullback(b, i, j));
            quad.integrate_elem(elem, integrand)
        });
    
    // Assemble matrix
    DMatrix::from_iterator(num_basis, num_basis, mij)
}