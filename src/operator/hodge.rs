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

/// The weak discrete Hodge operator
/// ```text
/// M[i,j] = ∫ b[i] · b[j] dx ,
/// ```
/// where the `b[i]` are the basis functions.
pub struct Hodge<'a, T, X, M, B, const D: usize> {
    /// Mesh defining the geometry discretization.
    msh: &'a M,

    /// Space of discrete basis functions.
    space: &'a Space<T, X, B, D>
}

impl <'a, T, X, M, B, const D: usize> Hodge<'a, T, X, M, B, D> {
    /// Constructs a new `Hodge` operator from the given `msh` and `space`,
    pub fn new(msh: &'a M, space: &'a Space<T, X, B, D>) -> Self {
        Hodge { msh, space }
    }

    /// Assembles the discrete Hodge operator (*mass matrix*)
    /// using the given quadrature rule `quad`.
    pub fn assemble<E, Q>(&self, quad: PullbackQuad<T, X, E, Q, D>) -> CooMatrix<T>
        where T: RealField + Copy + Product<T> + Sum<T>,
              X: Dimensioned<T, D>,
              E: Cell<T, X, D, D>,
              M: Mesh<'a, T, X, D, D, Elem = B::Elem, GeoElem = E>,
              B: LocalBasis<T, X>,
              Q: Quadrature<T, X, E::RefCell>,
              DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
              Const<D>: DimMin<Const<D>, Output = Const<D>>
    {
        // Create empty matrix
        let mut mij = CooMatrix::<T>::zeros(self.space.dim(), self.space.dim());

        // Iteration over all mesh elements
        for elem in self.msh.elem_iter() {
            // Build local space and local mass matrix
            let (sp_local, idx) = self.space.local_space_with_idx(&elem);
            let geo_elem = self.msh.geo_elem(elem);
            let mij_local = assemble_hodge_local(&geo_elem, &sp_local, &quad);

            // Fill global mass matrix with local entries
            let idx_local_global = idx.enumerate();
            for ((i_local, i), (j_local, j)) in idx_local_global.clone().cartesian_product(idx_local_global) {
                mij.push(i, j, mij_local[(i_local, j_local)]);
            }
        }

        mij
    }
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