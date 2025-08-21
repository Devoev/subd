use crate::basis::eval::{EvalGrad, EvalGradAllocator};
use crate::basis::local::LocalBasis;
use crate::basis::space::Space;
use crate::cells::geo::Cell;
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::{Const, DMatrix, DefaultAllocator, DimMin, OMatrix, RealField, SMatrix, U1};
use nalgebra_sparse::CooMatrix;
use std::iter::{zip, Product, Sum};

/// The weak discrete Laplace operator
/// ```text
/// K[i,j] = ∫ grad b[i] · grad b[j] dx ,
/// ```
/// where the `b[i]` are nodal basis functions.
pub struct Laplace<'a, T, M, B, const D: usize> {
    /// Mesh defining the geometry discretization.
    msh: &'a M,

    /// Space of discrete basis functions.
    space: &'a Space<T, B, D>
}


impl <'a, T, M, B, const D: usize> Laplace<'a, T, M, B, D> {
    /// Constructs a new `Laplace` operator from the given `msh` and `space`,
    pub fn new(msh: &'a M, space: &'a Space<T, B, D>) -> Self {
        Laplace { msh, space }
    }

    /// Assembles the discrete Laplace operator (*stiffness matrix*)
    /// using the given quadrature rule `quad`.
    pub fn assemble<E, Q>(&self, quad: PullbackQuad<T, B::Coord<T>, E, Q, D>) -> CooMatrix<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          B::Coord<T>: Dimensioned<T, D>,
          E: Cell<T, D, D>,
          E::GeoMap: Chart<T, D, D, Coord = B::Coord<T>>,
          M: Mesh<'a, T, D, D, Elem = B::Elem, GeoElem = E>,
          B: LocalBasis<T, NumComponents = U1>,
          B::ElemBasis: EvalGrad<T, D>,
          Q: Quadrature<T, B::Coord<T>, E::RefCell>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>,
    {
        // Create empty matrix
        let mut kij = CooMatrix::<T>::zeros(self.space.dim(), self.space.dim());

        // Iteration over all mesh elements
        for elem in self.msh.elem_iter() {
            // Build local space and local stiffness matrix
            let (sp_local, idx) = self.space.local_space_with_idx(&elem);
            let geo_elem = self.msh.geo_elem(&elem);
            let kij_local = assemble_laplace_local(&geo_elem, &sp_local, &quad);

            // Fill global stiffness matrix with local entries
            let idx_local_global = idx.enumerate();
            for ((i_local, i), (j_local, j)) in idx_local_global.clone().cartesian_product(idx_local_global) {
                kij.push(i, j, kij_local[(i_local, j_local)]);
            }
        }

        kij
    }
}

/// Assembles the local discrete Laplace operator.
pub fn assemble_laplace_local<T, E, B, Q, const D: usize>(
    elem: &E,
    sp_local: &Space<T, B, D>,
    quad: &PullbackQuad<T, B::Coord<T>, E, Q, D>,
) -> DMatrix<T>
where T: RealField + Copy + Product<T> + Sum<T>,
      B::Coord<T>: Dimensioned<T, D>,
      E: Cell<T, D, D>,
      E::GeoMap: Chart<T, D, D, Coord = B::Coord<T>>,
      B: EvalGrad<T, D>,
      Q: Quadrature<T, B::Coord<T>, E::RefCell>,
      DefaultAllocator: EvalGradAllocator<B, D>,
      Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    // Evaluate all basis functions and inverse gram matrices at every quadrature point
    // and store them into buffers
    let ref_elem = elem.ref_cell();
    let geo_map = elem.geo_map();
    let buf_grads: Vec<OMatrix<T, Const<D>, B::NumBasis>> = quad.nodes_ref(&ref_elem)
        .map(|p| sp_local.basis.eval_grad(p)).collect();
    let buf_g_inv: Vec<SMatrix<T, D, D>> = quad.nodes_ref(&ref_elem)
        .map(|p| {
            let j = geo_map.eval_diff(p);
            (j.transpose() * j).try_inverse().unwrap()
        }).collect();

    // Calculate pullback of product grad_u * grad_v
    let gradu_gradv_pullback = |grad_b: &OMatrix<T, Const<D>, B::NumBasis>, g_inv: &SMatrix<T, D, D>, i: usize, j: usize| {
        // Get gradients
        let grad_bi = grad_b.column(i);
        let grad_bj = grad_b.column(j);

        // Calculate integrand
        (grad_bi.transpose() * g_inv * grad_bj).x
    };

    // Integrate over all combinations of grad_b[i] * grad_b[j] and integrate
    let num_basis = sp_local.dim();
    let kij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            let integrand = zip(&buf_grads, &buf_g_inv)
                .map(|(b_grad, g_inv)| gradu_gradv_pullback(b_grad, g_inv, i, j));

            quad.integrate_elem(elem, integrand)
        });

    // Assemble matrix
    DMatrix::from_iterator(num_basis, num_basis, kij)
}