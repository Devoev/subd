use std::borrow::Borrow;
use crate::diffgeo::chart::Chart;
use crate::element::traits::{ElemAllocator, VolumeElement};
use crate::mesh::cell_topology::VolumetricElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{Mesh, MeshAllocator};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::{Quadrature, QuadratureOnMesh, QuadratureOnParametricElem};
use crate::space::eval_basis::{EvalGrad, EvalGradAllocator};
use crate::space::local::{ElemBasis, MeshElemBasis, MeshGradBasis};
use crate::space::Space;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{DMatrix, DefaultAllocator, OMatrix, RealField, U1};
use nalgebra_sparse::CooMatrix;
use std::iter::{zip, Product, Sum};

/// The weak discrete Laplace operator
/// ```text
/// K[i,j] = ∫ grad b[i] · grad b[j] dx ,
/// ```
/// where the `b[i]` are nodal basis functions.
pub struct Laplace<'a, T, Basis, Verts, Cells> {
    /// Mesh defining the geometry discretization.
    msh: &'a Mesh<T, Verts, Cells>,

    /// Space of discrete basis functions.
    space: &'a Space<T, Basis>
}


impl <'a, T, Basis, Verts, Cells> Laplace<'a, T, Basis, Verts, Cells> {
    /// Constructs a new `Laplace` operator from the given `msh` and `space`.
    pub fn new(msh: &'a Mesh<T, Verts, Cells>, space: &'a Space<T, Basis>) -> Self {
        Laplace { msh, space }
    }

    /// Assembles the discrete Laplace operator (*stiffness matrix*)
    /// using the given quadrature rule `quad`.
    pub fn assemble<Quadrature>(&self, quad: &PullbackQuad<Quadrature>) -> CooMatrix<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, Cells> + MeshGradBasis<T>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          DefaultAllocator: MeshAllocator<T, Verts, Cells> + EvalGradAllocator<Basis::LocalBasis> + Allocator<U1, Basis::ParametricDim> // todo: last allocator is the same as below
    {
        // Create empty matrix
        let mut kij = CooMatrix::<T>::zeros(self.space.dim(), self.space.dim());

        // Iteration over all mesh elements
        for (elem, cell) in self.msh.elem_cell_iter() {
            // Build local space and local stiffness matrix
            let (sp_local, idx) = self.space.local_space_with_idx(cell.borrow());
            let kij_local = assemble_laplace_local(&elem, &sp_local, quad);

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
pub fn assemble_laplace_local<T, Elem, Basis, Quadrature>(
    elem: &Elem,
    sp_local: &Space<T, Basis>,
    quad: &PullbackQuad<Quadrature>,
) -> DMatrix<T>
where T: RealField + Copy + Product<T> + Sum<T>,
      Elem: VolumeElement<T>,
      Basis: EvalGrad<T> + ElemBasis<T, Elem>,
      Quadrature: QuadratureOnParametricElem<T, Elem>,
      DefaultAllocator: EvalGradAllocator<Basis> + ElemAllocator<T, Elem> + Allocator<U1, Basis::ParametricDim> // todo: the last allocator bound is for the single gradient evaluations. Possibly hide that behind trait?
{
    // Evaluate all basis functions and inverse gram matrices at every quadrature point
    // and store them into buffers
    let ref_elem = elem.parametric_element();
    let geo_map = elem.geo_map();
    let buf_grads: Vec<_> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| sp_local.basis.eval_grad(p)).collect();
    let buf_g_inv: Vec<_> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| {
            let j = geo_map.eval_diff(p);
            (j.transpose() * j).try_inverse().unwrap()
        }).collect();

    // Calculate pullback of product grad_u * grad_v
    let gradu_gradv_pullback = |grad_b: &OMatrix<T, Basis::ParametricDim, Basis::NumBasis>, g_inv: &OMatrix<T, Basis::ParametricDim, Basis::ParametricDim>, i: usize, j: usize| {
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