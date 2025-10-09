use std::borrow::Borrow;
use crate::element::traits::{ElemAllocator, VolumeElement};
use crate::mesh::cell_topology::VolumetricElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{Mesh, MeshAllocator};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::{Quadrature, QuadratureOnMesh, QuadratureOnParametricElem};
use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator};
use crate::space::local::{ElemBasis, MeshElemBasis};
use crate::space::Space;
use itertools::Itertools;
use nalgebra::{DMatrix, DefaultAllocator, OMatrix, RealField};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};

/// The weak discrete Hodge operator
/// ```text
/// M[i,j] = ∫ b[i] · b[j] dx ,
/// ```
/// where the `b[i]` are the basis functions.
pub struct Hodge<'a, T, Basis, Verts, Cells> {
    /// Mesh defining the geometry discretization.
    msh: &'a Mesh<T, Verts, Cells>,

    /// Space of discrete basis functions.
    space: &'a Space<T, Basis>
}

impl <'a, T, Basis, Verts, Cells> Hodge<'a, T, Basis, Verts, Cells> {
    /// Constructs a new `Hodge` operator from the given `msh` and `space`.
    pub fn new(msh: &'a Mesh<T, Verts, Cells>, space: &'a Space<T, Basis>) -> Self {
        Hodge { msh, space }
    }

    /// Assembles the discrete Hodge operator (*mass matrix*)
    /// using the given quadrature rule `quad`.
    pub fn assemble<Quadrature>(&self, quad: &PullbackQuad<Quadrature>) -> CooMatrix<T>
        where T: RealField + Copy + Product<T> + Sum<T>,
              Verts: VertexStorage<T>,
              Cells: VolumetricElementTopology<T, Verts>,
              Basis: MeshElemBasis<T, Verts, Cells>,
              Quadrature: QuadratureOnMesh<T, Verts, Cells>,
              DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + MeshAllocator<T, Verts, Cells>,
    {
        // Create empty matrix
        let mut mij = CooMatrix::<T>::zeros(self.space.dim(), self.space.dim());

        // Iteration over all mesh elements
        for (elem, cell) in self.msh.elem_cell_iter() {
            // Build local space and local mass matrix
            let (sp_local, idx) = self.space.local_space_with_idx(cell.borrow());
            let mij_local = assemble_hodge_local(&elem, &sp_local, quad);

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
pub fn assemble_hodge_local<T, Elem, Basis, Quadrature>(
    elem: &Elem,
    sp_local: &Space<T, Basis>,
    quad: &PullbackQuad<Quadrature>,
) -> DMatrix<T> 
    where T: RealField + Copy + Product<T> + Sum<T>,
          Elem: VolumeElement<T>,
          Basis: EvalBasis<T> + ElemBasis<T, Elem>,
          Quadrature: QuadratureOnParametricElem<T, Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis> + ElemAllocator<T, Elem>,
{
    // Evaluate all basis functions and store in 'buf'
    let ref_elem = elem.parametric_element();
    let buf: Vec<OMatrix<T, Basis::NumComponents, Basis::NumBasis>> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product uv
    let uv_pullback = |b: &OMatrix<T, Basis::NumComponents, Basis::NumBasis>, i: usize, j: usize| {
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