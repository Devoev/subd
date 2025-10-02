use crate::space::eval::{EvalBasis, EvalBasisAllocator};
use crate::space::local::MeshBasis;
use crate::space::Space;
use crate::mesh::cell_topology::{CellTopology};
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::Mesh;
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricElem};
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DMatrix, DefaultAllocator, OMatrix, RealField, ToTypenum};
use nalgebra_sparse::CooMatrix;
use std::iter::{Product, Sum};
use crate::element::traits::{HasBasisCoord, HasDim};

/// The weak discrete Hodge operator
/// ```text
/// M[i,j] = ∫ b[i] · b[j] dx ,
/// ```
/// where the `b[i]` are the basis functions.
pub struct Hodge<'a, T, Basis, Coords, Cells, const D: usize> {
    /// Mesh defining the geometry discretization.
    msh: &'a Mesh<T, Coords, Cells>,

    /// Space of discrete basis functions.
    space: &'a Space<T, Basis, D>
}

impl <'a, T, Basis, Coords, Cells, const D: usize> Hodge<'a, T, Basis, Coords, Cells, D> {
    /// Constructs a new `Hodge` operator from the given `msh` and `space`.
    pub fn new(msh: &'a Mesh<T, Coords, Cells>, space: &'a Space<T, Basis, D>) -> Self {
        Hodge { msh, space }
    }

    /// Assembles the discrete Hodge operator (*mass matrix*)
    /// using the given quadrature rule `quad`.
    pub fn assemble<Elem, Quadrature>(&self, quad: PullbackQuad<Quadrature, D>) -> CooMatrix<T>
        where T: RealField + Copy + Product<T> + Sum<T>,
              Elem: HasBasisCoord<T, Basis> + HasDim<T, D>,
              Coords: VertexStorage<T>,
              Cells: CellTopology<Cell= Basis::Cell>,
              Basis: MeshBasis<T>,
              Quadrature: QuadratureOnParametricElem<T, Elem>,
              DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + Allocator<Coords::GeoDim>,
              Const<D>: DimMinSelf + ToTypenum
    {
        // Create empty matrix
        let mut mij = CooMatrix::<T>::zeros(self.space.dim(), self.space.dim());

        // Iteration over all mesh elements
        for elem in self.msh.elem_iter() {
            // Build local space and local mass matrix
            let (sp_local, idx) = self.space.local_space_with_idx(&elem);
            let geo_elem = self.msh.geo_elem(&elem);
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
pub fn assemble_hodge_local<T, E, B, Q, const D: usize>(
    elem: &E,
    sp_local: &Space<T, B, D>,
    quad: &PullbackQuad<Q, D>,
) -> DMatrix<T> 
    where T: RealField + Copy + Product<T> + Sum<T>,
          E: HasBasisCoord<T, B> + HasDim<T, D>,
          B: EvalBasis<T>,
          Q: QuadratureOnParametricElem<T, E>,
          DefaultAllocator: EvalBasisAllocator<B>,
          Const<D>: DimMinSelf
{
    // Evaluate all basis functions and store in 'buf'
    let ref_elem = elem.ref_cell();
    let buf: Vec<OMatrix<T, B::NumComponents, B::NumBasis>> = quad.nodes_ref::<T, E>(&ref_elem)
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