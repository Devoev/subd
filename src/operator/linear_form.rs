use std::borrow::Borrow;
use crate::element::traits::{ElemAllocator, ElemGeoDim, VolumeElement};
use crate::mesh::cell_topology::VolumetricElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{Mesh, MeshAllocator};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::{Quadrature, QuadratureOnMesh, QuadratureOnParametricElem};
use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator};
use crate::space::lin_combination::EvalFunctionAllocator;
use crate::space::local::{ElemBasis, MeshElemBasis};
use crate::space::Space;
use itertools::Itertools;
use nalgebra::{DVector, DefaultAllocator, OMatrix, OPoint, OVector, RealField};
use std::iter::{zip, Product, Sum};

/// The weak discrete version of a function or 3-form, i.e. the linear form
/// ```text
/// L[j] = ∫ f · b[j] dx ,
/// ```
/// where the `b[j]` are the basis functions.
pub struct LinearForm<'a, T, Verts, Cells, Basis, F> {
    /// Mesh defining the geometry discretization.
    msh: &'a Mesh<T, Verts, Cells>,

    /// Space of discrete basis functions.
    space: &'a Space<T, Basis>,

    /// Function defining the linear form.
    f: F // todo: replace generic F with dedicated function type (introduce super-trait also for LinearCombination)
}

impl <'a, T, Verts, Cells, Basis, F> LinearForm<'a, T, Verts, Cells, Basis, F> {
    /// Constructs a new `LinearForm` of the function `f` from the given `msh` and `space`.
    pub fn new(msh: &'a Mesh<T, Verts, Cells>, space: &'a Space<T, Basis>, f: F) -> Self {
        LinearForm { msh, space, f }
    }

    /// Assembles the discrete linear form (*load vector*) using the given quadrature rule `quad`.
    pub fn assemble<Quadrature>(&self, quad: &PullbackQuad<Quadrature>) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, Cells>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          F: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + MeshAllocator<T, Verts, Cells>
    {
        // Create empty vector
        let mut fi = DVector::<T>::zeros(self.space.dim());

        // Iteration over all mesh elements
        for (elem, cell) in self.msh.elem_cell_iter() {
            // Build local space and local stiffness matrix
            let (sp_local, idx) = self.space.local_space_with_idx(cell.borrow());
            let fi_local = assemble_linear_form_local(&elem, &sp_local, quad, &self.f);

            // Fill global stiffness matrix with local entries
            let idx_local_global = idx.enumerate();
            for (i_local, i) in idx_local_global {
                fi[i] += fi_local[i_local];
            }
        }

        fi
    }
}

/// Assembles a local discrete linear form.
pub fn assemble_linear_form_local<T, Elem, Basis, F, Quadrature>(
    elem: &Elem,
    sp_local: &Space<T, Basis>,
    quad: &PullbackQuad<Quadrature>,
    f: &F
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Elem: VolumeElement<T>,
          Basis: EvalBasis<T> + ElemBasis<T, Elem>,
          Quadrature: QuadratureOnParametricElem<T, Elem>,
          F: Fn(OPoint<T, ElemGeoDim<T, Elem>>) -> OVector<T, Basis::NumComponents>,
          DefaultAllocator: EvalBasisAllocator<Basis> + EvalFunctionAllocator<Basis> + ElemAllocator<T, Elem>
{
    // Evaluate all basis functions at every quadrature point
    // and store them into a buffer
    let ref_elem = elem.parametric_element();
    let quad_nodes = quad.nodes_elem(elem).collect_vec();
    let buf: Vec<OMatrix<T, Basis::NumComponents, Basis::NumBasis>> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product f * v
    let fv_pullback = |b: &OMatrix<T, Basis::NumComponents, Basis::NumBasis>, p: OPoint<T, ElemGeoDim<T, Elem>>, j: usize| {
        b.column(j).dot(&f(p))
    };

    // Integrate over all combinations of grad_b[i] * grad_b[j] and integrate
    let num_basis = sp_local.dim();
    let fi = (0..num_basis).map(|j| {
            let integrand = zip(&buf, &quad_nodes)
                .map(|(b, p)| fv_pullback(b, p.clone(), j));

            quad.integrate_elem(elem, integrand)
        });

    // Assemble matrix
    DVector::from_iterator(num_basis, fi)
}