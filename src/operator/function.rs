// todo: rename

use crate::element::traits::{ElemAllocator, ElemCoord, ElemDim, VolumeElement};
use crate::mesh::cell_topology::ElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{ElemOfMesh, Mesh};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricElem};
use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator};
use crate::space::lin_combination::EvalFunctionAllocator;
use crate::space::local::{MeshElemBasis};
use crate::space::Space;
use itertools::Itertools;
use nalgebra::{DVector, DefaultAllocator, OMatrix, OPoint, OVector, RealField};
use std::iter::{zip, Product, Sum};

/// Assembles a discrete function (load vector).
pub fn assemble_function<'a, T, Basis, Verts, Cells: 'a, Quadrature>(
    msh: &'a Mesh<T, Verts, Cells>,
    space: &Space<T, Basis>,
    quad: PullbackQuad<Quadrature>,
    f: impl Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          &'a Cells: ElementTopology<T, Verts>,
          ElemOfMesh<T, Verts, &'a Cells>: VolumeElement<T>, // todo: move this requirement to a sub-trait VolumeElementTopology
          Basis: MeshElemBasis<T, Verts, &'a Cells>,
          Quadrature: QuadratureOnParametricElem<T, ElemOfMesh<T, Verts, &'a Cells>>, // todo: possibly introduce MeshQuadrature trait
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + ElemAllocator<T, ElemOfMesh<T, Verts, &'a Cells>>
{
    // Create empty matrix
    let mut fi = DVector::<T>::zeros(space.dim());

    // Iteration over all mesh elements
    for (elem, cell) in msh.elem_cell_iter() {
        // Build local space and local stiffness matrix
        let (sp_local, idx) = space.local_space_with_idx(&cell);
        let fi_local = assemble_function_local(&elem, &sp_local, &quad, &f);

        // Fill global stiffness matrix with local entries
        let idx_local_global = idx.enumerate();
        for (i_local, i) in idx_local_global {
            fi[i] += fi_local[i_local];
        }
    }

    fi
}

/// Assembles a local discrete function vector.
pub fn assemble_function_local<T, Elem, Basis, Quadrature>(
    elem: &Elem,
    sp_local: &Space<T, Basis>,
    quad: &PullbackQuad<Quadrature>,
    f: &impl Fn(OPoint<T, ElemDim<T, Elem>>) -> OVector<T, Basis::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Elem: VolumeElement<T>,
          Basis: EvalBasis<T, Coord<T> = ElemCoord<T, Elem>>,
          Quadrature: QuadratureOnParametricElem<T, Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis> + EvalFunctionAllocator<Basis> + ElemAllocator<T, Elem>
{
    // Evaluate all basis functions at every quadrature point
    // and store them into a buffer
    let ref_elem = elem.parametric_element();
    let quad_nodes = quad.nodes_elem(elem).collect_vec();
    let buf: Vec<OMatrix<T, Basis::NumComponents, Basis::NumBasis>> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product f * v
    let fv_pullback = |b: &OMatrix<T, Basis::NumComponents, Basis::NumBasis>, p: OPoint<T, ElemDim<T, Elem>>, j: usize| {
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