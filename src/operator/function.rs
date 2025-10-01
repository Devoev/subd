// todo: rename

use crate::basis::eval::{EvalBasis, EvalBasisAllocator};
use crate::basis::lin_combination::EvalFunctionAllocator;
use crate::basis::local::LocalBasis;
use crate::basis::space::Space;
use crate::mesh::traits::{Mesh, MeshTopology, VertexStorage};
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricCell};
use itertools::Itertools;
use nalgebra::{Const, DVector, DefaultAllocator, OMatrix, OVector, Point, RealField, ToTypenum};
use std::iter::{zip, Product, Sum};
use nalgebra::allocator::Allocator;
use crate::element::traits::{HasBasisCoord, HasDim};

/// Assembles a discrete function (load vector).
pub fn assemble_function<'a, T, E, Basis, Coords, Cells, Quadrature, const D: usize>(
    msh: &Mesh<T, Coords, Cells>,
    space: &Space<T, Basis, D>,
    quad: PullbackQuad<Quadrature, D>,
    f: impl Fn(Point<T, D>) -> OVector<T, Basis::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          E: HasBasisCoord<T, Basis> + HasDim<T, D>,
          Basis: LocalBasis<T>,
          Coords: VertexStorage<T>, 
          Cells: MeshTopology<Cell= Basis::Elem>,
          Quadrature: QuadratureOnParametricCell<T, E>,
          DefaultAllocator: EvalBasisAllocator<Basis::ElemBasis> + EvalFunctionAllocator<Basis> + Allocator<Coords::GeoDim>,
          Const<D>: DimMinSelf + ToTypenum
{
    // Create empty matrix
    let mut fi = DVector::<T>::zeros(space.dim());

    // Iteration over all mesh elements
    for elem in msh.elem_iter() {
        // Build local space and local stiffness matrix
        let (sp_local, idx) = space.local_space_with_idx(&elem);
        let geo_elem = msh.geo_elem(&elem);
        let fi_local = assemble_function_local(&geo_elem, &sp_local, &quad, &f);

        // Fill global stiffness matrix with local entries
        let idx_local_global = idx.enumerate();
        for (i_local, i) in idx_local_global {
            fi[i] += fi_local[i_local];
        }
    }

    fi
}

/// Assembles a local discrete function vector.
pub fn assemble_function_local<T, E, B, Q, const D: usize>(
    elem: &E,
    sp_local: &Space<T, B, D>,
    quad: &PullbackQuad<Q, D>,
    f: &impl Fn(Point<T, D>) -> OVector<T, B::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          E: HasBasisCoord<T, B> + HasDim<T, D>,
          B: EvalBasis<T>,
          Q: QuadratureOnParametricCell<T, E>,
          DefaultAllocator: EvalBasisAllocator<B> + EvalFunctionAllocator<B>,
          Const<D>: DimMinSelf
{
    // Evaluate all basis functions at every quadrature point
    // and store them into a buffer
    let ref_elem = elem.parametric_element();
    let quad_nodes = quad.nodes_elem(elem).collect_vec();
    let buf: Vec<OMatrix<T, B::NumComponents, B::NumBasis>> = quad.nodes_ref::<T, E>(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product f * v
    let fv_pullback = |b: &OMatrix<T, B::NumComponents, B::NumBasis>, p: Point<T, D>, j: usize| {
        b.column(j).dot(&f(p))
    };

    // Integrate over all combinations of grad_b[i] * grad_b[j] and integrate
    let num_basis = sp_local.dim();
    let fi = (0..num_basis).map(|j| {
            let integrand = zip(&buf, &quad_nodes)
                .map(|(b, &p)| fv_pullback(b, p, j));

            quad.integrate_elem(elem, integrand)
        });

    // Assemble matrix
    DVector::from_iterator(num_basis, fi)
}