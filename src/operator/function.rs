// todo: rename

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
use nalgebra::{Const, DVector, DefaultAllocator, DimMin, OMatrix, OVector, Point, RealField};
use std::iter::{zip, Product, Sum};

/// Assembles a discrete function (load vector).
pub fn assemble_function<'a, T, X, E, B, M, Q, const D: usize>(
    msh: &'a M,
    space: &Space<T, X, B, D>,
    quad: PullbackQuad<T, X, E, Q, D>,
    f: impl Fn(Point<T, D>) -> OVector<T, B::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D>,
          E: Cell<T, X, D, D>,
          M: Mesh<'a, T, X, D, D, Elem = B::Elem, GeoElem = E>,
          B: LocalBasis<T, X>,
          Q: Quadrature<T, X, E::RefCell>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumComponents, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<B::NumComponents>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
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
pub fn assemble_function_local<T, X, E, B, Q, const D: usize>(
    elem: &E,
    sp_local: &Space<T, X, B, D>,
    quad: &PullbackQuad<T, X, E, Q, D>,
    f: &impl Fn(Point<T, D>) -> OVector<T, B::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D>,
          E: Cell<T, X, D, D>,
          B: EvalBasis<T, X>,
          Q: Quadrature<T, X, E::RefCell>,
          DefaultAllocator: Allocator<B::NumComponents, B::NumBasis>,
          DefaultAllocator: Allocator<B::NumComponents>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    // Evaluate all basis functions at every quadrature point
    // and store them into a buffer
    let ref_elem = elem.ref_cell();
    let quad_nodes = quad.nodes_elem(elem).collect_vec();
    let buf: Vec<OMatrix<T, B::NumComponents, B::NumBasis>> = quad.nodes_ref(&ref_elem)
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