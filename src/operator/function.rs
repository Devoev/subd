// todo: rename

use crate::space::eval_basis::{EvalBasis, EvalBasisAllocator};
use crate::space::lin_combination::EvalFunctionAllocator;
use crate::space::local::MeshBasis;
use crate::space::Space;
use crate::cells::traits::ToElement;
use crate::element::traits::{HasBasisCoord, HasDim};
use crate::mesh::cell_topology::ElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::Mesh;
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricElem};
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DVector, DefaultAllocator, OMatrix, OVector, Point, RealField, ToTypenum};
use std::iter::{zip, Product, Sum};
use crate::diffgeo::chart::ChartAllocator;

/// Assembles a discrete function (load vector).
pub fn assemble_function<'a, T, Elem, Basis, Coords, Cells: 'a, Quadrature, const D: usize>(
    msh: &'a Mesh<T, Coords, Cells>,
    space: &Space<T, Basis>,
    quad: PullbackQuad<Quadrature>,
    f: impl Fn(Point<T, D>) -> OVector<T, Basis::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Elem: HasBasisCoord<T, Basis> + HasDim<T, Coords::GeoDim>,
          Basis: MeshBasis<T>,
          Basis::Cell: ToElement<T, Coords::GeoDim, Elem = Elem>, // todo: this is STILL required, because ElementTopology does not restrict Elem = Elem right now
          Coords: VertexStorage<T, GeoDim = Const<D>>,
          &'a Cells: ElementTopology<T, Coords, Cell = Basis::Cell>,
          Quadrature: QuadratureOnParametricElem<T, Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + Allocator<Coords::GeoDim> + ChartAllocator<T, Elem::GeoMap>,
          Const<D>: DimMinSelf + ToTypenum
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
pub fn assemble_function_local<T, Elem, Basis, Quadrature, const D: usize>(
    elem: &Elem,
    sp_local: &Space<T, Basis>,
    quad: &PullbackQuad<Quadrature>,
    f: &impl Fn(Point<T, D>) -> OVector<T, Basis::NumComponents>
) -> DVector<T>
    where T: RealField + Copy + Product<T> + Sum<T>,
          Elem: HasBasisCoord<T, Basis> + HasDim<T, Const<D>>,
          Basis: EvalBasis<T>,
          Quadrature: QuadratureOnParametricElem<T, Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis> + EvalFunctionAllocator<Basis>,
          Const<D>: DimMinSelf
{
    // Evaluate all basis functions at every quadrature point
    // and store them into a buffer
    let ref_elem = elem.parametric_element();
    let quad_nodes = quad.nodes_elem(elem).collect_vec();
    let buf: Vec<OMatrix<T, Basis::NumComponents, Basis::NumBasis>> = quad.nodes_ref::<T, Elem>(&ref_elem)
        .map(|p| sp_local.basis.eval(p)).collect();

    // Calculate pullback of product f * v
    let fv_pullback = |b: &OMatrix<T, Basis::NumComponents, Basis::NumBasis>, p: Point<T, D>, j: usize| {
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