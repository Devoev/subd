use crate::basis::eval::EvalBasisAllocator;
use crate::basis::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::basis::local::MeshBasis;
use crate::cells::geo::{Cell, HasBasisCoord, HasDim};
use crate::mesh::traits::{Mesh, MeshTopology, VertexStorage};
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricCell};
use nalgebra::{Const, DefaultAllocator, OVector, Point, RealField, SVector, ToTypenum};
use std::iter::{zip, Product, Sum};
use nalgebra::allocator::Allocator;
use num_traits::real::Real;
use crate::cells::traits::ToElement;

/// L2-norm on a mesh.
pub struct L2Norm<'a, T, Coords, Cells> {
    /// Mesh defining the geometry discretization.
    pub(crate) msh: &'a Mesh<T, Coords, Cells>,
}

impl<'a, T, Coords, Cells> L2Norm<'a, T, Coords, Cells> {
    /// Constructs a new [`L2Norm`] on the given `msh`. 
    pub fn new(msh: &'a Mesh<T, Coords, Cells>) -> Self {
        L2Norm { msh }
    }

    /// Calculates the squared L2 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<const N: usize, const D: usize, U, Quadrature>(&self, u: U, quad: &PullbackQuad<Quadrature, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Coords: VertexStorage<T>,
          Cells: MeshTopology,
          Cells::Cell: ToElement<T, Coords::GeoDim>,
          <Cells::Cell as ToElement<T, Coords::GeoDim>>::Elem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> SVector<T, N>,
          Quadrature: QuadratureOnParametricCell<T, <Cells::Cell as ToElement<T, Coords::GeoDim>>::Elem>,
          DefaultAllocator: Allocator<Coords::GeoDim>,
          Const<D>: DimMinSelf
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(&elem);

                // Evaluate function at quadrature nodes of element
                let u = quad.nodes_elem(&geo_elem).map(&u);

                // Calculate L2 error on element
                let u_norm_squared = u.map(|u| u.norm_squared());
                quad.integrate_elem(&geo_elem, u_norm_squared)
            })
            .sum::<T>()
    }

    /// Calculates the L2 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<const N: usize, const D: usize, U, Quadrature>(&self, u: U, quad: &PullbackQuad<Quadrature, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Coords: VertexStorage<T>,
          Cells: MeshTopology,
          Cells::Cell: ToElement<T, Coords::GeoDim>,
          <Cells::Cell as ToElement<T, Coords::GeoDim>>::Elem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> SVector<T, N>,
          Quadrature: QuadratureOnParametricCell<T, <Cells::Cell as ToElement<T, Coords::GeoDim>>::Elem>,
          DefaultAllocator: Allocator<Coords::GeoDim>,
          Const<D>: DimMinSelf
    {
        self.norm_squared(u, quad).sqrt()
    }

    /// Calculates the squared L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error_squared<Basis, const D: usize, U, Quadrature>(&self, uh: &LinCombination<T, Basis, D>, u: &U, quad: &PullbackQuad<Quadrature, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Coords: VertexStorage<T>,
          Cells: MeshTopology<Cell= Basis::Cell>,
          Basis: MeshBasis<T>,
          Basis::Cell: ToElement<T, Coords::GeoDim>,
          <Basis::Cell as ToElement<T, Coords::GeoDim>>::Elem: HasBasisCoord<T, Basis> + HasDim<T, D>,
          U: Fn(Point<T, D>) -> OVector<T, Basis::NumComponents>,
          Quadrature: QuadratureOnParametricCell<T, <Basis::Cell as ToElement<T, Coords::GeoDim>>::Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis> + Allocator<Coords::GeoDim>,
          Const<D>: DimMinSelf
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(&elem);
                let ref_elem = geo_elem.ref_cell();

                // Evaluate functions at quadrature nodes of element
                let uh = quad.nodes_ref::<T, <Basis::Cell as ToElement<T, Coords::GeoDim>>::Elem>(&ref_elem).map(|x| uh.eval_on_elem(&elem, x));
                let u = quad.nodes_elem(&geo_elem).map(u);

                // Calculate L2 error on element
                let du_norm_squared = zip(uh, u).map(|(uh, u)| (uh - u).norm_squared());
                quad.integrate_elem(&geo_elem, du_norm_squared)
            })
            .sum::<T>()
    }

    /// Calculates the L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error<Basis, const D: usize, U, Quadrature>(&self, uh: &LinCombination<T, Basis, D>, u: &U, quad: &PullbackQuad<Quadrature, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Coords: VertexStorage<T>,
          Cells: MeshTopology<Cell= Basis::Cell>,
          Basis: MeshBasis<T>,
          Basis::Cell: ToElement<T, Coords::GeoDim>,
          <Basis::Cell as ToElement<T, Coords::GeoDim>>::Elem: HasBasisCoord<T, Basis> + HasDim<T, D>,
          U: Fn(Point<T, D>) -> OVector<T, Basis::NumComponents>,
          Quadrature: QuadratureOnParametricCell<T, <Basis::Cell as ToElement<T, Coords::GeoDim>>::Elem>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis> + Allocator<Coords::GeoDim>,
          Const<D>: DimMinSelf + ToTypenum
    {
        self.error_squared(uh, u, quad).sqrt()
    }
}