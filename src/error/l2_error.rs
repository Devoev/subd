use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::cells::geo::Cell;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::Quadrature;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimMin, DimName, Dyn, OVector, Point, RealField};
use std::iter::{zip, Product, Sum};

/// L2-norm on a mesh.
pub struct L2Norm<'a, M> {
    /// Mesh defining the geometry discretization.
    msh: &'a M,
}

impl<'a, M> L2Norm<'a, M> {
    /// Constructs a new [`L2Norm`] from the given `msh` and `space`,
    pub fn new(msh: &'a M) -> Self {
        L2Norm { msh }
    }

    /// Calculates the L2 norm between the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<T, X, N: DimName, const D: usize, U, Q>(&self, u: U, quad: PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, X, D, D>,
          M::GeoElem: Cell<T, X, D, D>,
          U: Fn(Point<T, D>) -> OVector<T, N>,
          Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
          DefaultAllocator: Allocator<N>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(elem);

                // Evaluate function at quadrature nodes of element
                let u = quad.nodes_elem(&geo_elem).map(|p| u(p));

                // Calculate L2 error on element
                let u_norm_squared = u.map(|u| u.norm_squared());
                quad.integrate_elem(&geo_elem, u_norm_squared)
            })
            .sum::<T>()
            .sqrt()
    }

    /// Calculates the L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error<T, X, B, const D: usize, U, Q>(&self, uh: &LinCombination<T, X, B, D>, u: &U, quad: PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, X, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, X, D, D>,
          B: LocalBasis<T, X>,
          B::Elem: Clone, // todo: this can be removed, if msh.geo_elem would take a reference
          B::ElemBasis: Basis<NumBasis=Dyn>,
          U: Fn(Point<T, D>) -> OVector<T, B::NumComponents>,
          Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
          DefaultAllocator: Allocator<B::NumComponents>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(elem.clone());
                let ref_elem = geo_elem.ref_cell();

                // Evaluate functions at quadrature nodes of element
                let uh = quad.nodes_ref(&ref_elem).map(|x| uh.eval_on_elem(&elem, x));
                let u = quad.nodes_elem(&geo_elem).map(|p| u(p));

                // Calculate L2 error on element
                let du_norm_squared = zip(uh, u).map(|(uh, u)| (uh - u).norm_squared());
                quad.integrate_elem(&geo_elem, du_norm_squared)
            })
            .sum::<T>()
            .sqrt()
    }
}