use crate::basis::eval::EvalBasisAllocator;
use crate::basis::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::basis::local::LocalBasis;
use crate::cells::geo::{Cell, HasBasisCoord, HasDim};
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricCell};
use nalgebra::{Const, DefaultAllocator, OVector, Point, RealField, SVector};
use std::iter::{zip, Product, Sum};

/// L2-norm on a mesh.
pub struct L2Norm<'a, M> {
    /// Mesh defining the geometry discretization.
    msh: &'a M,
}

impl<'a, M> L2Norm<'a, M> {
    /// Constructs a new [`L2Norm`] on the given `msh`. 
    pub fn new(msh: &'a M) -> Self {
        L2Norm { msh }
    }

    /// Calculates the squared L2 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<T, const N: usize, const D: usize, U, Q>(&self, u: U, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D>,
          M::GeoElem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> SVector<T, N>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
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
    pub fn norm<T, const N: usize, const D: usize, U, Q>(&self, u: U, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D>,
          M::GeoElem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> SVector<T, N>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          Const<D>: DimMinSelf
    {
        self.norm_squared(u, quad).sqrt()
    }

    /// Calculates the squared L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error_squared<T, B, const D: usize, U, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D, Elem = B::Elem>,
          M::GeoElem: HasDim<T, D> + HasBasisCoord<T, B>,
          B: LocalBasis<T>,
          U: Fn(Point<T, D>) -> OVector<T, B::NumComponents>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          DefaultAllocator: EvalBasisAllocator<B::ElemBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::ElemBasis>,
          Const<D>: DimMinSelf
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(&elem);
                let ref_elem = geo_elem.ref_cell();

                // Evaluate functions at quadrature nodes of element
                let uh = quad.nodes_ref::<T, M::GeoElem>(&ref_elem).map(|x| uh.eval_on_elem(&elem, x));
                let u = quad.nodes_elem(&geo_elem).map(u);

                // Calculate L2 error on element
                let du_norm_squared = zip(uh, u).map(|(uh, u)| (uh - u).norm_squared());
                quad.integrate_elem(&geo_elem, du_norm_squared)
            })
            .sum::<T>()
    }

    /// Calculates the L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error<T, B, const D: usize, U, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D, Elem = B::Elem>,
          M::GeoElem: HasDim<T, D> + HasBasisCoord<T, B>,
          B: LocalBasis<T>,
          U: Fn(Point<T, D>) -> OVector<T, B::NumComponents>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          DefaultAllocator: EvalBasisAllocator<B::ElemBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::ElemBasis>,
          Const<D>: DimMinSelf
    {
        self.error_squared(uh, u, quad).sqrt()
    }
}