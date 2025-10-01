use crate::diffgeo::chart::Chart;
use crate::basis::eval::EvalBasisAllocator;
use crate::basis::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::basis::local::MeshBasis;
use crate::cells::geo::{Cell, HasBasisCoord, HasDim};
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::{Quadrature, QuadratureOnParametricCell};
use nalgebra::{Const, DefaultAllocator, OVector, Point, RealField, SVector};
use std::iter::{zip, Product, Sum};

/// L∞-norm on a mesh.
pub struct LInfNorm<'a, M> {
    /// Mesh defining the geometry discretization.
    pub(crate) msh: &'a M,
}

impl<'a, M> LInfNorm<'a, M> {
    /// Constructs a new [`LInfNorm`] on the given `msh`.
    pub fn new(msh: &'a M) -> Self {
        LInfNorm { msh }
    }

    /// Estimates the L∞ norm of the given exact solution `u`
    /// by evaluating the function at the mesh nodes.
    pub fn norm_est<T, const N: usize, const D: usize, U>(&self, u: U) -> T
    where T: RealField,
          M: Mesh<'a, T, D, D>,
          M::GeoElem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> SVector<T, N>,
          Const<D>: DimMinSelf
    {
        // Iterate over mesh vertices and calculate norm
        self.msh.vertex_iter()
            .map(|p| u(p).norm())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(T::zero())
    }

    /// Estimates the L∞ error between the given discrete solution `uh` and the exact one `u`
    /// by evaluating the functions at the element centroids.
    pub fn error_est<T, B, const D: usize, U>(&self, uh: &LinCombination<T, B, D>, u: &U) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D, Elem = B::Cell>,
          M::GeoElem: HasDim<T, D> + HasBasisCoord<T, B>,
          B: MeshBasis<T>,
          B::Coord<T>: From<[T; D]> + Clone,
          U: Fn(Point<T, D>) -> OVector<T, B::NumComponents>,
          DefaultAllocator: EvalBasisAllocator<B::LocalBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::LocalBasis>,
    {
        // Iterate over element centroids and calculate norm
        self.msh.elem_iter()
            .map(|elem| {
                // Get mapping
                let phi = self.msh.geo_elem(&elem).geo_map();

                // Evaluate functions at (0.5,0.5)
                let x: B::Coord<T> = [T::from_f64(0.5).unwrap(); D].into();
                let p = phi.eval(x.clone());

                // Calculate |u - uh|
                (u(p) - uh.eval_on_elem(&elem, x)).norm()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(T::zero())
    }
}