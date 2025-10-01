use crate::basis::eval::EvalGradAllocator;
use crate::basis::grad::{GradBasis, GradBasisPullback};
use crate::basis::lin_combination::{LinCombination, SelectCoeffsAllocator};
use crate::basis::local::MeshGradBasis;
use crate::cells::geo::{Cell, HasDim};
use crate::diffgeo::chart::Chart;
use crate::error::l2_error::L2Norm;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::{DimMinSelf, PullbackQuad};
use crate::quadrature::traits::QuadratureOnParametricCell;
use nalgebra::{Const, DefaultAllocator, OVector, Point, RealField, SVector, U1};
use std::iter::{Product, Sum};
use crate::basis::space::Space;

/// H1-norm on a mesh.
pub struct H1Norm<'a, M>(L2Norm<'a, M>);

impl<'a, M> H1Norm<'a, M> {
    /// Constructs a new [`H1Norm`] for the given `msh`.
    pub fn new(msh: &'a M) -> Self {
        H1Norm(L2Norm::new(msh))
    }

    /// Calculates the squared H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<T, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D>,
          M::GeoElem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          Const<D>: DimMinSelf
    {
        // Calculate ||u||^2 + ||grad u||^2
        self.0.norm_squared(u, quad) + self.0.norm_squared(u_grad, quad)
    }

    /// Calculates the H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<T, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D>,
          M::GeoElem: HasDim<T, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          Const<D>: DimMinSelf
    {
       self.norm_squared(u, u_grad, quad).sqrt()
    }

    /// Calculates the squared H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`, using the quadrature rule `quad`.
    pub fn error_squared<T, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D, Elem = B::Cell>,
          M::GeoElem: Cell<T> + HasDim<T, D>,
          <M::GeoElem as Cell<T>>::GeoMap: Chart<T, Coord = B::Coord<T>>, // todo: replace with HasBasisCoord<T, B>, but this does not infer HasBasisCoord<T, GradBasis<B, D>> for some reason?
          B: MeshGradBasis<T, D> + Clone,
          B::Coord<T>: Copy,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          Const<D>: DimMinSelf,
          DefaultAllocator: EvalGradAllocator<B::LocalBasis, D> + SelectCoeffsAllocator<B::LocalBasis>,
          DefaultAllocator: EvalGradAllocator<GradBasis<B::LocalBasis, D>, D> + SelectCoeffsAllocator<GradBasis<B::LocalBasis, D>> // fixme: this bound should be automatically fulfilled. Why isn't it?
    {
        // Compute gradient of uh (todo: possibly add as input argument?)
        // todo: very ugly. When GradBasisPullback gets refactored, update this code!
        let grad_space_parametric = uh.space.clone().grad();
        let grad_basis = GradBasisPullback { msh: self.0.msh, grad_basis: grad_space_parametric.basis };
        let grad_space = Space::new(grad_basis);
        // let uh_grad = uh.clone().grad(&grad_space);
        let uh_grad = LinCombination::new(uh.coeffs.clone(), &grad_space).unwrap();

        // Calculate ||u - uh||^2 + ||grad u - grad uh||^2
        self.0.error_squared(uh, u, quad) + self.0.error_squared(&uh_grad, u_grad, quad)
    }

    /// Calculates the H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`,
    /// using the quadrature rule `quad`.
    pub fn error<T, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          M: Mesh<'a, T, D, D, Elem = B::Cell>,
          M::GeoElem: Cell<T> + HasDim<T, D>,
          <M::GeoElem as Cell<T>>::GeoMap: Chart<T, Coord = B::Coord<T>>, // todo: replace with HasBasisCoord<T, B>, but this does not infer HasBasisCoord<T, GradBasis<B, D>> for some reason?
          B: MeshGradBasis<T, D> + Clone,
          B::Coord<T>: Copy,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: QuadratureOnParametricCell<T, M::GeoElem>,
          Const<D>: DimMinSelf,
          DefaultAllocator: EvalGradAllocator<B::LocalBasis, D> + SelectCoeffsAllocator<B::LocalBasis>,
          DefaultAllocator: EvalGradAllocator<GradBasis<B::LocalBasis, D>, D> + SelectCoeffsAllocator<GradBasis<B::LocalBasis, D>> // fixme: this bound should be automatically fulfilled. Why isn't it?
    {
        self.error_squared(uh, u, u_grad, quad).sqrt()
    }
}