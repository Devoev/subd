use crate::basis::eval::{EvalGrad, EvalGradAllocator};
use crate::basis::grad::GradBasis;
use crate::basis::lin_combination::{LinCombination, SelectCoeffsAllocator};
use crate::basis::local::LocalBasis;
use crate::cells::geo::Cell;
use crate::error::l2_error::L2Norm;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::Quadrature;
use nalgebra::{Const, DefaultAllocator, DimMin, OVector, Point, RealField, SVector, U1};
use std::iter::{Product, Sum};
use crate::diffgeo::chart::Chart;

/// H1-norm on a mesh.
pub struct H1Norm<'a, M>(L2Norm<'a, M>);

impl<'a, M> H1Norm<'a, M> {
    /// Constructs a new [`H1Norm`] for the given `msh`.
    pub fn new(msh: &'a M) -> Self {
        H1Norm(L2Norm::new(msh))
    }

    /// Calculates the squared H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<T, X, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<T, M::GeoElem, Q, D>) -> T
    where
        T: RealField + Copy + Product<T> + Sum<T>,
        X: Dimensioned<T, D> + Copy,
        M: Mesh<'a, T, D, D>,
        M::GeoElem: Cell<T, D, D>,
        <M::GeoElem as Cell<T, D, D>>::GeoMap: Chart<T, D, D, Coord = X>,
        U: Fn(Point<T, D>) -> OVector<T, U1>,
        UGrad: Fn(Point<T, D>) -> SVector<T, D>,
        Q: Quadrature<T, <M::GeoElem as Cell<T, D, D>>::RefCell, Node = X>,
        Const<D>: DimMin<Const<D>, Output=Const<D>>
    {
        // Calculate ||u||^2 + ||grad u||^2
        self.0.norm_squared(u, quad) + self.0.norm_squared(u_grad, quad)
    }

    /// Calculates the H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<T, X, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<T, M::GeoElem, Q, D>) -> T
    where
        T: RealField + Copy + Product<T> + Sum<T>,
        X: Dimensioned<T, D> + Copy,
        M: Mesh<'a, T, D, D>,
        M::GeoElem: Cell<T, D, D>,
        <M::GeoElem as Cell<T, D, D>>::GeoMap: Chart<T, D, D, Coord = X>,
        U: Fn(Point<T, D>) -> OVector<T, U1>,
        UGrad: Fn(Point<T, D>) -> SVector<T, D>,
        Q: Quadrature<T, <M::GeoElem as Cell<T, D, D>>::RefCell, Node = X>,
        Const<D>: DimMin<Const<D>, Output=Const<D>>
    {
       self.norm_squared(u, u_grad, quad).sqrt()
    }

    /// Calculates the squared H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`, using the quadrature rule `quad`.
    pub fn error_squared<T, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<T, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          B::Coord<T>: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, D, D>,
          <M::GeoElem as Cell<T, D, D>>::GeoMap: Chart<T, D, D, Coord = B::Coord<T>>,
          B: LocalBasis<T, NumComponents=U1> + Clone,
          B::ElemBasis: EvalGrad<T::RealField, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: Quadrature<T, <M::GeoElem as Cell<T, D, D>>::RefCell, Node = B::Coord<T>>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D> + SelectCoeffsAllocator<B::ElemBasis>,
          DefaultAllocator: EvalGradAllocator<GradBasis<B::ElemBasis, D>, D> + SelectCoeffsAllocator<GradBasis<B::ElemBasis, D>> // fixme: this bound should be automatically fulfilled. Why isn't it?
    {
        // Compute gradient of uh (todo: possibly add as input argument?)
        let grad_space = uh.space.clone().grad();
        let uh_grad = uh.clone().grad(&grad_space);

        // Calculate ||u - uh||^2 + ||grad u - grad uh||^2
        self.0.error_squared(uh, u, quad) + self.0.error_squared(&uh_grad, u_grad, quad)
    }

    /// Calculates the H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`,
    /// using the quadrature rule `quad`.
    pub fn error<T, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<T, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          B::Coord<T>: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, D, D>,
          <M::GeoElem as Cell<T, D, D>>::GeoMap: Chart<T, D, D, Coord = B::Coord<T>>,
          B: LocalBasis<T, NumComponents=U1> + Clone,
          B::ElemBasis: EvalGrad<T::RealField, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: Quadrature<T, <M::GeoElem as Cell<T, D, D>>::RefCell, Node = B::Coord<T>>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>,
          DefaultAllocator: EvalGradAllocator<B::ElemBasis, D> + SelectCoeffsAllocator<B::ElemBasis>,
          DefaultAllocator: EvalGradAllocator<GradBasis<B::ElemBasis, D>, D> + SelectCoeffsAllocator<GradBasis<B::ElemBasis, D>> // fixme: this bound should be automatically fulfilled. Why isn't it?
    {
        self.error_squared(uh, u, u_grad, quad).sqrt()
    }
}