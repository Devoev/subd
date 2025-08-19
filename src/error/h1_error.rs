use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::cells::geo::Cell;
use crate::error::l2_error::L2Norm;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::traits::Mesh;
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::Quadrature;
use nalgebra::{Const, DefaultAllocator, DimMin, Dyn, OVector, Point, RealField, SVector, U1};
use std::iter::{zip, Product, Sum};
use nalgebra::allocator::Allocator;
use crate::basis::eval::EvalGrad;

/// H1-norm on a mesh.
pub struct H1Norm<'a, M>(L2Norm<'a, M>);

impl<'a, M> H1Norm<'a, M> {
    /// Constructs a new [`H1Norm`] for the given `msh`.
    pub fn new(msh: &'a M) -> Self {
        H1Norm(L2Norm::new(msh))
    }

    /// Calculates the squared H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<T, X, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where
        T: RealField + Copy + Product<T> + Sum<T>,
        X: Dimensioned<T, D> + Copy,
        M: Mesh<'a, T, X, D, D>,
        M::GeoElem: Cell<T, X, D, D>,
        U: Fn(Point<T, D>) -> OVector<T, U1>,
        UGrad: Fn(Point<T, D>) -> SVector<T, D>,
        Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
        Const<D>: DimMin<Const<D>, Output=Const<D>>
    {
        // Calculate ||u||^2 + ||grad u||^2
        self.0.norm_squared(u, quad) + self.0.norm_squared(u_grad, quad)
    }

    /// Calculates the H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<T, X, const D: usize, U, UGrad, Q>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where
        T: RealField + Copy + Product<T> + Sum<T>,
        X: Dimensioned<T, D> + Copy,
        M: Mesh<'a, T, X, D, D>,
        M::GeoElem: Cell<T, X, D, D>,
        U: Fn(Point<T, D>) -> OVector<T, U1>,
        UGrad: Fn(Point<T, D>) -> SVector<T, D>,
        Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
        Const<D>: DimMin<Const<D>, Output=Const<D>>
    {
       self.norm_squared(u, u_grad, quad).sqrt()
    }

    /// Calculates the squared H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`, using the quadrature rule `quad`.
    pub fn error_squared<T, X, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, X, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, X, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, X, D, D>,
          B: LocalBasis<T, X, NumComponents=U1> + Clone,
          B::ElemBasis: EvalGrad<T::RealField, X, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>,
          DefaultAllocator: Allocator<U1>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>,
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
    pub fn error<T, X, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, X, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, X, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, X, D, D>,
          B: LocalBasis<T, X, NumComponents=U1> + Clone,
          B::ElemBasis: EvalGrad<T::RealField, X, D>,
          U: Fn(Point<T, D>) -> OVector<T, U1>,
          UGrad: Fn(Point<T, D>) -> SVector<T, D>,
          Q: Quadrature<T, X, <M::GeoElem as Cell<T, X, D, D>>::RefCell>,
          Const<D>: DimMin<Const<D>, Output = Const<D>>,
          DefaultAllocator: Allocator<U1>,
          DefaultAllocator: Allocator<<B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<U1, <B::ElemBasis as Basis>::NumBasis>,
          DefaultAllocator: Allocator<Const<D>, <B::ElemBasis as Basis>::NumBasis>,
    {
        self.error_squared(uh, u, u_grad, quad).sqrt()
    }
}