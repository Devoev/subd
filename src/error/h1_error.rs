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
pub struct H1Norm<'a, M> {
    /// Mesh defining the geometry discretization.
    msh: &'a M,
}

impl<'a, M> H1Norm<'a, M> {
    /// Constructs a new [`H1Norm`] for the given `msh`.
    pub fn new(msh: &'a M) -> Self {
        H1Norm { msh }
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
        // Calculate sqrt( |u|^2 + |grad u|^2 )
        let l2 = L2Norm::new(self.msh);
        (l2.norm(u, quad).powi(2) + l2.norm(u_grad, quad).powi(2)).sqrt()
    }

    /// Calculates the H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`,
    /// using the quadrature rule `quad`.
    pub fn error<T, X, B, const D: usize, U, UGrad, Q>(&self, uh: &LinCombination<T, X, B, D>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<T, X, M::GeoElem, Q, D>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          X: Dimensioned<T, D> + Copy,
          M: Mesh<'a, T, X, D, D, Elem = B::Elem>,
          M::GeoElem: Cell<T, X, D, D>,
          B: LocalBasis<T, X, NumComponents=U1>,
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
        // Calculate sqrt( |u - uh|^2 + |grad u - grad uh|^2 )
        let l2 = L2Norm::new(self.msh);
        let err_l2 = l2.error(uh, u, quad);

        // Compute L2 error of gradient
        // todo: this should be computed by L2Norm instead. Adjust API for that
        let err_h1 = self.msh.elem_iter()
            .map(|elem| {
                // Get geometrical and reference element
                let geo_elem = self.msh.geo_elem(&elem);
                let ref_elem = geo_elem.ref_cell();

                // Evaluate functions at quadrature nodes of element
                let uh_grad = quad.nodes_ref(&ref_elem).map(|x| uh.eval_grad_on_elem(&elem, x));
                let u_grad = quad.nodes_elem(&geo_elem).map(u_grad);

                // Calculate L2 error on element
                let du_norm_squared = zip(uh_grad, u_grad).map(|(uh_grad, u_grad)| (uh_grad - u_grad).norm_squared());
                quad.integrate_elem(&geo_elem, du_norm_squared)
            })
            .sum::<T>()
            .sqrt();

        (err_l2.powi(2) + err_h1.powi(2)).sqrt()
    }
}