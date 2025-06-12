//! Functions and methods for evaluating and integrating over subdivision surfaces and patches.

use crate::subd_legacy::basis;
use crate::subd_legacy::mesh::QuadMesh;
use crate::subd_legacy::patch::{NodeConnectivity, Patch};
use gauss_quad::GaussLegendre;
use nalgebra::{DVector, Dyn, Matrix2, Matrix2x1, OMatrix, Point2, RealField, U2};
use num_traits::ToPrimitive;
use std::iter::Sum;

/// A mapping from the parametric domain `(0,1)²` to an arbitrary codomain.
pub trait ParametricMap<T> {
    /// This map evaluated at a parametric point.
    type Eval;

    /// Evaluates this map at `(u,v)`.
    fn eval(&self, u: T, v: T) -> Self::Eval;
}

/// Parametrization `φ: (0,1)² ⟶ Ω` of a surface patch.
pub struct Parametrization<'a, T: RealField>(&'a Patch<'a, T>);

impl <T: RealField + Copy + ToPrimitive> ParametricMap<T> for Parametrization<'_, T> {
    type Eval = Point2<T>;

    fn eval(&self, u: T, v: T) -> Self::Eval {
        let b = self.0.basis().eval(u, v);
        let c = self.0.coords();
        Point2::from(c * b)
    }
}

/// Set of basis functions `b_k: (0,1)² ⟶ ℝ` of a surface patch.
pub struct Basis<'a, T: RealField>(&'a Patch<'a, T>);

impl<T: RealField + Copy + ToPrimitive> ParametricMap<T> for Basis<'_, T> {
    type Eval = DVector<T>;

    fn eval(&self, u: T, v: T) -> Self::Eval {
        let b = match self.0.connectivity {
            NodeConnectivity::Regular { .. } => basis::eval_regular(u, v).as_slice().to_vec(),
            NodeConnectivity::Irregular (_, n) => basis::eval_irregular(u, v, n).as_slice().to_vec(),
            NodeConnectivity::Boundary { .. } => basis::eval_boundary(u, v, false, true).as_slice().to_vec(),
            NodeConnectivity::Corner { .. } => basis::eval_boundary(u, v, true, true).as_slice().to_vec(),
        };
        DVector::from_vec(b)
    }
}

/// Set of gradients of basis functions `grad(b_k): (0,1)² ⟶ ℝ²` of a surface patch.
pub struct BasisGrad<'a, T: RealField>(&'a Patch<'a, T>);

impl<T: RealField + Copy + ToPrimitive> ParametricMap<T> for BasisGrad<'_, T> {
    type Eval = OMatrix<T, Dyn, U2>;

    fn eval(&self, u: T, v: T) -> Self::Eval {
        let b = match self.0.connectivity {
            NodeConnectivity::Regular { .. } => basis::eval_regular_grad(u, v).as_slice().to_vec(),
            NodeConnectivity::Irregular (_, n) => basis::eval_irregular_grad(u, v, n).as_slice().to_vec(),
            NodeConnectivity::Boundary { .. } => basis::eval_boundary_grad(u, v, false, true).as_slice().to_vec(),
            NodeConnectivity::Corner { .. } => basis::eval_boundary_grad(u, v, true, true).as_slice().to_vec(),
        };

        OMatrix::<T, Dyn, U2>::from_vec(b)
    }
}

/// Jacobian matrix `dφ: (0,1)² ⟶ ℝ²ˣ²` of the [`Parametrization`].
pub struct Jacobian<'a, T: RealField>(&'a Patch<'a, T>);

impl<T: RealField + Copy + ToPrimitive> ParametricMap<T> for Jacobian<'_, T> {
    type Eval = Matrix2<T>;

    fn eval(&self, u: T, v: T) -> Self::Eval {
        // Get patch coordinates
        let c = self.0.coords();

        // Calculate columns for Jacobian (derivatives d_phi / t_i)
        let cols = match self.0.connectivity {
            NodeConnectivity::Regular { .. } => {
                let b_du = basis::eval_regular_du(u, v);
                let b_dv = basis::eval_regular_dv(u, v);
                [c.clone() * b_du, c * b_dv]
            }
            NodeConnectivity::Irregular (_, n) => {
                let b_du = basis::eval_irregular_du(u, v, n);
                let b_dv = basis::eval_irregular_dv(u, v, n);
                [c.clone() * b_du, c * b_dv]
            }
            NodeConnectivity::Boundary { .. } => {
                let b_du = basis::eval_boundary_du(u, v, false, true);
                let b_dv = basis::eval_boundary_dv(u, v, false, true);
                [c.clone() * b_du, c * b_dv]
            }
            NodeConnectivity::Corner { .. } => {
                let b_du = basis::eval_boundary_du(u, v, true, true);
                let b_dv = basis::eval_boundary_dv(u, v, true, true);
                [c.clone() * b_du, c * b_dv]
            }
        };

        Matrix2::from_columns(&cols)
    }
}

impl <T: RealField + Copy + ToPrimitive> Patch<'_, T> {
    /// Constructs the [`Parametrization`] of this patch.
    pub fn parametrization(&self) -> Parametrization<'_, T> {
        Parametrization(self)
    }

    /// Constructs the [`Basis`] of this patch.
    pub fn basis(&self) -> Basis<'_, T> {
        Basis(self)
    }

    /// Constructs the gradients [`BasisGrad`] of the basis functions of this patch.
    pub fn basis_grad(&self) -> BasisGrad<'_, T> {
        BasisGrad(self)
    }

    /// Constructs the [`Jacobian`] of this patch.
    pub fn jacobian(&self) -> Jacobian<'_, T> {
        Jacobian(self)
    }

    /// Numerically integrates the given parametric function `f: (0,1)² ⟶ ℝ` over this patch
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate_pullback(&self, f: impl Fn(T, T) -> T, num_quad: usize) -> T {
        let quad = GaussLegendre::new(num_quad).unwrap();
        let integrand = |u, v| {
            let u = T::from_f64(u).unwrap();
            let v = T::from_f64(v).unwrap();
            let d_phi = self.jacobian().eval(u, v);
            (f(u, v) * d_phi.determinant().abs()).to_f64().unwrap()
        };
        T::from_f64(
            quad.integrate(0.0, 1.0, |v| quad.integrate(0.0, 1.0, |u| integrand(u, v)))
        ).unwrap()
    }

    /// Numerically integrates the given function `f: S ⟶ ℝ` over this patch in physical domain
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate(&self, f: impl Fn(Point2<T>) -> T, num_quad: usize) -> T {
        self.integrate_pullback(|u, v| f(self.parametrization().eval(u, v)), num_quad)
    }

    /// Numerically calculates the area of this patch using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.integrate_pullback(|_, _| T::one(), 2)
    }
}

impl<T: RealField + Copy + ToPrimitive + Sum> QuadMesh<T> {

    /// Numerically integrates the per-patch pullback functions `f_k: (0,1)² ⟶ ℝ` over this surface
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate_pullback(&self, fk: impl Fn(&Patch<T>, T, T) -> T + Clone, num_quad: usize) -> T {
        self.patches().map(|patch| patch.integrate_pullback(|u, v| fk(&patch, u, v), num_quad)).sum()
    }

    /// Numerically integrates the given function `f: S ⟶ ℝ` over this surface
    /// using `num_quad` Gaussian quadrature points per parametric direction per patch.
    pub fn integrate(&self, f: impl Fn(Point2<T>) -> T + Clone, num_quad: usize) -> T {
        self.integrate_pullback(|patch, u, v| f(patch.parametrization().eval(u, v)), num_quad)
    }

    /// Numerically calculates the area of this surface using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.patches().map(|patch| patch.calc_area()).sum()
    }
}