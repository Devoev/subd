//! Functions and methods for evaluating and integrating over subdivision surfaces and patches.

use crate::subd::basis;
use crate::subd::mesh::QuadMesh;
use crate::subd::patch::{NodeConnectivity, Patch};
use gauss_quad::GaussLegendre;
use nalgebra::{DVector, Dyn, Matrix2, OMatrix, Point2, RealField, U2};
use num_traits::ToPrimitive;
use std::iter::Sum;

impl <T: RealField + Copy + ToPrimitive> Patch<'_, T> {
    /// Evaluates the basis functions on this patch at the parametric point `(u,v)`.
    pub fn eval_basis(&self, u: T, v: T) -> DVector<T> {
        let b = match self.connectivity {
            NodeConnectivity::Regular { .. } => basis::eval_regular(u, v).as_slice().to_vec(),
            NodeConnectivity::Irregular { .. } => {
                let (_, n) = self.irregular_node().expect("Patch must be irregular!");
                basis::eval_irregular(u, v, n).as_slice().to_vec()
            },
            NodeConnectivity::Boundary { .. } => basis::eval_boundary(u, v, false, true).as_slice().to_vec(),
            NodeConnectivity::Corner { .. } => basis::eval_boundary(u, v, true, true).as_slice().to_vec(),
        };
        DVector::from_vec(b)
    }

    /// Evaluates the gradients of the basis functions on this patch at the parametric point `(u,v)`.
    pub fn eval_basis_grad(&self, u: T, v: T) -> OMatrix<T, Dyn, U2> {
        let b = match self.connectivity {
            NodeConnectivity::Regular { .. } => basis::eval_regular_grad(u, v).as_slice().to_vec(),
            NodeConnectivity::Irregular { .. } => {
                let (_, n) = self.irregular_node().expect("Patch must be irregular!");
                basis::eval_irregular_grad(u, v, n).as_slice().to_vec()
            },
            NodeConnectivity::Boundary { .. } => basis::eval_boundary_grad(u, v, false, true).as_slice().to_vec(),
            NodeConnectivity::Corner { .. } => basis::eval_boundary_grad(u, v, true, true).as_slice().to_vec(),
        };

        OMatrix::<T, Dyn, U2>::from_vec(b)
    }

    /// Evaluates this patch at the parametric point `(u,v)`.
    pub fn eval(&self, u: T, v: T) -> Point2<T> {
        let b = self.eval_basis(u, v);
        let c = self.coords();
        Point2::from(c * b)
    }

    /// Evaluates the jacobian of this patch at the parametric point `(u,v)`.
    pub fn eval_jacobian(&self, u: T, v: T) -> Matrix2<T> {
        // Get patch coordinates
        let c = self.coords();

        // Calculate columns for Jacobian (derivatives d_phi / t_i)
        let cols = match self.connectivity {
            NodeConnectivity::Regular { .. } => {
                let b_du = basis::eval_regular_du(u, v);
                let b_dv = basis::eval_regular_dv(u, v);
                [c.clone() * b_du, c * b_dv]
            }
            NodeConnectivity::Irregular { .. } => {
                let (_, n) = self.irregular_node().expect("Patch must be irregular!");
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

    /// Numerically integrates the given parametric function `f: (0,1)² ⟶ ℝ` over this patch
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate_pullback(&self, f: impl Fn(T, T) -> T, num_quad: usize) -> T {
        let quad = GaussLegendre::new(num_quad).unwrap();
        let integrand = |u, v| {
            let u = T::from_f64(u).unwrap();
            let v = T::from_f64(v).unwrap();
            let d_phi = self.eval_jacobian(u, v);
            (f(u, v) * d_phi.determinant().abs()).to_f64().unwrap()
        };
        T::from_f64(
            quad.integrate(0.0, 1.0, |v| quad.integrate(0.0, 1.0, |u| integrand(u, v)))
        ).unwrap()
    }

    /// Numerically integrates the given function `f: S ⟶ ℝ` over this patch in physical domain
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate(&self, f: impl Fn(Point2<T>) -> T, num_quad: usize) -> T {
        self.integrate_pullback(|u, v| f(self.eval(u, v)), num_quad)
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
        self.integrate_pullback(|patch, u, v| f(patch.eval(u, v)), num_quad)
    }

    /// Numerically calculates the area of this surface using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.patches().map(|patch| patch.calc_area()).sum()
    }
}