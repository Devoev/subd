//! Functions and methods for evaluating and integrating over subdivision surfaces and patches.

use std::iter::Sum;
use gauss_quad::GaussLegendre;
use nalgebra::{DVector, Matrix2, Point2, RealField};
use num_traits::ToPrimitive;
use crate::subd::basis;
use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;

impl <T: RealField + Copy + ToPrimitive> Patch<'_, T> {

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
        let cols = match self {
            Patch::Regular { .. } => {
                let b_du = basis::eval_regular_du(u, v);
                let b_dv = basis::eval_regular_dv(u, v);
                [c.clone() * b_du, c * b_dv]
            }
            Patch::Irregular { .. } => {
                let (_, n) = self.irregular_node().expect("Patch must be irregular!");
                let b_du = basis::eval_irregular_du(u, v, n);
                let b_dv = basis::eval_irregular_dv(u, v, n);
                [c.clone() * b_du, c * b_dv]
            }
            Patch::BoundaryRegular { .. } => {
                let b_du = basis::eval_boundary_du(u, v, false, true);
                let b_dv = basis::eval_boundary_dv(u, v, false, true);
                [c.clone() * b_du, c * b_dv]
            }
            Patch::BoundaryRegularCorner { .. } => {
                let b_du = basis::eval_boundary_du(u, v, true, true);
                let b_dv = basis::eval_boundary_dv(u, v, true, true);
                [c.clone() * b_du, c * b_dv]
            }
        };

        Matrix2::from_columns(&cols)
    }

    /// Numerically integrates the given parametric function `f: (0,1)² ⟶ ℝ` over this patch
    /// using `num_quad` Gaussian quadrature points per parametric direction.
    pub fn integrate(&self, f: impl Fn(T, T) -> T, num_quad: usize) -> T {
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

    /// Numerically calculates the area of this patch using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.integrate(|_, _| T::one(), 2)
    }

    /// Evaluates the basis functions on this patch at the parametric point `(u,v)`.
    pub fn eval_basis(&self, u: T, v: T) -> DVector<T> {
        let b = match self {
            Patch::Regular { .. } => basis::eval_regular(u, v).as_slice().to_vec(),
            Patch::Irregular { .. } => {
                let (_, n) = self.irregular_node().expect("Patch must be irregular!");
                basis::eval_irregular(u, v, n).as_slice().to_vec()
            },
            Patch::BoundaryRegular { .. } => basis::eval_boundary(u, v, false, true).as_slice().to_vec(),
            Patch::BoundaryRegularCorner { .. } => basis::eval_boundary(u, v, true, true).as_slice().to_vec(),
        };
        DVector::from_vec(b)
    }
}

impl<T: RealField + Copy + ToPrimitive + Sum> QuadMesh<T> {

    /// Numerically calculates the area of this surface using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.patches().map(|patch| patch.calc_area()).sum()
    }
}