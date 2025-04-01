//! Functions and methods for evaluating and integrating over subdivision surfaces and patches.

use std::iter::Sum;
use gauss_quad::GaussLegendre;
use nalgebra::{Matrix2, Point2, RealField};
use num_traits::ToPrimitive;
use crate::subd::basis;
use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;

impl <T: RealField + Copy + ToPrimitive> Patch<'_, T> {

    /// Evaluates this patch at the parametric point `(u,v)`.
    pub fn eval(&self, u: T, v: T) -> Point2<T> {
        match self {
            Patch::Regular { .. } => self.eval_regular(u, v),
            Patch::Irregular { .. } => self.eval_irregular(u, v),
            Patch::BoundaryRegular { .. } => self.eval_boundary_planar(u, v),
            Patch::BoundaryRegularCorner { .. } => self.eval_boundary_convex(u, v),
        }
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

    /// Evaluates this regular patch at the parametric point `(u,v)`.
    fn eval_regular(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_regular(u, v);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this planar boundary patch at the parametric point `(u,v)`.
    fn eval_boundary_planar(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_boundary(u, v, false, true);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this convex boundary patch at the parametric point `(u,v)`.
    fn eval_boundary_convex(&self, u: T, v: T) -> Point2<T> {
        // Evaluate basis functions and patch
        let b = basis::eval_boundary(u, v, true, true);
        Point2::from(self.coords() * b)
    }

    /// Evaluates this irregular patch at the parametric point `(u,v)`.
    fn eval_irregular(&self, u: T, v: T) -> Point2<T> {
        // Get valence of irregular node
        let (_, n) = self.irregular_node().expect("Patch must be irregular!");

        // Evaluate basis functions and patch
        let b = basis::eval_irregular(u, v, n);
        Point2::from(self.coords() * b)
    }
}

impl<T: RealField + Copy + ToPrimitive + Sum> QuadMesh<T> {

    /// Numerically calculates the area of this surface using Gaussian quadrature.
    pub fn calc_area(&self) -> T {
        self.patches().map(|patch| patch.calc_area()).sum()
    }
}