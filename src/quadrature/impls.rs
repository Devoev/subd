use crate::cells::cartesian::CartCell;
use crate::cells::geo::Cell;
use crate::cells::lerp::Lerp;
use crate::cells::unit_cube::{SymmetricUnitCube, UnitCube};
use crate::quadrature::traits::Quadrature;
use gauss_quad::GaussLegendre;
use nalgebra::{vector, RealField};
use std::iter::Sum;

// todo: are the two implementations really needed? or is the 2nd one sufficient

impl <T: RealField + Sum> Quadrature<T, T, SymmetricUnitCube<1>> for GaussLegendre {
    fn nodes_elem(&self, _elem: &SymmetricUnitCube<1>) -> impl Iterator<Item=T> {
        self.nodes()
            .map(|&xi| T::from_f64(xi).unwrap())
    }

    fn weights_elem(&self, _elem: &SymmetricUnitCube<1>) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi).unwrap())
    }
}

impl <T: RealField + Sum> Quadrature<T, T, UnitCube<1>> for GaussLegendre {
    fn nodes_elem(&self, _elem: &UnitCube<1>) -> impl Iterator<Item=T> {
        self.nodes()
            .map(|&xi| T::from_f64((xi + 1.0) / 2.0).unwrap())
    }

    fn weights_elem(&self, _elem: &UnitCube<1>) -> impl Iterator<Item=T> {
        self.weights()
            .map(|&wi| T::from_f64(wi / 2.0).unwrap())
    }
}

impl <T: RealField + Copy + Sum> Quadrature<T, T, CartCell<T, 1>> for GaussLegendre {
    fn nodes_elem(&self, elem: &CartCell<T, 1>) -> impl Iterator<Item=T> {
        let lerp: Lerp<T, 1> = <CartCell<T, 1> as Cell<T, T, 1, 1>>::geo_map(elem);
        self.nodes_elem(&SymmetricUnitCube).map(move |xi: T| lerp.transform_symmetric(vector![xi]).x)
    }

    fn weights_elem(&self, elem: &CartCell<T, 1>) -> impl Iterator<Item=T> {
        let lerp: Lerp<T, 1> = <CartCell<T, 1> as Cell<T, T, 1, 1>>::geo_map(elem);
        let d_phi = lerp.jacobian().x / T::from_i32(2).unwrap(); // todo: add new jacobian method for symmetric interval
        self.weights_elem(&SymmetricUnitCube).map(move |wi: T| wi * d_phi)
    }
}

// todo: remove collect and use next().unwrap()
//  this can be done when the return value of nodes_elem and weights_elem is changed

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use itertools::Itertools;
    use num_traits::Pow;
    use super::*;

    /// Returns a 2-point and 4-point Gauss-Legendre rule.
    fn setup() -> (GaussLegendre, GaussLegendre) {
        (GaussLegendre::new(2).unwrap(), GaussLegendre::new(4).unwrap())
    }

    #[test]
    fn nodes_elem() {
        let (q2, q4) = setup();

        // Test symmetric unit interval [-1,1]
        let nodes: Vec<f64> = q2.nodes_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(nodes[0], 0.57735, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], -0.57735, epsilon = 1e-5);

        let nodes: Vec<f64> = q4.nodes_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(nodes[0], 0.861136, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.339981, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], -0.339981, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], -0.861136, epsilon = 1e-5);

        // Test unit interval [0,1]
        let nodes: Vec<f64> = q2.nodes_elem(&UnitCube).collect();
        assert_relative_eq!(nodes[0], 0.788675, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.211325, epsilon = 1e-5);

        let nodes: Vec<f64> = q4.nodes_elem(&UnitCube).collect();
        assert_relative_eq!(nodes[0], 0.930568, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.6699905, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], 0.3300095, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], 0.069432, epsilon = 1e-5);

        // Test interval [0.5,1]
        let interval = CartCell::new_univariate(0.5, 1.0);

        let nodes: Vec<f64> = q2.nodes_elem(&interval).collect();
        assert_relative_eq!(nodes[0], 0.8943375, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.6056625, epsilon = 1e-5);

        let nodes: Vec<f64> = q4.nodes_elem(&interval).collect();
        assert_relative_eq!(nodes[0], 0.965284, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.83499525, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], 0.66500475, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], 0.534716, epsilon = 1e-5);
    }

    #[test]
    fn weights_elem() {
        let (q2, q4) = setup();

        // Test symmetric unit interval [-1,1]
        let weights: Vec<f64> = q2.weights_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(weights[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 1.0, epsilon = 1e-5);

        let weights: Vec<f64> = q4.weights_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(weights[0], 0.347855, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 0.652145, epsilon = 1e-5);
        assert_relative_eq!(weights[2], 0.652145, epsilon = 1e-5);
        assert_relative_eq!(weights[3], 0.347855, epsilon = 1e-5);

        // Test unit interval [0,1]
        let weights: Vec<f64> = q2.weights_elem(&UnitCube).collect();
        assert_relative_eq!(weights[0], 0.5, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 0.5, epsilon = 1e-5);

        let weights: Vec<f64> = q4.weights_elem(&UnitCube).collect();
        assert_relative_eq!(weights[0], 0.1739275, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 0.3260725, epsilon = 1e-5);
        assert_relative_eq!(weights[2], 0.3260725, epsilon = 1e-5);
        assert_relative_eq!(weights[3], 0.1739275, epsilon = 1e-5);

        // Test interval [0.5,1]
        let interval = CartCell::new_univariate(0.5, 1.0);

        let weights: Vec<f64> = q2.weights_elem(&interval).collect();
        assert_relative_eq!(weights[0], 0.25, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 0.25, epsilon = 1e-5);

        let weights: Vec<f64> = q4.weights_elem(&interval).collect();
        assert_relative_eq!(weights[0], 0.08696375, epsilon = 1e-5);
        assert_relative_eq!(weights[1], 0.16303625, epsilon = 1e-5);
        assert_relative_eq!(weights[2], 0.16303625, epsilon = 1e-5);
        assert_relative_eq!(weights[3], 0.08696375, epsilon = 1e-5);
    }

    #[test]
    fn integrate_exact() {
        // q2: n = 2 is exact to degree 2n-1 = 3
        // q4: n = 4 is exact to degree 2n-1 = 7
        let (q2, q4) = setup();

        // Integrate x^3 - x on [-5,-3]
        let interval = CartCell::new_univariate(-5.0, -3.0);
        let f = |x: f64| x.powi(3) - x;

        let int = q2.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, -128.0, epsilon = 1e-13);
        let int = q4.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, -128.0, epsilon = 1e-13);

        // Integrate 2x^3 + 2x^2 - 5 on [-1,6]
        let interval = CartCell::new_univariate(-1.0, 6.0);
        let f = |x: f64| 2.0*x.powi(3) + 2.0*x.powi(2) - 5.0;

        let int = q2.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, 4543.0 / 6.0, epsilon = 1e-13);
        let int = q4.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, 4543.0 / 6.0, epsilon = 1e-13);

        // Integrate x^7 on [-2,4]
        let interval = CartCell::new_univariate(-2.0, 4.0);
        let f = |x: f64| x.powi(7);

        let int = q4.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, 8160.0, epsilon = 1e-13);

        // Integrate -x^7 + 5x^5 + 2(x - 1)^2 on [-1,1]
        let interval = CartCell::new_univariate(-1.0, 1.0);
        let f = |x: f64| -x.powi(7) + 5.0*x.powi(5) + 2.0*(x - 1.0).powi(2);

        let int = q4.integrate_fn_elem(&interval, f);
        assert_relative_eq!(int, 16.0 / 3.0, epsilon = 1e-13);
    }
}