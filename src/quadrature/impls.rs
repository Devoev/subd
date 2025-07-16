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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use super::*;

    fn setup() -> (GaussLegendre, GaussLegendre) {
        (GaussLegendre::new(2).unwrap(), GaussLegendre::new(4).unwrap())
    }

    #[test]
    fn nodes_elem() {
        let (q2, q4) = setup();

        // Test quadrature of degree 2
        let nodes: Vec<f64> = q2.nodes_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(nodes[0], 0.57735, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], -0.57735, epsilon = 1e-5);

        let nodes: Vec<f64> = q2.nodes_elem(&UnitCube).collect();
        assert_relative_eq!(nodes[0], 0.788675, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.211325, epsilon = 1e-5);

        let interval = CartCell::new_univariate(0.5, 1.0);
        let nodes: Vec<f64> = q2.nodes_elem(&interval).collect();
        assert_relative_eq!(nodes[0], 0.8943375, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.6056625, epsilon = 1e-5);

        // Test quadrature of degree 4
        let nodes: Vec<f64> = q4.nodes_elem(&SymmetricUnitCube).collect();
        assert_relative_eq!(nodes[0], 0.861136, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.339981, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], -0.339981, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], -0.861136, epsilon = 1e-5);

        let nodes: Vec<f64> = q4.nodes_elem(&UnitCube).collect();
        assert_relative_eq!(nodes[0], 0.930568, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.6699905, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], 0.3300095, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], 0.069432, epsilon = 1e-5);

        let interval = CartCell::new_univariate(0.5, 1.0);
        let nodes: Vec<f64> = q4.nodes_elem(&interval).collect();
        assert_relative_eq!(nodes[0], 0.965284, epsilon = 1e-5);
        assert_relative_eq!(nodes[1], 0.83499525, epsilon = 1e-5);
        assert_relative_eq!(nodes[2], 0.66500475, epsilon = 1e-5);
        assert_relative_eq!(nodes[3], 0.534716, epsilon = 1e-5);
    }
}