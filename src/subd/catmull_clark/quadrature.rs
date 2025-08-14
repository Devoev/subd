use crate::cells::unit_cube::UnitCube;
use crate::quadrature::traits::Quadrature;
use crate::subd::patch::subd_unit_square::SubdUnitSquare;
use itertools::Itertools;
use nalgebra::RealField;
use std::iter::Sum;
use std::marker::PhantomData;

/// Quadrature rule on the unit square in [subdivision form](SubdUnitSquare).
#[derive(Clone, Copy, Debug)]
pub struct SubdUnitSquareQuad<T: RealField, Q, const M: usize> {
    /// Regular quadrature rule on the unit square.
    quad_reg: Q,

    /// Maximum number of sub-segments to evaluate during quadrature of an irregular patch.
    m_max: usize,

    _phantom: PhantomData<T>,
}

impl <T: RealField, Q, const M: usize> SubdUnitSquareQuad<T, Q, M> {
    /// Constructs a new [`SubdUnitSquareQuad`] from the given
    /// quadrature rule `ref_quad` on the reference domain.
    pub fn new(ref_quad: Q, m_max: usize) -> Self {
        SubdUnitSquareQuad { quad_reg: ref_quad, m_max, _phantom: Default::default() }
    }
}

impl <T, Q> SubdUnitSquareQuad<T, Q, 2>
    where T: RealField + Sum + Copy,
          Q: Quadrature<T, (T, T), UnitCube<2>>
{
    /// Returns an iterator over all nodes in the regular reference domain (unit square).
    pub fn nodes_ref_regular(&self) -> impl Iterator<Item = (T, T)> + '_ {
        self.quad_reg.nodes_elem(&UnitCube)
    }

    /// Returns an iterator over all weights in the regular reference domain (unit square).
    pub fn weights_ref_regular(&self) -> impl Iterator<Item=T> + '_ {
        self.quad_reg.weights_elem(&UnitCube)
    }

    /// Returns an iterator over all nodes in the irregular reference domain.
    pub fn nodes_ref_irregular(&self) -> Vec<(T, T)> {
        let nodes_unit_square = self.nodes_ref_regular().collect_vec();

        if self.m_max == 0 { return nodes_unit_square }

        let mut nodes = vec![];

        let two = T::from_i8(2).unwrap();
        let half = T::from_f64(0.5).unwrap();
        let mut scale = T::one();
        for _ in 0..self.m_max {
            // Calculate scaled nodes
            scale *= two;
            let nodes_scaled = nodes_unit_square.iter()
                .map(|&(u, v)| (u / scale, v / scale));

            // Transform nodes into all 3 squares in L-shape
            nodes.extend(nodes_scaled.clone().map(|(u, v)| (u + half, v)));
            nodes.extend(nodes_scaled.clone().map(|(u, v)| (u + half, v + half)));
            nodes.extend(nodes_scaled.clone().map(|(u, v)| (u, v + half)));
        }

        nodes
    }

    /// Returns a vector of weights in the irregular reference domain (tiled unit square).
    pub fn weights_ref_irregular(&self) -> Vec<T> {
        let weights_unit_square = self.weights_ref_regular().collect_vec();

        if self.m_max == 0 { return weights_unit_square }

        let mut weights = vec![];

        let four = T::from_i8(4).unwrap();
        let mut scale = T::one();
        for _ in 0..self.m_max {
            // Calculate scaled weights
            scale *= four;
            let weights_scaled = weights_unit_square.iter()
                .map(|&wi| wi / scale);

            // Add scaled weights for all 3 squares in L-shape
            weights.extend(weights_scaled.clone());
            weights.extend(weights_scaled.clone());
            weights.extend(weights_scaled);
        }

        weights
    }
}

impl <T, Q> Quadrature<T, (T, T), SubdUnitSquare> for SubdUnitSquareQuad<T, Q, 2>
where T: RealField + Sum + Copy,
      Q: Quadrature<T, (T, T), UnitCube<2>>
{
    fn nodes_elem(&self, elem: &SubdUnitSquare) -> impl Iterator<Item=(T, T)> {
        match elem {
            SubdUnitSquare::Regular => self.nodes_ref_regular().collect_vec().into_iter(),
            SubdUnitSquare::Irregular => self.nodes_ref_irregular().into_iter()
        }
    }

    fn weights_elem(&self, elem: &SubdUnitSquare) -> impl Iterator<Item=T> {
        match elem {
            SubdUnitSquare::Regular => self.weights_ref_regular().collect_vec().into_iter(),
            SubdUnitSquare::Irregular => self.weights_ref_irregular().into_iter()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadrature::tensor_prod::GaussLegendreMulti;
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use gauss_quad::GaussLegendre;

    /// Returns a 2D Gauss-Legendre quadrature with degree `2`
    /// in both `x`-direction and `y`-direction.
    fn setup() -> GaussLegendreMulti<f64, 2> {
        GaussLegendreMulti::new([GaussLegendre::new(2).unwrap(), GaussLegendre::new(2).unwrap()])
    }

    #[test]
    fn nodes_ref_irregular() {
        let gauss_quad = setup();
        let quad = SubdUnitSquareQuad::<f64, _, 2>::new(gauss_quad, 1);

        let nodes = quad.nodes_ref_irregular().iter().map(|&(u, v)| [u, v]).collect_vec();
        assert_eq!(nodes.len(), 12);
        assert_abs_diff_eq!(nodes[0].as_slice(), [0.394337567297406 + 0.5, 0.394337567297406].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[1].as_slice(), [0.394337567297406 + 0.5, 0.105662432702594].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[2].as_slice(), [0.105662432702594 + 0.5, 0.394337567297406].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[3].as_slice(), [0.105662432702594 + 0.5, 0.105662432702594].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[4].as_slice(), [0.394337567297406 + 0.5, 0.394337567297406 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[5].as_slice(), [0.394337567297406 + 0.5, 0.105662432702594 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[6].as_slice(), [0.105662432702594 + 0.5, 0.394337567297406 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[7].as_slice(), [0.105662432702594 + 0.5, 0.105662432702594 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[8].as_slice(), [0.394337567297406, 0.394337567297406 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[9].as_slice(), [0.394337567297406, 0.105662432702594 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[10].as_slice(), [0.105662432702594, 0.394337567297406 + 0.5].as_slice(), epsilon = 1e-10);
        assert_abs_diff_eq!(nodes[11].as_slice(), [0.105662432702594, 0.105662432702594 + 0.5].as_slice(), epsilon = 1e-10);
    }

    #[test]
    fn weights_ref_irregular() {
        let gauss_quad = setup();
        let quad = SubdUnitSquareQuad::<f64, _, 2>::new(gauss_quad, 1);

        let weights = quad.weights_ref_irregular();
        assert_eq!(weights.len(), 12);
        assert!(weights.iter().all(|&wi| abs_diff_eq!(wi, 0.0625, epsilon = 1e-13)));
    }
}