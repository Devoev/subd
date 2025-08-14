use std::iter::{Product, Sum};
use itertools::Itertools;
use crate::quadrature::pullback::PullbackQuad;
use crate::subd::catmull_clark::patch::CatmarkPatch;
use nalgebra::{one, Point2, RealField};
use num_traits::ToPrimitive;
use crate::cells::unit_cube::UnitCube;
use crate::quadrature::traits::Quadrature;

/// Quadrature rule on [Catmull-Clark patches](CatmarkPatch).
struct CatmarkQuad<T: RealField, Q, const M: usize> {
    /// Regular quadrature rule.
    quad_reg: PullbackQuad<T, (T, T), CatmarkPatch<T, M>, Q, 2>,

    /// Maximum number of sub-rings to evaluate during quadrature of an irregular patch.
    m_max: usize,
}

impl <T: RealField, Q, const M: usize> CatmarkQuad<T, Q, M> {
    /// Constructs a new [`CatmarkQuad`] from the given
    /// quadrature rule `ref_quad` on the reference domain.
    pub fn new(ref_quad: Q, m_max: usize) -> Self {
        CatmarkQuad { quad_reg: PullbackQuad::new(ref_quad), m_max }
    }
}

impl <T, Q> CatmarkQuad<T, Q, 2>
    where T: RealField + Sum + Product + Copy + ToPrimitive,
          Q: Quadrature<T, (T, T), UnitCube<2>>
{
    /// Returns an iterator over all nodes in the regular reference domain (unit square).
    pub fn nodes_ref_regular(&self) -> impl Iterator<Item = (T, T)> + '_ {
        self.quad_reg.nodes_ref(&UnitCube)
    }

    /// Returns an iterator over all weights in the regular reference domain (unit square).
    pub fn weights_ref_regular(&self) -> impl Iterator<Item=T> + '_ {
        self.quad_reg.weights_ref(&UnitCube)
    }

    /// Returns an iterator over all nodes in the irregular reference domain.
    pub fn nodes_ref_irregular(&self) -> Vec<(T, T)> {
        let nodes_unit_square = self.nodes_ref_regular().collect_vec();
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
        let mut weights = vec![];

        let two = T::from_i8(2).unwrap();
        let mut scale = T::one();
        for _ in 0..self.m_max {
            // Calculate scaled weights
            scale *= two;
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

impl <T, Q> Quadrature<T, Point2<T>, CatmarkPatch<T, 2>> for CatmarkQuad<T, Q, 2>
where T: RealField + Sum + Product + Copy + ToPrimitive,
      Q: Quadrature<T, (T, T), UnitCube<2>>
{
    fn nodes_elem(&self, elem: &CatmarkPatch<T, 2>) -> impl Iterator<Item=Point2<T>> {
        match elem {
            CatmarkPatch::Irregular(_, _) => {
                todo!("This can be implemented later, since only the reference nodes are usually needed")
            }
            _ => self.quad_reg.nodes_elem(elem),
        }
    }

    fn weights_elem(&self, elem: &CatmarkPatch<T, 2>) -> impl Iterator<Item=T> {
        match elem {
            CatmarkPatch::Irregular(_, _) => {
                todo!("Implement nodes_ref_irregular and add weights_ref_irregular")
            }
            _ => self.quad_reg.weights_elem(elem),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::{abs_diff_eq, assert_abs_diff_eq};
    use gauss_quad::GaussLegendre;
    use crate::quadrature::tensor_prod::GaussLegendreMulti;
    use super::*;

    /// Returns a 2D Gauss-Legendre quadrature with degree `2`
    /// in both `x`-direction and `y`-direction.
    fn setup() -> GaussLegendreMulti<f64, 2> {
        GaussLegendreMulti::new([GaussLegendre::new(2).unwrap(), GaussLegendre::new(2).unwrap()])
    }

    #[test]
    fn nodes_ref_irregular() {
        let gauss_quad = setup();
        let quad = CatmarkQuad::<f64, _, 2>::new(gauss_quad, 1);

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
        let quad = CatmarkQuad::<f64, _, 2>::new(gauss_quad, 1);

        let weights = quad.weights_ref_irregular();
        assert_eq!(weights.len(), 12);
        assert!(weights.iter().all(|&wi| abs_diff_eq!(wi, 0.125, epsilon = 1e-13)));
    }
}