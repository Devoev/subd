use crate::cells::geo::Cell;
use crate::diffgeo::chart::Chart;
use crate::quadrature::tensor_prod::GaussLegendreMulti;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::{Const, DimMin, Point, RealField, SquareMatrix};
use std::iter::{zip, Product, Sum};

// todo: possibly rename and add docs

/// Quadrature rule on an element.
/// Integration is performed by pulling back the function to the reference domain.
#[derive(Clone, Debug)]
pub struct PullbackQuad<Q, const D: usize> {
    /// Quadrature rule on the reference domain.
    ref_quad: Q,
}

/// Pullback version of [`GaussLegendreMulti`].
pub type GaussLegendrePullback<const D: usize> = PullbackQuad<GaussLegendreMulti<D>, D>;

impl <Q, const D: usize> PullbackQuad<Q, D> {
    /// Constructs a new [`PullbackQuad`] from the given
    /// quadrature rule `ref_quad` on the reference domain.
    pub fn new(ref_quad: Q) -> Self {
        PullbackQuad { ref_quad }
    }
}

impl <Q, const D: usize>  PullbackQuad<Q, D> {
    /// Returns an iterator over all nodes in the reference domain.
    pub fn nodes_ref<'a, T, E>(&'a self, ref_elem: &'a E::RefCell) -> impl Iterator<Item = Q::Node> + 'a
    where T: RealField + Sum + Product + Copy,
          E: Cell<T, D, D>,
          Q: Quadrature<T, E::RefCell>
    {
        self.ref_quad.nodes_elem(ref_elem)
    }

    /// Returns an iterator over all weights in the reference domain.
    pub fn weights_ref<'a, T, E>(&'a self, ref_elem: &'a E::RefCell) -> impl Iterator<Item = Q::Weight> + 'a
    where T: RealField + Sum + Product + Copy,
          E: Cell<T, D, D>,
          Q: Quadrature<T, E::RefCell>
    {
        self.ref_quad.weights_elem(ref_elem)
    }
}

// todo: should this trait be really used? maybe hide behind other trait?
/// Constrains that the [`DimMin::min`] of `Self` with `Self` is `Self`,
/// i.e. `DimMin<Self, Output = Self>`.
///
/// This is required for example in [`SquareMatrix::determinant`].
pub trait DimMinSelf: DimMin<Self, Output = Self> {}

impl <D: DimMin<Self, Output = Self>> DimMinSelf for D {}

impl <T, E, Q, const D: usize> Quadrature<T, E> for PullbackQuad<Q, D>
where T: RealField + Sum + Product + Copy,
      E: Cell<T, D, D>,
      E::GeoMap: Chart<T, D, D>,
      Q: Quadrature<T, E::RefCell, Node = <E::GeoMap as Chart<T, D, D>>::Coord>,
      Const<D>: DimMinSelf
{
    type Node = Point<T, D>;
    type Weight = T;

    fn nodes_elem(&self, elem: &E) -> impl Iterator<Item=Point<T, D>> {
        let res = self
            .nodes_ref::<T, E>(&elem.ref_cell())
            .map(|xi| elem.geo_map().eval(xi))
            .collect_vec(); // todo: remove collect
        res.into_iter()
    }

    fn weights_elem(&self, elem: &E) -> impl Iterator<Item=T> {
        let ref_elem = elem.ref_cell();
        let geo_map = elem.geo_map();
        let res = zip(self.weights_ref::<T, E>(&ref_elem), self.nodes_ref::<T, E>(&ref_elem))
            .map(|(wi, xi)| {
                let d_phi = geo_map.eval_diff(xi);
                wi * d_phi.determinant().abs()
            })
            .collect_vec(); // todo: remove collect
        res.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bspline::space::BsplineSpace;
    use crate::bspline::spline_geo::SplineGeo;
    use crate::cells::cartesian::CartCell;
    use approx::assert_abs_diff_eq;
    use gauss_quad::GaussLegendre;
    use nalgebra::{matrix, point};
    use crate::cells::bezier_elem::BezierElem;

    /// Returns a 2D Gauss-Legendre quadrature with degree `2`
    /// in both `x`-direction and `y`-direction.
    fn setup() -> GaussLegendreMulti<2> {
        GaussLegendreMulti::new([GaussLegendre::new(2).unwrap(), GaussLegendre::new(2).unwrap()])
    }

    #[test]
    fn nodes_elem() {
        let ref_quad = setup();

        // Test flat element [-1,1]^2
        let quad = PullbackQuad::new(ref_quad.clone());
        let cell = CartCell::new(point![-1.0, -1.0], point![1.0, 1.0]);

        let mut nodes = quad.nodes_elem(&cell);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.57735, 0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.57735, -0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![-0.57735, 0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![-0.57735, -0.57735], epsilon = 1e-5);
        assert_eq!(nodes.next(), None);

        // Test flat Bezier element
        let space_geo = BsplineSpace::new_open_uniform([2, 2], [1, 1]);
        let control_points = matrix![
            0.0, 0.0;
            1.0, 0.0;
            0.0, 1.0;
            1.0, 1.0
        ];
        let map = SplineGeo::from_matrix(control_points, &space_geo).expect("Failed to B-Spline mapping");

        let quad = PullbackQuad::new(ref_quad);
        let cell = CartCell::new(point![0.0, 0.0], point![1.0, 1.0]);
        let bezier_elem = BezierElem::new(cell, &map);

        let mut nodes= quad.nodes_elem(&bezier_elem);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.788675, 0.788675], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.788675, 0.211325], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.211325, 0.788675], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes.next().unwrap(), point![0.211325, 0.211325], epsilon = 1e-5);
        assert_eq!(nodes.next(), None);

        // Test curved Bezier element
        // todo
    }
}