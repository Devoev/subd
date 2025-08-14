use crate::cells::geo::Cell;
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use crate::quadrature::traits::Quadrature;
use itertools::Itertools;
use nalgebra::{Const, DimMin, Point, RealField};
use std::iter::{zip, Product, Sum};
use std::marker::PhantomData;
use crate::cells::bezier_elem::BezierElem;
use crate::quadrature::tensor_prod::GaussLegendreMulti;

// todo: possibly rename and add docs

/// Quadrature rule on an element.
/// Integration is performed by pulling back the function to the reference domain.
#[derive(Clone, Debug)]
pub struct PullbackQuad<T, X, E, Q, const D: usize> {
    /// Quadrature rule on the reference domain.
    ref_quad: Q,

    _phantom: PhantomData<(T, X, E)>,
}

/// Quadrature rule on [Bezier elements](BezierElem).
pub type BezierQuad<'a, T, const D: usize> = PullbackQuad<T, [T; D], BezierElem<'a, T, D, D>, GaussLegendreMulti<T, D>, D>;

impl <T, X, E, Q, const D: usize> PullbackQuad<T, X, E, Q, D> {
    /// Constructs a new [`PullbackQuad`] from the given
    /// quadrature rule `ref_quad` on the reference domain.
    pub fn new(ref_quad: Q) -> Self {
        PullbackQuad { ref_quad, _phantom: PhantomData }
    }
}

impl <T, X, E, Q, const D: usize>  PullbackQuad<T, X, E, Q, D>
where T: RealField + Sum + Product + Copy,
      X: Dimensioned<T, D>,
      E: Cell<T, X, D, D>,
      Q: Quadrature<T, X, E::RefCell>
{
    /// Returns an iterator over all nodes in the reference domain.
    pub fn nodes_ref<'a>(&'a self, ref_elem: &'a E::RefCell) -> impl Iterator<Item = X> + 'a {
        self.ref_quad.nodes_elem(ref_elem)
    }

    /// Returns an iterator over all weights in the reference domain.
    pub fn weights_ref<'a>(&'a self, ref_elem: &'a E::RefCell) -> impl Iterator<Item = T> + 'a {
        self.ref_quad.weights_elem(ref_elem)
    }
}

impl <T, X, E, Q, const D: usize> Quadrature<T, Point<T, D>, E> for PullbackQuad<T, X, E, Q, D>
where T: RealField + Sum + Product + Copy,
      X: Dimensioned<T, D>,
      E: Cell<T, X, D, D>,
      Q: Quadrature<T, X, E::RefCell>,
      Const<D>: DimMin<Const<D>, Output = Const<D>>
{

    fn nodes_elem(&self, elem: &E) -> impl Iterator<Item=Point<T, D>> {
        let res = self
            .nodes_ref(&elem.ref_cell())
            .map(|xi| elem.geo_map().eval(xi))
            .collect_vec(); // todo: remove collect
        res.into_iter()
    }

    fn weights_elem(&self, elem: &E) -> impl Iterator<Item=T> {
        let ref_elem = elem.ref_cell();
        let geo_map = elem.geo_map();
        let res = zip(self.weights_ref(&ref_elem), self.nodes_ref(&ref_elem))
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
    use approx::assert_abs_diff_eq;
    use gauss_quad::GaussLegendre;
    use nalgebra::{matrix, point, Point2};
    use crate::bspline::space::BsplineSpace;
    use crate::bspline::spline_geo::SplineGeo;
    use crate::cells::cartesian::CartCell;
    use super::*;

    /// Returns a 2D Gauss-Legendre quadrature with degree `2`
    /// in both `x`-direction and `y`-direction.
    fn setup() -> GaussLegendreMulti<f64, 2> {
        GaussLegendreMulti::new([GaussLegendre::new(2).unwrap(), GaussLegendre::new(2).unwrap()])
    }

    #[test]
    fn nodes_elem() {
        let ref_quad = setup();

        // Test flat element [-1,1]^2
        let quad = PullbackQuad::new(ref_quad.clone());
        let cell = CartCell::new(point![-1.0, -1.0], point![1.0, 1.0]);
        let nodes: Vec<Point2<f64>> = quad.nodes_elem(&cell).collect();
        assert_abs_diff_eq!(nodes[0], point![0.57735, 0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[1], point![0.57735, -0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[2], point![-0.57735, 0.57735], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[3], point![-0.57735, -0.57735], epsilon = 1e-5);

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

        let nodes: Vec<Point2<f64>> = quad.nodes_elem(&bezier_elem).collect();
        assert_abs_diff_eq!(nodes[0], point![0.788675, 0.788675], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[1], point![0.788675, 0.211325], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[2], point![0.211325, 0.788675], epsilon = 1e-5);
        assert_abs_diff_eq!(nodes[3], point![0.211325, 0.211325], epsilon = 1e-5);

        // Test curved Bezier element

    }
}