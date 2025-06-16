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
      Q: Quadrature<T, E::RefCell, D, Node=X>
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

impl <T, X, E, Q, const D: usize> Quadrature<T, E, D> for PullbackQuad<T, X, E, Q, D>
where T: RealField + Sum + Product + Copy,
      X: Dimensioned<T, D>,
      E: Cell<T, X, D, D>,
      Q: Quadrature<T, E::RefCell, D, Node=X>,
      Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    type Node = Point<T, D>;

    fn nodes_elem(&self, elem: &E) -> impl Iterator<Item=Self::Node> {
        // todo: remove collect
        let res = self
            .nodes_ref(&elem.ref_cell())
            .map(|xi| Point::from(elem.geo_map().eval(xi)))
            .collect_vec();
        res.into_iter()
    }

    fn weights_elem(&self, elem: &E) -> impl Iterator<Item=T> {
        // todo: remove collect
        let ref_elem = elem.ref_cell();
        let res = zip(self.weights_ref(&ref_elem), self.nodes_ref(&ref_elem))
            .map(|(wi, xi)| {
                let d_phi = elem.geo_map().eval_diff(xi);
                wi * d_phi.determinant().abs()
            })
            .collect_vec();
        res.into_iter()
    }
}