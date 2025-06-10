use crate::cells::bezier_elem::BezierElem;
use crate::index::dimensioned::Dimensioned;
use crate::quadrature::tensor_prod::GaussLegendreMulti;
use crate::quadrature::traits::{Quadrature, RefQuadrature};
use nalgebra::{Const, DimMin, Point, RealField};
use std::iter::{empty, zip, Product, Sum};
use std::marker::PhantomData;
use crate::diffgeo::chart::Chart;

/// Quadrature rule for a [`BezierElem`].
#[derive(Clone, Debug)]
pub struct BezierQuad<'a, T, const D: usize> {
    /// Gauss-Legendre quadrature rule on the reference domain.
    ref_quad: GaussLegendreMulti<T, D>,

    _phantom: PhantomData<&'a ()>,
}

impl <'a, T, const D: usize> BezierQuad<'a, T, D> {
    /// Constructs a new [`BezierQuad`] from the given `ref_quad`.
    pub fn new(ref_quad: GaussLegendreMulti<T, D>) -> Self {
        BezierQuad { ref_quad, _phantom: PhantomData }
    }
}

impl <'a, T, const D: usize> RefQuadrature<T> for BezierQuad<'a, T, D>
    where T: RealField + Sum + Product + Copy,
{
    type Node = [T; D];

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        empty()
    }

    fn weights_ref(&self) -> impl Iterator<Item=T> {
        empty()
    }
}

impl <'a, T, const D: usize> Quadrature<T, D> for BezierQuad<'a, T, D>
    where T: RealField + Sum + Product + Copy,
          Const<D>: DimMin<Const<D>, Output = Const<D>>
{
    type Elem = BezierElem<'a, T, D, D>;

    fn nodes_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=Point<T, D>> {
        self.ref_quad
            .nodes_elem(&elem.ref_elem)
            .map(|xi| Point::from(elem.geo_map.eval(xi.into_arr())))
    }

    fn weights_elem(&self, elem: &Self::Elem) -> impl Iterator<Item=T> {
        zip(self.ref_quad.weights_elem(&elem.ref_elem), self.ref_quad.nodes_elem(&elem.ref_elem))
            .map(move |(wi, xi)| {
                let d_phi = elem.geo_map.eval_diff(xi.into_arr());
                wi * d_phi.determinant().abs()
            })
    }
}