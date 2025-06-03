use std::iter::Sum;
use std::marker::PhantomData;
use nalgebra::{Const, Point, RealField};
use crate::cells::chart::Chart;
use crate::cells::geo::Cell;
use crate::quadrature::traits::{Quadrature, RefQuadrature};

// todo: this struct isn't yet used. Maybe replace the ElementQuadrature trait with this

/// Quadrature rule on an element.
pub struct ElemQuad<T, Q, E, const D: usize> {
    /// Quadrature rule on the reference domain.
    pub ref_quad: Q,

    _phantoms: PhantomData<(T, E)>,
}

impl<T, Q, E, const D: usize> ElemQuad<T, Q, E, D>
    where T: RealField + Sum,
          Q: RefQuadrature<T>,
          E: Cell<T, Q::Node, Const<D>, D>
{
    /// Returns an iterator over all quadrature nodes in the given `elem`.
    fn nodes_elem(&self, elem: &E) -> impl Iterator<Item = Point<T, D>> + '_ {
        let geo_map = elem.geo_map();
        self.ref_quad.nodes_ref().map(move |n| geo_map.eval(n))
    }

    /// Evaluates the function `f` on every [quadrature node][Self::nodes_elem]
    /// of the given `elem`.
    fn eval_fn_elem<'a>(&'a self, elem: &E, f: impl Fn(Point<T, D>) -> T + 'a) -> impl Iterator<Item = T> + 'a {
        self.nodes_elem(elem).map(f)
    }

    /// Numerically integrates a function `f` on the given `elem`,
    /// by evaluating the function at every quadrature point using [`Self::eval_fn_elem`].
    /// The actual quadrature is then performed using [Self::integrate_ref].
    fn integrate_fn_elem(&self, elem: &E, f: impl Fn(Point<T, D>) -> T) -> T {
        self.ref_quad.integrate_ref(self.eval_fn_elem(elem, f))
        // todo: this is incorrect. Jacobian determinant needs to be computed as well for integration
    }
}

impl<T, Q, E, const D: usize> RefQuadrature<T> for ElemQuad<T, Q, E, D>
    where T: RealField + Sum,
          Q: RefQuadrature<T>,
          E: Cell<T, Q::Node, Const<D>, D>
{
    type Node = Q::Node;

    fn nodes_ref(&self) -> impl Iterator<Item=Self::Node> {
        self.ref_quad.nodes_ref()
    }

    fn weights_ref(&self) -> impl Iterator<Item=T> {
        self.ref_quad.weights_ref()
    }
}

impl <T, Q, E, const D: usize> Quadrature<T, D> for ElemQuad<T, Q, E, D>
    where T: RealField + Sum,
          Q: RefQuadrature<T>,
          E: Cell<T, Q::Node, Const<D>, D>
{
    type Elem = E;
}
