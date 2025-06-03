use std::iter::{zip, Sum};
use crate::cells::chart::Chart;
use crate::cells::geo::Cell;
use nalgebra::{Const, Point, RealField};

// todo: possibly add RefCell and parametrize RefQuadrature?

/// Quadrature rule on a reference domain. In 1D usually `[0,1]`.
pub trait RefQuadrature<T: RealField + Sum> {
    /// Quadrature node inside the element.
    type Node;

    // todo: maybe replace iterators with DVector's and implement quadrature by dot product

    /// Returns an iterator over all quadrature nodes on the fixed reference domain.
    fn nodes_ref(&self) -> impl Iterator<Item = Self::Node>;

    /// Returns an iterator over all quadrature weights on the fixed reference domain.
    fn weights_ref(&self) -> impl Iterator<Item = T>;

    /// Numerically integrates a function `f` on the reference domain.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes`]
    /// are given as `f[i] = f(x[i])`.
    fn integrate_ref(&self, f: impl IntoIterator<Item = T>) -> T {
        zip(self.weights_ref(), f)
            .map(|(w, f)| w * f)
            .sum::<T>()
    }

    /// Evaluates the function `f` on every [quadrature node][Self::nodes] in the reference domain.
    fn eval_fn_ref<'a>(&'a self, f: impl Fn(Self::Node) -> T + 'a) -> impl Iterator<Item = T> + 'a where T: 'a {
        self.nodes_ref().map(f)
    }

    /// Numerically integrates a function `f` on the reference domain,
    /// by evaluating the function at every quadrature point using [`Self::eval_fn_ref`].
    /// The actual quadrature is then performed using [Self::integrate_ref].
    fn integrate_fn_ref(&self, f: impl Fn(Self::Node) -> T) -> T {
        self.integrate_ref(self.eval_fn_ref(f))
    }
}

// todo: possibly replace this trait with a struct
//  the nodes_elem method doesn't work, if the reference quadrature uses a different ref domain
//  then [0,1]. For example Gauss Quad uses [-1,1]. What to do about that?

/// Quadrature rule on an element.
pub trait Quadrature<T: RealField + Sum, const D: usize>: RefQuadrature<T> {
    /// Element defining the integration domain.
    type Elem: Cell<T, Self::Node, Const<D>, D>;

    /// Returns an iterator over all quadrature nodes in the given `elem`.
    fn nodes_elem(&self, elem: &Self::Elem) -> impl Iterator<Item = Point<T, D>> {
        let geo_map = elem.geo_map();
        self.nodes_ref().map(move |n| geo_map.eval(n))
    }
    
    /// Returns an iterator over all quadrature weights in the given `elem`.
    fn weights_elem(&self, elem: &Self::Elem) -> impl Iterator<Item = T> {
        // todo: multiply by Jacobian of geo_map!
        self.weights_ref()
    }

    /// Numerically integrates a function `f` on the given `elem`.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes_elem`]
    /// are given as `f[i] = f(x[i])`.
    fn integrate_elem(&self, elem: &Self::Elem, f: impl IntoIterator<Item = T>) -> T {
        zip(self.weights_elem(elem), f)
            .map(|(w, f)| w * f)
            .sum::<T>()
    }

    /// Evaluates the function `f` on every [quadrature node][Self::nodes_elem]
    /// of the given `elem`.
    fn eval_fn_elem(&self, elem: &Self::Elem, f: impl Fn(Point<T, D>) -> T) -> impl Iterator<Item = T> {
        self.nodes_elem(elem).map(f)
    }

    /// Numerically integrates a function `f` on the given `elem`,
    /// by evaluating the function at every quadrature point using [`Self::eval_fn_elem`].
    /// The actual quadrature is then performed using [Self::integrate_ref].
    fn integrate_fn_elem(&self, elem: &Self::Elem, f: impl Fn(Point<T, D>) -> T) -> T {
        self.integrate_elem(elem, self.eval_fn_elem(elem, f))
    }
}