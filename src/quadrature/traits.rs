use crate::cells::geo::Cell;
use nalgebra::{Point, RealField};
use std::iter::{zip, Sum};
use std::ops::Mul;
use crate::index::dimensioned::Dimensioned;
// todo: possibly add RefCell and parametrize RefQuadrature?

/// Performs the numerical integration by evaluating the sum
/// ```text
/// I = ∑ w[i] f(x[i])
/// ```
/// where the weights `w` and the evaluated function `f` at the quadrature nodes are given.
fn integrate_with_weights<T: Sum, W: Mul<T, Output=T>>(w: impl IntoIterator<Item = W>, f: impl IntoIterator<Item = T>) -> T {
    zip(w, f).map(|(w, f)| w * f).sum::<T>()
}

// todo: replace Node associated type with generic, such that different impls for tuples and arrays can work

/// Quadrature rule on a [`D`]-dimensional element of type [`E`].
pub trait Quadrature<T, Node, Elem, Weight = T> where T: Sum, Weight: Mul<T, Output=T> {
    /// Returns an iterator over all quadrature nodes in the given `elem`.
    fn nodes_elem(&self, elem: &Elem) -> impl Iterator<Item = Node>;
    
    /// Returns an iterator over all quadrature weights in the given `elem`.
    fn weights_elem(&self, elem: &Elem) -> impl Iterator<Item = Weight>;

    /// Numerically integrates a function `f` on the given `elem`.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes_elem`]
    /// are given as `f[i] = f(x[i])`.
    fn integrate_elem(&self, elem: &Elem, f: impl IntoIterator<Item = T>) -> T {
        integrate_with_weights(self.weights_elem(elem), f)
    }

    /// Evaluates the function `f` on every [quadrature node][Self::nodes_elem]
    /// of the given `elem`.
    fn eval_fn_elem(&self, elem: &Elem, f: impl Fn(Node) -> T) -> impl Iterator<Item = T> {
        self.nodes_elem(elem).map(f)
    }

    /// Numerically integrates a function `f` on the given `elem`,
    /// by evaluating the function at every quadrature point using [`Self::eval_fn_elem`].
    /// The actual quadrature is then performed using [Self::integrate_ref].
    fn integrate_fn_elem(&self, elem: &Elem, f: impl Fn(Node) -> T) -> T {
        self.integrate_elem(elem, self.eval_fn_elem(elem, f))
    }
}