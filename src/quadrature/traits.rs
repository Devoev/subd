use std::iter::{zip, Sum};
use std::ops::Mul;

/// Performs the numerical integration by evaluating the sum
/// ```text
/// I = ∑ w[i] f(x[i])
/// ```
/// where the weights `w` and the evaluated function `f` at the quadrature nodes are given.
fn integrate_with_weights<T: Sum, W: Mul<T, Output=T>>(w: impl IntoIterator<Item = W>, f: impl IntoIterator<Item = T>) -> T {
    zip(w, f).map(|(w, f)| w * f).sum::<T>()
}

/// Quadrature rule for the [numerical integration](https://en.wikipedia.org/wiki/Numerical_integration)
/// of a function, using node-weights pairs.
/// 
/// The integral of a function `f` is approximated as the weighted sum
/// ```text
/// ∫ f dx ≅ ∑ w[i] f(x[i])
/// ```
/// where `w[i]` are the weights and `x[i]` the nodes.
/// 
/// # How to use a quadrature rule
/// The `Quadrature` trait provides the method [`integrate_fn_elem`](Quadrature::integrate_fn_elem)
/// which, given an integration domain `elem: Elem` and a function `f: Node -> T`,
/// performs the numerical integration of `f` on `elem`. This is done by evaluating the function
/// on every quadrature node, given by [`nodes_elem`](Quadrature::nodes_elem), and multiplying
/// the evaluated function values with the weights, given by [`weights_elem`](Quadrature::weights_elem).
/// 
/// For efficiency reasons it may make sense to evaluate the function apriori,
/// and then perform the numerical integration. This can be done by evaluating the function
/// using [`eval_fn_elem`](Quadrature::eval_fn_elem) on every quadrature node
/// (or simply getting the nodes using `nodes_elem` and performing the evaluation yourself) 
/// and then calling [`integrate_elem`](Quadrature::integrate_elem)
/// to perform the actual integration.
/// 
/// # Type parameters
/// - [`T`]: Scalar type.
/// - [`Elem`]: Type of integration domain.
/// - [`Node`]: Type of quadrature nodes.
/// - [`Weight`]: Type of quadrature weight. By default, equal to `T`.
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