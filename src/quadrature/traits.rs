use std::iter::{zip, Sum};
use std::ops::Mul;
use nalgebra::{DefaultAllocator, Scalar};
use crate::cells::geo::{Cell, CellCoord};
use crate::diffgeo::chart::{ChartAllocator};

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
/// performs the numerical integration of `f` on `elem`.
/// For example:
/// ```
/// # use approx::assert_relative_eq;
/// # use gauss_quad::GaussLegendre;
/// # use subd::cells::cartesian::CartCell;
/// # use subd::quadrature::traits::Quadrature;
///
/// // Construct quadrature rule
/// let degree = 3;
/// let quad = GaussLegendre::new(degree).unwrap();
///
/// // Define function and interval
/// let interval = CartCell::new_univariate(-2.0, 2.0);
/// let f = |x: f64| x.powi(3);
///
/// // Perform numerical integration
/// let int = quad.integrate_fn_elem(&interval, f);
/// assert_relative_eq!(int, 0.0, epsilon = 1e-13);
/// ```
/// The quadrature evaluates the function
/// on every quadrature node, given by [`nodes_elem`](Quadrature::nodes_elem), and multiplies
/// the evaluated function values with the weights, given by [`weights_elem`](Quadrature::weights_elem).
///
/// For efficiency reasons it may make sense to evaluate the function apriori,
/// and then perform the numerical integration.
/// This can be done using [`integrate_elem`](Quadrature::integrate_elem),
/// for example
/// ```
/// # use approx::assert_relative_eq;
/// # use gauss_quad::GaussLegendre;
/// # use itertools::Itertools;
/// # use subd::cells::cartesian::CartCell;
/// # use subd::quadrature::traits::Quadrature;
///
/// # let degree = 3;
/// # let quad = GaussLegendre::new(degree).unwrap();
/// # let interval = CartCell::new_univariate(-2.0, 2.0);
/// # let f = |x: f64| x.powi(3);
///
/// // Evaluate function at quadrature nodes
/// // You can also call `quad.eval_fn_elem` here
/// let fi = quad.nodes_elem(&interval).map(|node| f(node));
///
/// // Perform numerical integration
/// let int = quad.integrate_elem(&interval, fi);
/// assert_relative_eq!(int, 0.0, epsilon = 1e-13);
/// ```
/// Alternatively you could also use [`eval_fn_elem`](Quadrature::eval_fn_elem)
/// to evaluate the function on every quadrature node.
/// 
/// # Type parameters
/// A quadrature rule is parametrized by the types
/// - [`T`]: Scalar type.
/// - [`Elem`]: Type of integration domain.
///
/// That way a quadrature rule can be implemented once to work for different scalars
/// and on different elements.
pub trait Quadrature<T: Sum, Elem> {
    /// Quadrature node.
    type Node;

    /// Quadrature weight.
    type Weight: Mul<T, Output=T>;

    /// Returns an iterator over all quadrature nodes in the given `elem`.
    fn nodes_elem(&self, elem: &Elem) -> impl Iterator<Item = Self::Node>;
    
    /// Returns an iterator over all quadrature weights in the given `elem`.
    fn weights_elem(&self, elem: &Elem) -> impl Iterator<Item = Self::Weight>;

    /// Numerically integrates a function `f` on the given `elem`.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes_elem`]
    /// are given as `f[i] = f(x[i])`.
    fn integrate_elem(&self, elem: &Elem, f: impl IntoIterator<Item = T>) -> T {
        integrate_with_weights(self.weights_elem(elem), f)
    }

    /// Evaluates the function `f` on every [quadrature node][Self::nodes_elem]
    /// of the given `elem`.
    fn eval_fn_elem(&self, elem: &Elem, f: impl Fn(Self::Node) -> T) -> impl Iterator<Item = T> {
        self.nodes_elem(elem).map(f)
    }

    /// Numerically integrates a function `f` on the given `elem`,
    /// by evaluating the function at every quadrature point using [`Self::eval_fn_elem`].
    /// The actual quadrature is then performed using [Self::integrate_ref].
    fn integrate_fn_elem(&self, elem: &Elem, f: impl Fn(Self::Node) -> T) -> T {
        self.integrate_elem(elem, self.eval_fn_elem(elem, f))
    }
}

// todo: can't this be merged with Quadrature?
//  if the methods of Cell go to Chart, merging won't work,
//  because Cell has no information about the Coord anymore

/// Constrains `Self` to be a quadrature on [`C::ParametricCell`]
/// with the coordinates [`C::GeoMap::Coord`] of the chart.
pub trait QuadratureOnParametricCell<T: Scalar + Sum, C: Cell<T>>: Quadrature<T, C::ParametricCell, Node = CellCoord<T, C>>
where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}

impl <T: Scalar + Sum, C: Cell<T>, Q: Quadrature<T, C::ParametricCell, Node = CellCoord<T, C>>> QuadratureOnParametricCell<T, C> for Q
where DefaultAllocator: ChartAllocator<T, C::GeoMap>
{}