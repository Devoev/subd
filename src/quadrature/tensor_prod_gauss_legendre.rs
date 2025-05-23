use gauss_quad::legendre::GaussLegendreError;
use gauss_quad::{GaussLegendre, Node};
use itertools::{Itertools, MultiProduct};
use nalgebra::RealField;
use num_traits::FromPrimitive;
use std::iter::{zip, Product, Sum};
use std::vec::IntoIter;

/// Tensor product Gauss-Legendre quadrature rule.
pub struct TensorProdGaussLegendre<T> {
    /// Quadrature nodes for a parametric direction.
    nodes: Vec<T>,
    /// Quadrature weights for a parametric direction.
    weights: Vec<T>,
}

impl<T: FromPrimitive> TensorProdGaussLegendre<T> {
    // todo: maybe change this
    /// Constructs a new tensor-product Gauss-Legendre quadrature rule of the given degree by
    /// delegating the construction to [`GaussLegendre`].
    pub fn new(degree: usize) -> Result<Self, GaussLegendreError> {
        let quad = GaussLegendre::new(degree)?;
        Ok(TensorProdGaussLegendre {
            nodes: quad.nodes().map(|&n| T::from_f64(Self::transform_xi(n)).unwrap()).collect(),
            weights: quad.weights().map(|&w| T::from_f64(w).unwrap()).collect(),
        })
    }

    /// Transforms the unit interval `[0,1]` to `[-1,1]`.
    fn transform_xi(x: Node) -> Node {
        (x + 1.0) / 2.0
    }
}

impl<T: RealField + Copy + Product<T> + Sum<T>> TensorProdGaussLegendre<T> {
    /// Scale factor for the weights for each parametric direction.
    const SCALE_FACTOR: f64 = 0.5;

    /// Returns an iterator over the weights of this quadrature rule, in lexicographical order.
    pub fn weights<const D: usize>(&self) -> MultiProduct<IntoIter<T>> {
        let weights = (0..D).map(|_| self.weights.clone());
        weights.multi_cartesian_product()
    }

    /// Returns an iterator over the products of all weights for a single `D`-dimensional node.
    pub fn weight_products<const D: usize>(&self) -> impl Iterator<Item = T> {
        self.weights::<D>().map(|wi| wi.into_iter().product())
    }

    /// Returns an iterator over the nodes of this quadrature rule, in lexicographical order.
    pub fn nodes<const D: usize>(&self) -> MultiProduct<IntoIter<T>> {
        let nodes = (0..D).map(|_| self.nodes.clone());
        nodes.multi_cartesian_product()
    }

    /// Numerically integrates a function `f` on the `D`-dimensional tensor product domain.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes`]
    /// are given as `f[i] = f(x[i,1],...,x[i,D])` in lexicographical order.
    pub fn integrate<const D: usize>(&self, f: impl IntoIterator<Item = T>) -> T {
        let scale = T::from_f64(Self::SCALE_FACTOR).unwrap().powi(D as i32);
        zip(f, self.weight_products::<D>())
            .map(|(fi, wi)| fi * wi)
            .sum::<T>() * scale
    }

    /// Numerically integrates a function `f` on the `D`-dimensional tensor product domain,
    /// by evaluating the function at every [quadrature node][Self::nodes].
    /// The actual quadrature is then performed using [Self::integrate].
    pub fn integrate_fn<const D: usize>(&self, f: impl Fn([T; D]) -> T) -> T {
        let fi = self.nodes::<D>()
            .map(|n| n.try_into().unwrap())
            .map(|n: [T; D]| f(n));
        self.integrate::<D>(fi)
    }
}