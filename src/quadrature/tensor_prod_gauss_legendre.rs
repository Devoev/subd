use gauss_quad::legendre::GaussLegendreError;
use gauss_quad::{GaussLegendre, Node};
use itertools::{Itertools, MultiProduct};
use nalgebra::RealField;
use num_traits::FromPrimitive;
use std::iter::{zip, Product, Sum};
use std::vec::IntoIter;
use crate::cells::hyper_rectangle::HyperRectangle;
// todo: this currently only works on [0,1].
//  Must be adjusted to arbitrary intervals! Important for Bezier Elements

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
            nodes: quad.nodes().map(|&n| T::from_f64(n).unwrap()).collect(),
            weights: quad.weights().map(|&w| T::from_f64(w).unwrap()).collect(),
        })
    }
}

impl<T: RealField + Copy + Product<T> + Sum<T>> TensorProdGaussLegendre<T> {
    /// Scale factor for the weights for each parametric direction.
    const SCALE_FACTOR: f64 = 0.5;

    /// Transforms the interval `[-1,1]` to `[a, b]`.
    fn transform_coord(&self, x: T, a: T, b: T) -> T {
        ((b - a) * x + (b + a)) / T::from_usize(2).unwrap()
    }
    
    /// Transforms the cube `[-1,1]^D` to the hyper rectangle `elem`.
    fn transform_coords<const D: usize>(&self, x: Vec<T>, elem: HyperRectangle<T, D>) -> [T; D]{
        x.iter().enumerate()
            .map(|(i, &xi)| self.transform_coord(xi, elem.a[i], elem.b[i]))
            .collect_array()
            .unwrap()
    }

    /// Returns an iterator over the weights of this quadrature rule, in lexicographical order.
    pub fn weights<const D: usize>(&self) -> MultiProduct<IntoIter<T>> {
        let weights = (0..D).map(|_| self.weights.clone());
        weights.multi_cartesian_product()
    }

    /// Returns an iterator over the products of all weights for a single `D`-dimensional node.
    pub fn weight_products<const D: usize>(&self) -> impl Iterator<Item = T> {
        self.weights::<D>().map(|wi| wi.into_iter().product())
    }

    /// Returns an iterator over the quadrature nodes for the reference cube `[-1,1]^D`, 
    /// in lexicographical order.
    pub fn nodes_ref<const D: usize>(&self) -> MultiProduct<IntoIter<T>> {
        let nodes = (0..D).map(|_| self.nodes.clone());
        nodes.multi_cartesian_product()
    }
    
    /// Returns an iterator over the quadrature nodes for the hyper rectangle `elem`,
    /// in lexicographical order.
    pub fn nodes<const D: usize>(&self, elem: HyperRectangle<T, D>) -> impl Iterator<Item = [T; D]> + '_ {
        self.nodes_ref::<D>().map(move |xi| self.transform_coords(xi, elem))
    }
    
    /// Evaluates the function `f` on every [quadrature node][Self::nodes] of the given `elem`.
    pub fn eval_fn<'a, const D: usize>(&'a self, f: impl Fn([T; D]) -> T + 'a, elem: HyperRectangle<T, D>) -> impl Iterator<Item = T> + 'a {
        self.nodes::<D>(elem).map(move |n: [T; D]| f(n))
    }

    /// Numerically integrates a function `f` on a `D`-dimensional hyper rectangle.
    /// The values of the function evaluated at the [quadrature nodes][`Self::nodes`]
    /// are given as `f[i] = f(x[i,1],...,x[i,D])` in lexicographical order.
    pub fn integrate<const D: usize>(&self, f: impl IntoIterator<Item = T>) -> T {
        let scale = T::from_f64(Self::SCALE_FACTOR).unwrap().powi(D as i32);
        zip(f, self.weight_products::<D>())
            .map(|(fi, wi)| fi * wi)
            .sum::<T>() * scale
    }

    /// Numerically integrates a function `f` on the `D`-dimensional hyper rectangle `elem`,
    /// by evaluating the function at every using [`Self::eval_fn`].
    /// The actual quadrature is then performed using [Self::integrate].
    pub fn integrate_fn<const D: usize>(&self, f: impl Fn([T; D]) -> T, elem: HyperRectangle<T, D>) -> T {
        self.integrate::<D>(self.eval_fn(f, elem))
    }
}