//! Custom quadrature rules and extensions to the gauss_quad crate.

use std::iter::zip;
use gauss_quad::{GaussLegendre, Node, Weight};
use gauss_quad::legendre::GaussLegendreError;
use itertools::Itertools;

/// Gauss legendre quadrature for a parametric patch `[0,1]²`.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussLegendrePatch {
    node_weight_pairs: Vec<(Node, Weight)>
}

impl GaussLegendrePatch {
    /// Initializes a Gauss-Legendre quadrature rule of a parametric patch of the given degree by
    /// delegating the construction to [`GaussLegendre`].
    pub fn new(deg: usize) -> Result<Self, GaussLegendreError> {
        Ok(Self {
            node_weight_pairs: GaussLegendre::new(deg)?.into_node_weight_pairs()
        })
    }

    /// Transforms the unit interval `[0,1]` to `[-1,1]`.
    pub fn transform_uv(x: f64) -> f64 {
        0.5 * (x + 1.0)
    }

    const SCALE_FACTOR: f64 = 0.5;

    /// Returns an iterator over the weights of this quadrature rule, in lexicographical order.
    pub fn weights(&self) -> impl Iterator<Item = (Weight, Weight)> + Clone + '_ {
        let weights = self.node_weight_pairs.iter().map(|&(_, w)| w);
        weights.clone().cartesian_product(weights)
    }

    /// Returns an iterator over the nodes of this quadrature rule, in lexicographical order.
    pub fn nodes(&self) -> impl Iterator<Item = (Node, Node)> + Clone + '_ {
        let nodes = self.node_weight_pairs.iter().map(|&(x, _)| Self::transform_uv(x));
        nodes.clone().cartesian_product(nodes)
    }

    /// Numerically calculates the integral of `f: (0,1)² ⟶ ℝ`.
    ///
    /// The values of `f` evaluated at the nodes are given as a vector.
    pub fn integrate(&self, f: Vec<f64>) -> f64 {
        zip(f, self.weights())
            .map(|(fij, (wi, wj))| fij * wi * wj)
            .sum::<f64>() * Self::SCALE_FACTOR * Self::SCALE_FACTOR
    }
}