//! Custom quadrature rules and extensions to the gauss_quad crate.

use crate::subd::precompute::{QuadEval};
use crate::subd::surface::Jacobian;
use gauss_quad::legendre::GaussLegendreError;
use gauss_quad::{GaussLegendre, Node, Weight};
use itertools::Itertools;
use nalgebra::RealField;
use num_traits::ToPrimitive;
use std::iter::zip;

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

    /// Numerically integrates the function `f: (0,1)² ⟶ ℝ` on the parametric domain.
    ///
    /// The values of `f` evaluated at the nodes are given as a flattened vector `fij = f(xi,xj)`.
    pub fn integrate(&self, f: Vec<f64>) -> f64 {
        zip(f, self.weights())
            .map(|(fij, (wi, wj))| fij * wi * wj)
            .sum::<f64>() * Self::SCALE_FACTOR * Self::SCALE_FACTOR
    }

    /// Numerically integrates the pullback function `f: (0,1)² ⟶ ℝ` on a patch.
    /// 
    /// The values of `f` are given as in [`GaussLegendrePatch::integrate`].
    /// The surface patch is parametrized by the Jacobian matrices `jacobian_eval`.
    pub fn integrate_pullback<T: RealField + Copy + ToPrimitive>(&self, f: Vec<T>, jacobian_eval: &QuadEval<T, Jacobian<T>>) -> T {
        // todo:
        //  - change signature, especially for f. For example, use an IgaFn instead of f

        let integrand = zip(f, jacobian_eval.abs_det())
            .map(|(fi, det_i)| (fi * det_i).to_f64().unwrap())
            .collect();

        T::from_f64(
            self.integrate(integrand)
        ).unwrap()
    }
}