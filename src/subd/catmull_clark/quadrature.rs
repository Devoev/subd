use std::iter::{Product, Sum};
use crate::quadrature::pullback::PullbackQuad;
use crate::subd::catmull_clark::patch::CatmarkPatch;
use nalgebra::{Point2, RealField};
use num_traits::ToPrimitive;
use crate::cells::unit_cube::UnitCube;
use crate::quadrature::traits::Quadrature;

/// Quadrature rule on [Catmull-Clark patches](CatmarkPatch).
struct CatmarkQuad<T: RealField, Q, const M: usize> {
    /// Regular quadrature rule.
    quad_reg: PullbackQuad<T, (T, T), CatmarkPatch<T, M>, Q, 2>,

    /// Maximum number of sub-rings to evaluate during quadrature of an irregular patch.
    m_max: usize,
}

impl <T: RealField, Q, const M: usize> CatmarkQuad<T, Q, M> {
    /// Constructs a new [`CatmarkQuad`] from the given
    /// quadrature rule `ref_quad` on the reference domain.
    pub fn new(ref_quad: Q, m_max: usize) -> Self {
        CatmarkQuad { quad_reg: PullbackQuad::new(ref_quad), m_max }
    }
}

impl <T, Q> CatmarkQuad<T, Q, 2>
    where T: RealField + Sum + Product + Copy + ToPrimitive,
          Q: Quadrature<T, (T, T), UnitCube<2>>
{
    /// Returns an iterator over all nodes in the regular reference domain (unit square).
    pub fn nodes_ref_regular(&self) -> impl Iterator<Item = (T, T)> + '_ {
        self.quad_reg.nodes_ref(&UnitCube)
    }

    /// Returns an iterator over all weights in the regular reference domain (unit square).
    pub fn weights_ref_regular(&self) -> impl Iterator<Item=T> + '_ {
        self.quad_reg.weights_ref(&UnitCube)
    }

    /// Returns an iterator over all nodes in the irregular reference domain.
    pub fn nodes_ref_irregular(&self) {
        todo!("Subdivide UnitCube into 3 regular quads (L-shape) and one irregular quad.")
    }
}

impl <T, Q> Quadrature<T, Point2<T>, CatmarkPatch<T, 2>> for CatmarkQuad<T, Q, 2>
where T: RealField + Sum + Product + Copy + ToPrimitive,
      Q: Quadrature<T, (T, T), UnitCube<2>>
{
    fn nodes_elem(&self, elem: &CatmarkPatch<T, 2>) -> impl Iterator<Item=Point2<T>> {
        match elem {
            CatmarkPatch::Irregular(_, _) => {
                todo!("This can be implemented later, since only the reference nodes are usually needed")
            }
            _ => self.quad_reg.nodes_elem(elem),
        }
    }

    fn weights_elem(&self, elem: &CatmarkPatch<T, 2>) -> impl Iterator<Item=T> {
        match elem {
            CatmarkPatch::Irregular(_, _) => {
                todo!("Implement nodes_ref_irregular and add weights_ref_irregular")
            }
            _ => self.quad_reg.weights_elem(elem),
        }
    }
}