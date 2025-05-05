use crate::mesh::cell::CellTopo;
use nalgebra::{Const, DimNameSub, U0};

/// Topology of a vertex in a mesh. Represented by a global index.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VertexTopo(pub usize);

impl CellTopo<U0> for VertexTopo {
    fn nodes(&self) -> &[VertexTopo] {
        &[] // todo: return self?
    }

    fn is_connected<const M: usize>(&self, other: &Self) -> bool
    where
        U0: DimNameSub<Const<M>>
    {
        self == other
    }
}