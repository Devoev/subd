use crate::mesh::cell::CellTopo;
use nalgebra::U0;

/// Topology of a vertex in a mesh. Represented by a global index.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VertexTopo(pub usize);

impl CellTopo<U0> for VertexTopo {
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary { }

    fn nodes(&self) -> &[VertexTopo] {
        &[] // todo: return self?
    }
}