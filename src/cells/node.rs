use nalgebra::{Const, DimName, DimNameSub, U0};
use crate::cells::topo::Cell;

/// Index of a node aka. vertex in a mesh. Represented by a global index.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct NodeIdx(pub usize);

impl Cell<U0> for NodeIdx {
    fn nodes(&self) -> &[NodeIdx] {
        &[] // todo: return self?
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U0: DimNameSub<M>
    {
        self == other
    }
}