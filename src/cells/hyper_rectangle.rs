use crate::cells::cell::{Cell, CellBoundary};
use crate::cells::vertex::VertexTopo;
use crate::knots::index::MultiIndex;
use nalgebra::{Const, DimNameSub, U3};

// todo: index type is incorrect
//  - nodes and vols should have one multi index
//  - edges and faces should have 3 ?

#[derive(Debug, Clone, Copy)]
pub struct HyperRectangle<const K: usize>(pub MultiIndex<usize, K>);

impl <const K: usize> Cell<Const<K>> for HyperRectangle<K> {
    fn nodes(&self) -> &[VertexTopo] {
        todo!()
    }

    fn is_connected<const M: usize>(&self, other: &Self) -> bool
    where
        Const<K>: DimNameSub<Const<M>>
    {
        todo!()
    }
}

type Face = HyperRectangle<3>;

type Edge = HyperRectangle<2>;

type Node = HyperRectangle<1>;

impl CellBoundary<U3> for HyperRectangle<3> {
    const NUM_SUB_CELLS: usize = 6;
    type SubCell = HyperRectangle<2>;
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary {
        todo!("Get")
    }
}