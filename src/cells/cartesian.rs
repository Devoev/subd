use crate::cells::traits::{Cell, CellConnectivity, ToElement};
use crate::mesh::cartesian::MultiBreaks;
use crate::mesh::traits::VertexStorage;
use nalgebra::{Const, DimName, DimNameSub, RealField};
use crate::element::cartesian::CartCell;

/// Multi-index of a [`CartCell`].
#[derive(Debug, Clone, Copy)]
pub struct CartCellIdx<const K: usize>(pub [usize; K]);

impl<const K: usize> CartCellIdx<K> {
    /// Returns the underlying multi-index representing `self`.
    pub fn as_index(&self) -> &[usize; K] {
        &self.0
    }
}

impl <const K: usize> Cell for CartCellIdx<K> {
    type Dim = Const<K>;
    type Node = [usize; K];

    fn nodes(&self) -> &[Self::Node] {
        todo!("Implement by iterating over all 2^D vertices")
    }
}

impl <T: RealField + Copy, const K: usize> ToElement<T, Const<K>> for CartCellIdx<K> {
    type Elem = CartCell<T, K>;
    type Coords = MultiBreaks<T, K>;

    fn to_element(&self, coords: &Self::Coords) -> Self::Elem {
        let idx_a = *self.as_index();
        let idx_b = idx_a.map(|i| i + 1); // todo: implement this in the multi-index trait
        let a = coords.vertex(idx_a);
        let b = coords.vertex(idx_b);
        CartCell::new(a, b)
    }
}

impl <const K: usize> CellConnectivity for CartCellIdx<K> {
    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        Const<K>: DimNameSub<M>
    {
        todo!()
    }
}

// impl CellBoundary for CartCellIdx<3> {
//     const NUM_SUB_CELLS: usize = 6;
//     type SubCell = CartCellIdx<2>;
//     type Boundary = ();
//
//     fn boundary(&self) -> Self::Boundary {
//         todo!("Get")
//     }
// }