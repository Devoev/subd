use crate::cells::chain::Chain;
use crate::cells::node::NodeIdx;
use nalgebra::{DimName, DimNameDiff, DimNameSub, U1, U2, U3};

/// Topology of a [`K`]-dimensional cell inside a mesh.
pub trait Cell<K: DimName> {
    /// Returns a slice of all node indices in a mesh corresponding to the corner vertices of the cell.
    fn nodes(&self) -> &[NodeIdx];

    /// Returns `true` if this cell is topologically connected (or adjacent) to the `other` cell
    /// by an [`M`]-dimensional sub-cell with `M <= K`. 
    /// That is if the two cells share a common `M`-cell (up to node ordering and orientation).
    ///
    /// # Examples
    /// For quadrilateral faces (`K = 2`) the following faces are connected
    /// ```text
    /// +---+---+
    /// |   |   |
    /// +---+---+
    /// ```
    /// by an edge (`M = 1`) and `connected_to(other, U1)` returns `true`.
    ///
    /// The faces
    /// ```text
    /// +---+
    /// |   |
    /// +---+---+
    ///     |   |
    ///     +---+
    /// ```
    /// are not connected by an edge, but by a node (`M = 0`).
    /// In this case `connected_to(other, U1)` returns `false`
    /// but `connected_to(other, U0)` returns `true`.
    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where K: DimNameSub<M>;

    /// Returns `true` if the cell contains the given `node`.
    fn contains_node(&self, node: NodeIdx) -> bool {
        self.nodes().contains(&node)
    }
    
    /// Returns `true` if `self` and `other` are topologically equal.
    /// This is the same as testing `self.is_connected(other, K::name())`.
    fn topo_eq(&self, other: &Self) -> bool
        where K: DimNameSub<K>
    {
        self.is_connected(other, K::name())
    }
}

/// A [topological cell](Cell) with an ordering of its nodes.
pub trait OrderedCell<K: DimName>: Cell<K> {
    /// Returns a (globally) sorted copy of this cell.
    ///
    /// For cells `c₁` and `c₂` with the same nodes,
    /// the new ordering satisfies
    /// ```text
    /// sorted(c₁) = sorted(c₂)
    /// ```
    fn sorted(&self) -> Self;
}

// todo: merge ordered and oriented cell maybe?

/// A [topological cell](Cell) with a global orientation (`+1` or `-1`).
pub trait OrientedCell<K: DimName>: Cell<K> {
    /// Returns the global orientation of this cell (`+1` or `-1`).
    fn orientation(&self) -> i8; 
    // todo: update return value with Enum
    //  - a global orientation is not needed. Replace with orientation(other: &Self),
    //      that just checks using orientation_eq
    
    /// Returns true, if the orientation of this and `other` are the same.
    fn orientation_eq(&self, other: &Self) -> bool;

    /// Returns a copy of this cell with reversed orientation.
    fn reversed(&self) -> Self;
}

/// A [topological cell](Cell) with a boundary.
pub trait CellBoundary<K: DimName + DimNameSub<U1>>: Cell<K> {
    /// Number of [`K`]`-1`-dimensional sub-cells in the boundary of this cell.
    const NUM_SUB_CELLS: usize;

    /// Cell topology of the individual sub-cells of the boundary chain.
    type SubCell: Cell<DimNameDiff<K, U1>>;

    /// Topology of the [`K`]`-1`-dimensional boundary of this cell.
    type Boundary: Chain<DimNameDiff<K, U1>, Self::SubCell>;

    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
}

/// Type of sub-cells the [`K`]`-1`-dimensional boundary of [`C`] is composed.
pub type SubCell<K, C> = <C as CellBoundary<K>>::SubCell;

/// Edge of a `2`-dimensional face element [`F`].
pub type Edge2<F> = SubCell<U2, F>;

/// Face of a `3`-dimensional cell element [`C`].
pub type Face3<C> = SubCell<U3, C>;

/// Edge of a `3`-dimensional cell element [`C`].
pub type Edge3<C> = SubCell<Face3<C>, C>;

impl <K: DimName> Cell<K> for () {
    fn nodes(&self) -> &[NodeIdx] {
        &[]
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        K: DimNameSub<M>
    {
        false
    }
}