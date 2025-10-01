use crate::cells::chain::Chain;
use crate::element::traits::{ElemAllocator, Element};
use crate::mesh::traits::VertexStorage;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, DimNameDiff, DimNameSub, Scalar, U1};

/// Topology of a cell inside a mesh.
///
/// The topology inside a mesh is uniquely defined by its corner node indices
/// which can be obtained by the [`Cell::nodes`] method.
pub trait Cell {
    /// Topological dimension of this cell.
    type Dim: DimName;

    /// Node representing the indices of vertices inside a mesh.
    type Node: Copy;

    /// Returns a slice of node indices corresponding to corner vertices of `self`.
    fn nodes(&self) -> &[Self::Node];
}

/// Conversion of a topological cell into a geometric element.
///
/// Given a matching coordinate storage [`Cell::Coords`] of dimension `M`
/// the geometric representation of the cell can be constructed using [`Cell::to_geo_cell`].
/// This is useful for 'extracting' geometric information
/// about a part of a computational domain from the mesh (see [`Cell::GeoCell`]).
pub trait ToElement<T: Scalar, M: DimName>: Cell
    where DefaultAllocator: Allocator<M> + ElemAllocator<T, Self::Elem>
{
    /// The geometric element associated with this topology.
    type Elem: Element<T>;

    /// Coordinates storage of the associated mesh.
    type Coords: VertexStorage<T, GeoDim = M, NodeIdx = Self::Node>;

    /// Constructs the geometric element associated with `self` from the given vertex `coords`.
    fn to_element(&self, coords: &Self::Coords) -> Self::Elem;
}

/// The geometrical element associated with the topological cell `C`.
pub type ElemOfCell<T, C, M> = <C as ToElement<T, M>>::Elem;

/// A [topological cell](Cell) with connectivity and neighboring relations.
pub trait CellConnectivity: Cell {
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
    where Self::Dim: DimNameSub<M>;

    /// Returns `true` if the cell contains the given `node`.
    fn contains_node(&self, node: Self::Node) -> bool {
        self.nodes().contains(&node)
    }
    
    /// Returns `true` if `self` and `other` are topologically equal.
    /// This is the same as testing `self.is_connected(other, K::name())`.
    fn topo_eq(&self, other: &Self) -> bool
        where Self::Dim: DimNameSub<Self::Dim>
    {
        self.is_connected(other, Self::Dim::name())
    }
}

/// A [topological cell](CellConnectivity) with an ordering of its nodes.
pub trait OrderedCell: Cell {
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

/// A [topological cell](CellConnectivity) with a global orientation (`+1` or `-1`).
pub trait OrientedCell: Cell {
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

/// A [topological cell](CellConnectivity) with a boundary.
pub trait CellBoundary: Cell
where Self::Dim: DimNameSub<U1> {
    /// Number of [`K`]`-1`-dimensional sub-cells in the boundary of this cell.
    const NUM_SUB_CELLS: usize;

    /// Cell topology of the individual sub-cells of the boundary chain.
    type SubCell: Cell<Dim = DimNameDiff<Self::Dim, U1>, Node = Self::Node>;

    /// Topology of the [`K`]`-1`-dimensional boundary of this cell.
    type Boundary: Chain<Self::SubCell>;

    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
}

/// Type of sub-cells the [`K`]`-1`-dimensional boundary of [`C`] is composed.
pub type SubCell<C> = <C as CellBoundary>::SubCell;