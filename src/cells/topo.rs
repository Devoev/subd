use crate::cells::chain::Chain;
use crate::cells::node::NodeIdx;
use crate::mesh::traits::VertexStorage;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, DimNameDiff, DimNameSub, Scalar, U1};
use crate::cells::geo;
use crate::cells::geo::CellAllocator;
// todo: refactor
//  - replace T and M with GATs
//   => T and M MUST be generics and not be moves to GATs, because the concrete GeoCell and Coords impl possibly require stricter variants
//  - should a topological cell really have knowledge about the geometry? Should this just be a
//    sub-trait of geo:Cell (i.e. CellInMesh)
//   => this can't work, because there is exactly one geometry description for multiple different connectivity descriptions
//  - the associated types don't quite make sense. Ideally this should be independent of the exact mesh used
//    (for example 2D vs 3D quad mesh should both work for quads, any mesh for (Un-)DirectedEdge).
//    This can maybe also be fixed by moving the associated types to generics of the method.
//    Also for all topologies defined by only nodes, the Coords parameter is sufficient.
//   => just using Coords likely is sufficient, because that is the only way vertex coordinate info is stored anyway.
//      but move it to a generic parameter of to_geo_cell

/// Topology of a cell inside a mesh.
///
/// The topology inside a mesh is uniquely defined by its corner node indices
/// which can be obtained by the [`Cell::nodes`] method.
pub trait Cell {
    /// Topological dimension of this cell.
    type Dim: DimName;

    /// Node representing the indices of vertices inside a mesh.
    type Node;

    /// Returns a slice of node indices corresponding to corner vertices of `self`.
    fn nodes(&self) -> &[Self::Node];
}

/// Conversion of a topological into a geometric cell.
///
/// Given a matching coordinate storage [`Cell::Coords`] of dimension `M`
/// the geometric representation of the cell can be constructed using [`Cell::to_geo_cell`].
/// This is useful for 'extracting' geometric information
/// about a part of a computational domain from the mesh (see [`Cell::GeoCell`]).
pub trait ToGeoCell<T: Scalar, M: DimName>: Cell
    where DefaultAllocator: Allocator<M> + CellAllocator<T, Self::GeoCell>
{
    /// The geometric cell associated with this topology.
    type GeoCell: geo::Cell<T>;

    /// Coordinates storage of the associated mesh.
    type Coords: VertexStorage<T, GeoDim = M, NodeIdx = Self::Node>;

    /// Constructs the geometric cell associated with `self` from the given vertex `coords`.
    fn to_geo_cell(&self, coords: &Self::Coords) -> Self::GeoCell;
}

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
    fn contains_node(&self, node: NodeIdx) -> bool {
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
    type SubCell: Cell<Dim = DimNameDiff<Self::Dim, U1>>;

    /// Topology of the [`K`]`-1`-dimensional boundary of this cell.
    type Boundary: Chain<Self::SubCell>;

    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
}

/// Type of sub-cells the [`K`]`-1`-dimensional boundary of [`C`] is composed.
pub type SubCell<C> = <C as CellBoundary>::SubCell;