//! Data structures for a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::mesh::cell::{CellBoundaryTopo, CellTopo, Edge2, OrderedCellTopo};
use crate::mesh::chain::ChainTopo;
use crate::mesh::line_segment::LineSegment;
use crate::mesh::quad::{Quad, QuadTopo};
use crate::mesh::vertex::VertexTopo;
use crate::subd::patch::Patch;
use itertools::Itertools;
use nalgebra::{Const, DimNameSub, Point2, RealField, U1, U2};
use std::hash::Hash;

/// Topology of [`K`]-dimensional element-vertex mesh.
/// The elements are `K`-cells of type [`C`].
pub struct ElementVertexTopo<const K: usize, C: CellTopo<Const<K>>> {
    /// Element connectivity vector.
    pub elems: Vec<C>
}

impl <const K: usize, C: CellTopo<Const<K>>> ElementVertexTopo<K, C> {
    
    /// Constructs a new [`ElementVertexTopo`] from the given `elems` topology vector.
    pub fn new(elems: Vec<C>) -> Self {
        ElementVertexTopo { elems }
    }

    /// Finds all elements which contain the given `node` and returns them as an iterator.
    pub fn elems_of_node(&self, node: VertexTopo) -> impl Iterator<Item = &C> {
        self.elems
            .iter()
            .filter(move |elem| elem.contains_node(node))
    }

    /// Finds all elements adjacent to the given `elem` and returns them as an iterator.
    pub fn adjacent_elems<'a>(&'a self, elem: &'a C) -> impl Iterator<Item = &'a C> + 'a
        where Const<K>: DimNameSub<Const<K>>
    {
        self.elems
            .iter()
            .filter(move |e| e.is_connected::<K>(elem))
    }
    
}

impl <const K: usize, C: CellBoundaryTopo<Const<K>>> ElementVertexTopo<K, C>
    where Const<K>: DimNameSub<U1> + DimNameSub<Const<K>>
{
    /// Returns `true` if given `elem` is at the boundary of the mesh, 
    /// i.e. it has less than [`C::NUM_SUB_CELLS`] adjacent elements.
    pub fn is_boundary_elem(&self, elem: &C) -> bool {
        self.adjacent_elems(elem).count() < C::NUM_SUB_CELLS
    }
    
    // todo: is_boundary_node is inefficient. Update this by not calling is_boundary_elem ?
    /// Returns `true` if the given `node` is a boundary node,
    /// i.e. all elements containing the node are boundary elements.
    pub fn is_boundary_node(&self, node: VertexTopo) -> bool {
        self.elems_of_node(node).all(|elem| self.is_boundary_elem(elem))
    }

    // todo: move other methods for boundary computation here. Add info about regular/ irregular adjacency
}

/// A face-vertex mesh topology with `2`-dimensional faces [`F`].
type FaceVertexTopo<F> = ElementVertexTopo<2, F>;

impl <F: CellBoundaryTopo<U2>> FaceVertexTopo<F>
    where Edge2<F>: OrderedCellTopo<U1> + Clone + Eq + Hash
{
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.elems.iter()
            .flat_map(|face| face.boundary().cells().to_owned())
            .map(|edge: Edge2<F> | edge.sorted())
            .unique()
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: VertexTopo) -> impl Iterator<Item = Edge2<F>> + '_ {
        self.edges().filter(move |edge| edge.contains_node(node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: VertexTopo) -> usize {
        self.edges_of_node(node).count()
    }
}

/// A face-vertex mesh topology of quadrilateral faces.
type QuadVertexTopo = FaceVertexTopo<QuadTopo>;

impl QuadVertexTopo {
    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: QuadTopo) -> bool {
        face.nodes().iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: QuadTopo) -> Option<VertexTopo> {
        face.nodes()
            .into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: QuadTopo) -> Vec<VertexTopo> {
        face.nodes().into_iter().filter(|&v| self.valence(v) != 4).collect()
    }
}

/// Element-vertex mesh of topology [`ElementVertexTopo`]
/// with geometric data of the coordinates of each vertex.
pub struct ElementVertexMesh<const K: usize, T: RealField, C: CellTopo<Const<K>>> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the elements.
    pub topology: ElementVertexTopo<K, C>,
}

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C> = ElementVertexMesh<2, T, C>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T> = FaceVertexMesh<T, QuadTopo>;

impl<T: RealField> QuadVertexMesh<T> {
    /// Returns the [`Point2`] of the given `node` index.
    pub fn coords(&self, node: VertexTopo) -> &Point2<T> {
        &self.coords[node.0]
    }

    // todo: possibly move this method to topology
    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item=LineSegment<T>> + '_ {
        self.topology.edges().map(|edge_top| LineSegment::from_msh(edge_top, self))
    }

    /// Returns an iterator over all faces in this mesh.
    pub fn faces(&self) -> impl Iterator<Item=Quad<T>> + '_ {
        self.topology.elems.iter().map(|&face| Quad::from_msh(face, self))
    }
    
    // todo: possibly move this method to topology
    /// Returns an iterator over the indices of all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = VertexTopo> + '_ {
        (0..self.num_nodes())
            .map(VertexTopo)
            .filter(|&n| self.topology.is_boundary_node(n))
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: QuadTopo) -> Patch<T> {
        todo!()
    }

    /// Returns an iterator over all patches in this mesh.
    pub fn patches(&self) -> impl Iterator<Item =Patch<T>> {
        self.topology.elems.iter().map(|&face| self.find_patch(face))
    }
}