//! Topology of an element-vertex mesh.

use std::hash::Hash;
use itertools::Itertools;
use nalgebra::{Const, DimNameSub, U1, U2};
use crate::cells::cell::{Cell, CellBoundary, Edge2, OrderedCell};
use crate::cells::chain::Chain;
use crate::cells::quad::QuadTopo;
use crate::cells::vertex::VertexTopo;

/// Topology of [`K`]-dimensional element-vertex mesh.
/// The elements are `K`-cells of type [`C`].
pub struct ElementVertex<const K: usize, C: Cell<Const<K>>> {
    /// Element connectivity vector.
    pub elems: Vec<C>,
    
    /// Total number of nodes.
    pub num_nodes: usize,
}

impl <const K: usize, C: Cell<Const<K>>> ElementVertex<K, C> {

    /// Constructs a new [`ElementVertex`] from the given `elems` topology vector 
    /// and the total number of nodes `num_nodes`.
    pub fn new(elems: Vec<C>, num_nodes: usize) -> Self {
        ElementVertex { elems, num_nodes }
    }
    
    /// Constructs a new [`ElementVertex`] from the given `elems` topology vector.
    /// The number of nodes is calculated as the maximal node index in `elems`.
    pub fn from_elems(elems: Vec<C>) -> Self {
        let num_nodes = elems.iter()
            .flat_map(C::nodes)
            .max()
            .expect("Elements vector `elems` must not be empty.")
            .0;
        
        ElementVertex::new(elems, num_nodes)
    }
    
    /// Returns and iterator over all nodes in increasing index order.
    pub fn nodes(&self) -> impl Iterator<Item = VertexTopo> {
        (0..self.num_nodes).map(VertexTopo)
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

impl <const K: usize, C: CellBoundary<Const<K>>> ElementVertex<K, C>
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

    /// Returns an iterator over all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = VertexTopo> + '_ {
        self.nodes().filter(|&n| self.is_boundary_node(n))
    }

    // todo: add info about regular/ irregular adjacency
}

/// A face-vertex mesh topology with `2`-dimensional faces [`F`].
type FaceVertex<F> = ElementVertex<2, F>;

impl <F: CellBoundary<U2>> FaceVertex<F>
    where Edge2<F>: OrderedCell<U1> + Clone + Eq + Hash
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
type QuadVertex = FaceVertex<QuadTopo>;

impl QuadVertex {
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
