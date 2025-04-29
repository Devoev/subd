//! Data structures for a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use std::cmp::minmax;
use nalgebra::{Point2, RealField};

pub type NodeIdx = usize;

/// Topology of an edge, defined by the start and end nodes.
#[derive(Debug, Copy, Clone)]
pub struct EdgeTopo([NodeIdx; 2]);

impl EdgeTopo {

    /// Returns the start node of this edge.
    pub fn start(&self) -> NodeIdx {
        self.0[0]
    }

    /// Returns the end node of this edge.
    pub fn end(&self) -> NodeIdx {
        self.0[1]
    }

    /// Returns the boundary of this edge, i.e. the starting and end nodes.
    pub fn boundary(&self) -> [NodeIdx; 2] {
        self.0
    }

    /// Returns a sorted copy of this edge such that `self.start() < self.end()`.
    pub fn sorted(&self) -> EdgeTopo {
        EdgeTopo(minmax(self.start(), self.end()))
    }
    
    /// Changes the orientation of this edge by calling [`EdgeTopo::sorted`] on self.
    pub fn sort(&mut self) {
        *self = self.sorted();
    }
    
    /// Returns a copy of this edge with reversed orientation.
    pub fn reversed(&self) -> EdgeTopo {
        EdgeTopo([self.end(), self.start()])
    }
    
    /// Reversed the orientation of this edge by calling [`EdgeTopo::reversed`].
    pub fn reverse(&mut self) {
        *self = self.reversed();
    }
}

/// Topology of a 2D quadrilateral face, defined by its four corner vertices.
pub struct QuadTopo2d([NodeIdx; 4]);

impl QuadTopo2d {

    // todo: update doc and change to connected edges, i.e. a chain topology
    /// Returns the boundary of this quadrilateral face, i.e. all 4 edges.
    pub fn boundary(&self) -> [EdgeTopo; 4] {
        let [a, b, c, d] = self.0;
        [EdgeTopo([a, b]), EdgeTopo([b, c]), EdgeTopo([c, d]), EdgeTopo([d, a])]
    }
}

/// Topology of a 2D quadrilateral face-vertex mesh.
pub struct QuadMeshTopo2d {
    /// Face connectivity vector.
    pub faces: Vec<QuadTopo2d>
}

/// A 2D quadrilateral face-vertex mesh with geometric data of the coordinates of each vertex.
pub struct QuadMesh2d<T: RealField> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the faces.
    pub topology: QuadMeshTopo2d,
}