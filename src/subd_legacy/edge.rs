use std::cmp::minmax;
use crate::subd_legacy::face::edges_of_face;
use crate::subd_legacy::mesh::{Edge, Face};

/// Returns a new edge with reverse orientation.
pub fn reverse_edge(edge: Edge) -> Edge {
    [edge[1], edge[0]]
}

/// Changes the orientation of `edge` to match the edges of `face`.
/// Assumes that `edge` is included in `face`.
pub fn sort_edge_by_face(edge: Edge, face: Face) -> Edge {
    let edges = edges_of_face(face);
    if edges.contains(&edge) { edge } else { reverse_edge(edge) }
}

/// Changes the orientation of `edge` such that `edge[0] < edge[1]`.
pub fn sort_edge(edge: Edge) -> Edge {
    minmax(edge[0], edge[1])
}

/// Returns the edge after `edge` of the given `face`.
/// Assumes that `edge` is included in correct orientation in `face`.
pub fn next_edge(edge: Edge, face: Face) -> Edge {
    let edges = edges_of_face(face);
    let idx = edges.iter().position(|e| *e == edge).unwrap();
    edges[(idx + 1) % 4]
}