use std::iter::zip;
use nalgebra::{Point2, RealField};
use crate::subd::mesh::{Edge, Face, Node};

/// Returns whether the two faces are adjacent, i.e. share an edge.
pub fn is_adjacent(a: &Face, b: &Face) -> bool {
    a.iter().filter(|n| b.contains(n)).count() >= 2
}

/// Returns the edges of the given face in the following order
/// ```
///     + -- 2 -- +
///     |         |
/// v   3         1
/// ^   |         |
/// |   + -- 0 -- +
/// +---> u
/// ```
pub fn edges_of_face(face: Face) -> [Edge; 4] {
    let [a, b, c, d] = face;
    [[a, b], [b, c], [c, d], [d, a]]
}

/// Returns a sorted version of the given `face`. The face is sorted in positive orientation as
/// ```text
/// y   3 --- 2
/// ^   |  c  |
/// |   0 --- 1
/// +---> x
/// ```
/// where `0..3` are the coordinates of `nodes` and `c` is the `centroid` of the face.
pub fn sort_face<T: RealField + Copy>(face: Face, nodes: [Point2<T>; 4], centroid: Point2<T>) -> Face {
    let mut sorted: Face = [0; 4];
    for (i, node) in zip(face, nodes) {
        let idx = match (node.x, node.y) {
            (x, y) if x < centroid.x && y < centroid.y => 0,
            (x, y) if x > centroid.x && y < centroid.y => 1,
            (x, y) if x > centroid.x && y > centroid.y => 2,
            _ => 3,
        };
        sorted[idx] = i;
    }

    sorted
}

/// Returns a sorted face, such that the given `node` is at the face position `idx`.
///
/// For `idx=3` the faces nodes get sorted as
/// ```text
/// v   3 --- 2         n --- 0
/// ^   |     |   ==>   |     |
/// |   0 --- n         2 --- 3
/// +---> u
/// ```
/// i.e. the given node `n` moves from the original position `1` to position `idx=3`.
pub fn sort_by_node(face: Face, node: Node, idx: usize) -> Face {
    let original_idx = face.iter().position(|&n| n == node).unwrap();
    let mut sorted = face;
    if idx > original_idx {
        sorted.rotate_right(idx - original_idx);
    } else {
        sorted.rotate_left(original_idx - idx);
    }
    sorted
}

/// Returns a sorted face, such that the node `uv_origin` is the first node.
/// The sorting is
/// ```text
/// v   2 --- 1         3 --- 2
/// ^   |     |   ==>   |     |
/// |   3 --- 0         0 --- 1
/// +---> u
/// ```
/// where `0` is the `uv_origin`.
/// Assumes the face is initially sorted in positive orientation.
pub fn sort_by_origin(face: Face, uv_origin: Node) -> Face {
    // let idx = face.iter().position(|&n| n == uv_origin).unwrap();
    // let mut sorted = face;
    // sorted.rotate_left(idx);
    // sorted
    sort_by_node(face, uv_origin, 0)
}