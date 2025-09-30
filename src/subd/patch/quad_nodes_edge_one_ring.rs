use crate::cells::line_segment::DirectedEdge;
use crate::cells::node::NodeIdx;
use crate::mesh::face_vertex::QuadVertexMesh;
use itertools::Itertools;
use nalgebra::RealField;

/// Nodes of quadrilaterals in the one-ring neighborhood around a center edge.
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub enum QuadNodesEdgeOneRing {
    /// The regular interior case of valence `n=4`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   4 --- 5
    ///   |     |
    ///   2 --> 3
    ///   |     |
    ///   0 --- 1
    /// ```
    /// where the direction of the center edge is indicated by the arrow.
    Regular([NodeIdx; 6]),

    // todo: change this case to contain an entire quad, not just the edge
    /// The regular boundary case, which is equal to a single edge.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   0 --> 1
    /// ```
    /// where the arrow indicates the direction of the edge.
    Boundary([NodeIdx; 2])
}

impl QuadNodesEdgeOneRing {

    /// Finds all face nodes belonging to the one-ring around given `center` edge.
    pub fn find<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>, center: DirectedEdge) -> Self {
        // Find all faces with the `center` edge
        let faces = msh.faces_of_edge(center).collect_vec();
        match faces[..] {
            [] => {
                // todo: return Result instead of panic
                panic!("The given edge (is {center:?}) is not contained in the mesh.")
            }
            [q] => {
                QuadNodesEdgeOneRing::Boundary(center.0)
            }
            [mut q1, mut q2] => {
                // Get edge 2 -> 3
                let shared_edge = q1.shared_edge(*q2).unwrap();

                // If shared edge is 3 -> 2, change q1 and q2 to fix orientation
                if shared_edge == center {
                    std::mem::swap(&mut q1, &mut q2);
                }

                // Sort nodes
                let [n0, n1, n3, n2] = q1.sorted_by_node(center.end(), 2).nodes();
                let [_, _, n5, n4] = q2.sorted_by_origin(center.start()).nodes();
                QuadNodesEdgeOneRing::Regular([n0, n1, n2, n3, n4, n5])

            }
            _ => unreachable!("There can't be more than 2 faces in the edge one-ring")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::quad::QuadNodes;
    use nalgebra::SMatrix;

    /// Constructs the (irregular) quad mesh
    /// ```text
    ///   2 --- 3 --- 4
    ///   |     |     |
    ///   1 --- 0 --- 5
    ///  ╱    ╱ |     |
    /// 10   ╱  7 --- 6
    ///  ╲  ╱  ╱
    ///   9 - 8
    /// ```
    /// of valence `n=5` with all-zero control points.
    fn setup() -> QuadVertexMesh<f64, 2> {
        let faces = vec![
            QuadNodes::new(1, 0, 3, 2),
            QuadNodes::new(0, 5, 4, 3),
            QuadNodes::new(7, 6, 5, 0),
            QuadNodes::new(9, 8, 7, 0),
            QuadNodes::new(9, 0, 1, 10),
        ];

        QuadVertexMesh::from_coords_matrix(SMatrix::<f64, 11, 2>::zeros(), faces)
    }

    #[test]
    fn find() {
        let msh = setup();

        // Edge 0 -> 5
        let edge = DirectedEdge([NodeIdx(0), NodeIdx(5)]);
        let ring = QuadNodesEdgeOneRing::find(&msh, edge);
        let nodes_exp = [7, 6, 0, 5, 3, 4].map(NodeIdx);
        assert_eq!(ring, QuadNodesEdgeOneRing::Regular(nodes_exp));
        
        // Edge 9 -> 0
        let edge = DirectedEdge([NodeIdx(9), NodeIdx(0)]);
        let ring = QuadNodesEdgeOneRing::find(&msh, edge);
        let nodes_exp = [8, 7, 9, 0, 10, 1].map(NodeIdx);
        assert_eq!(ring, QuadNodesEdgeOneRing::Regular(nodes_exp));
        
        // Edge 3 -> 4
        let edge = DirectedEdge([NodeIdx(3), NodeIdx(4)]);
        let ring = QuadNodesEdgeOneRing::find(&msh, edge);
        let nodes_exp = [3, 4].map(NodeIdx);
        assert_eq!(ring, QuadNodesEdgeOneRing::Boundary(nodes_exp));
        
        // Edge 1 -> 10
        let edge = DirectedEdge([NodeIdx(1), NodeIdx(10)]);
        let ring = QuadNodesEdgeOneRing::find(&msh, edge);
        let nodes_exp = [1, 10].map(NodeIdx);
        assert_eq!(ring, QuadNodesEdgeOneRing::Boundary(nodes_exp));
    }
}