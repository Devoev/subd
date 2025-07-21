use crate::cells::node::NodeIdx;
use crate::mesh::face_vertex::QuadVertexMesh;
use itertools::Itertools;
use nalgebra::RealField;
use crate::cells::line_segment::DirectedEdge;
use crate::cells::quad::QuadNodes;

/// Nodes of 2-by-2 quadrilaterals.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum QuadNodes2x2 {
    /// The regular interior case of valence `n=4`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   6 --- 7 --- 8
    ///   |     |     |
    ///   3 --- 4 --- 5
    ///   |     |     |
    ///   0 --- 1 --- 2
    /// ```
    /// where node `4` is the center node.
    Regular([NodeIdx; 9]),

    /// The regular boundary case of valence `n=3`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   |     |     |
    ///   3 --- 4 --- 5
    ///   |     |     |
    ///   0 --- 1 --- 2
    /// ```
    /// where node `1` is the center node.
    Boundary([NodeIdx; 6]),

    /// The regular corner case of valence `n=2` (equivalent to a single quad).
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   |     |
    ///   2 --- 3 ---
    ///   |     |
    ///   0 --- 1 ---
    /// ```
    /// where node `0` is the center node.
    Corner([NodeIdx; 4]),

    /// The irregular interior case of valence `n≠4`.
    /// The nodes are ordered in the following order
    /// ```text
    ///   2 --- 3 --- 4
    ///   |     |     |
    ///   1 --- 0 --- 5
    ///  ╱    ╱ |     |
    /// 2N   ╱  7 --- 6
    ///  ╲  ╱  ╱
    ///   ○ - 8
    /// ```
    /// where node `0` is the irregular center node.
    Irregular(Vec<NodeIdx>, usize)
}

impl QuadNodes2x2 {
    /// Finds all face nodes belonging to the 2-by-2 quad patch of the given `center` node.
    pub fn find<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>, center: NodeIdx) -> Self {
        // Find all faces with `center` as a node
        let faces = msh.elems_of_node(center).collect_vec();

        if msh.is_boundary_node(center) || msh.is_regular_node(center) {
            match faces[..] {
                [q1, q2, q3, q4] => {
                    // In this case, there is no preferred orientation (the patch is rotationally symmetric)
                    // Hence randomly choose q1 as the lower left face (and sort it)
                    let q1 = q1.sorted_by_node(center, 2);
                    let mut nodes = [q1.nodes()[0]; 9];

                    // Find right, top and diagonal faces and update nodes
                    for &other in [q2, q3, q4] {
                        match q1.shared_edge(other) {
                            None => { // no shared edge = diagonal face
                                let q_diag = other.sorted_by_origin(center);
                                [nodes[4], nodes[5], nodes[8], nodes[7]] = q_diag.nodes();
                            }
                            Some(edge) if edge.end() == center => { // edge 1 -> 4 == right face
                                let q_right = other.sorted_by_node(center, 3);
                                [nodes[1], nodes[2], nodes[5], nodes[4]] = q_right.nodes();
                            },
                            Some(edge) => { // edge 4 -> 3 == top face
                                let q_top = other.sorted_by_node(center, 1);
                                [nodes[3], nodes[4], nodes[7], nodes[6]] = q_top.nodes();
                            }
                        }
                    }

                    QuadNodes2x2::Regular(nodes)
                }
                [mut q1, mut q2] => {
                    // Get edge 0 -> 4
                    let shared_edge = q1.shared_edge(*q2).unwrap();

                    // If edge is 4 -> 0, change q1 and q2 to fix orientation
                    if shared_edge.end() == center {
                        std::mem::swap(&mut q1, &mut q2);
                    }

                    // Sort nodes
                    let [n0, n1, n4, n3] = q1.sorted_by_node(center, 1).nodes();
                    let [_, n2, n5, _] = q2.sorted_by_origin(center).nodes();
                    QuadNodes2x2::Boundary([n0, n1, n2, n3, n4, n5])
                }
                [q] => {
                    let [a, b, c, d] = q.sorted_by_origin(center).nodes();
                    QuadNodes2x2::Corner([a, b, d, c])
                }
                _ => todo!("This case can only occur if there is an irregular node directly at the boundary. \
                    Should probably be implemented sometime.")
            }
        } else {
            // As in the regular case, there is preferred orientation, so just choose the first face
            let n = msh.valence(center);
            let q1 = faces[0].sorted_by_origin(center);
            let mut nodes_regular = [q1.nodes()[2]; 8];

            // First find the two regular faces adjacent to q1
            let mut faces_irregular = Vec::with_capacity(n - 3);

            for &other in faces.into_iter().skip(1) {
                match q1.shared_edge(other) {
                    None => { // no shared edge = irregular face
                        let q_irregular = other.sorted_by_origin(center);
                        faces_irregular.push(q_irregular);
                    }
                    Some(edge) if edge.end() == center => { // edge 3 -> 0 == left face
                        let q_left = other.sorted_by_node(center, 1);
                        [nodes_regular[1], nodes_regular[0], nodes_regular[3], nodes_regular[2]] = q_left.nodes();
                    },
                    Some(_) => { // edge 0 -> 5 == bottom face
                        let q_bottom = other.sorted_by_node(center, 3);
                        [nodes_regular[7], nodes_regular[6], nodes_regular[5], nodes_regular[0]] = q_bottom.nodes();
                    }
                }
            }

            let mut nodes_irregular = Vec::with_capacity(2*n - 6);
            let mut next_edge = DirectedEdge([nodes_regular[7], center]);
            while !faces_irregular.is_empty() {
                let face_idx = faces_irregular.iter()
                    .position(|face| face.edges().contains(&next_edge))
                    .expect("No face contains given edge. Check faces argument.");

                let next_face = faces_irregular.swap_remove(face_idx).sorted_by_origin(center);
                nodes_irregular.push(next_face.nodes()[3]);
                nodes_irregular.push(next_face.nodes()[2]);
                next_edge = next_face.edges()[0].reversed();
            }

            let mut nodes = nodes_regular.to_vec();
            nodes.extend_from_slice(&nodes_irregular[1..]);
            QuadNodes2x2::Irregular(nodes, n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cells::quad::QuadNodes;
    use nalgebra::SMatrix;

    /// Constructs the `3✕3` regular quad mesh
    /// ```text
    ///   6 --- 7 --- 8
    ///   |  1  |  2  |
    ///   3 --- 4 --- 5
    ///   |  3  |  0  |
    ///   0 --- 1 --- 2
    /// ```
    /// with all-zero control points.
    fn setup_regular() -> QuadVertexMesh<f64, 2> {
        let faces = vec![
            QuadNodes::from_indices(2, 5, 4, 1),
            QuadNodes::from_indices(6, 3, 4, 7),
            QuadNodes::from_indices(0, 1, 4, 3),
            QuadNodes::from_indices(7, 4, 5, 8),
        ];

        QuadVertexMesh::from_matrix(SMatrix::<f64, 9, 2>::zeros(), faces)
    }

    /// Constructs the irregular quad mesh
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
    fn setup_irregular() -> QuadVertexMesh<f64, 2> {
        let faces = vec![
            QuadNodes::from_indices(1, 0, 3, 2),
            QuadNodes::from_indices(0, 5, 4, 3),
            QuadNodes::from_indices(7, 6, 5, 0),
            QuadNodes::from_indices(9, 8, 7, 0),
            QuadNodes::from_indices(9, 0, 1, 10),
        ];

        QuadVertexMesh::from_matrix(SMatrix::<f64, 11, 2>::zeros(), faces)
    }

    #[test]
    fn find_regular() {
        let msh = setup_regular();

        // Regular case (test against 4 alignments, because this case is rotationally symmetric)
        let patch = QuadNodes2x2::find(&msh, NodeIdx(4));
        let nodes_exp_bottom_align = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(NodeIdx);
        let nodes_exp_right_align = [2, 5, 8, 1, 4, 7, 0, 3, 6].map(NodeIdx);
        let nodes_exp_top_align = [8, 7, 6, 5, 4, 3, 2, 1, 0].map(NodeIdx);
        let nodes_exp_left_align = [6, 3, 0, 7, 4, 1, 8, 5, 2].map(NodeIdx);
        assert!(
            patch == QuadNodes2x2::Regular(nodes_exp_bottom_align)
            || patch == QuadNodes2x2::Regular(nodes_exp_right_align)
            || patch == QuadNodes2x2::Regular(nodes_exp_top_align)
            || patch == QuadNodes2x2::Regular(nodes_exp_left_align)
        );

        // Boundary case
        let patch = QuadNodes2x2::find(&msh, NodeIdx(1));
        let nodes_exp_bottom_bnd = [0, 1, 2, 3, 4, 5].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Boundary(nodes_exp_bottom_bnd));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(3));
        let nodes_exp_left_bnd = [6, 3, 0, 7, 4, 1].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Boundary(nodes_exp_left_bnd));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(5));
        let nodes_exp_right_bnd = [2, 5, 8, 1, 4, 7].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Boundary(nodes_exp_right_bnd));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(7));
        let nodes_exp_top_bnd = [8, 7, 6, 5, 4, 3].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Boundary(nodes_exp_top_bnd));

        // Corner case
        let patch = QuadNodes2x2::find(&msh, NodeIdx(0));
        let nodes_exp_bottom_left_corner = [0, 1, 3, 4].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Corner(nodes_exp_bottom_left_corner));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(2));
        let nodes_exp_bottom_right_corner = [2, 5, 1, 4].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Corner(nodes_exp_bottom_right_corner));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(6));
        let nodes_exp_top_left_corner = [6, 3, 7, 4].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Corner(nodes_exp_top_left_corner));

        let patch = QuadNodes2x2::find(&msh, NodeIdx(8));
        let nodes_exp_top_right_corner = [8, 7, 5, 4].map(NodeIdx);
        assert_eq!(patch, QuadNodes2x2::Corner(nodes_exp_top_right_corner));
    }

    #[test]
    pub fn find_irregular() {
        let msh = setup_irregular();

        // Irregular case (test against 5 alignments, because of rotational symmetry)
        let patch = QuadNodes2x2::find(&msh, NodeIdx(0));
        let nodes_exp_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().map(NodeIdx).collect();
        let nodes_exp_2 = [0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8].into_iter().map(NodeIdx).collect();
        assert!(
            patch == QuadNodes2x2::Irregular(nodes_exp_1, 5)
            || patch == QuadNodes2x2::Irregular(nodes_exp_2, 5)
        );
        // todo: implement the other 3 alignments as well
    }
}