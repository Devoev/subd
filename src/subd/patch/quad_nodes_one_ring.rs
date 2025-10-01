use crate::cells::node::Node;
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use itertools::Itertools;
use nalgebra::RealField;

/// Nodes of quadrilaterals in the one-ring neighborhood around a center node.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum QuadNodesOneRing {
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
    Regular([Node; 9]),

    /// The regular boundary case of valence `n=3`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   |     |     |
    ///   3 --- 4 --- 5
    ///   |     |     |
    ///   0 --- 1 --- 2
    /// ```
    /// where node `1` is the center node.
    Boundary([Node; 6]),

    /// The regular corner case of valence `n=2` (equivalent to a single quad).
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///   |     |
    ///   2 --- 3 ---
    ///   |     |
    ///   0 --- 1 ---
    /// ```
    /// where node `0` is the center node.
    Corner([Node; 4]),

    /// The irregular interior case of valence `n≠4`.
    /// The nodes are ordered as
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
    Irregular(Vec<Node>, usize)
}

impl QuadNodesOneRing {
    /// Finds all face nodes belonging to the one-ring around given `center` node.
    pub fn find<T: RealField, const M: usize>(msh: &QuadVertexMesh<T, M>, center: Node) -> Self {
        // Find all faces with `center` as a node
        let faces = msh.elems_of_node(center).collect_vec();

        if msh.is_boundary_node(center) || msh.is_regular_node(center) {
            match faces[..] {
                [q1, q2, q3, q4] => {
                    Self::traverse_faces_regular(center, q1, q2, q3, q4)
                }
                [q1, q2] => {
                    Self::traverse_faces_boundary(center, *q1, *q2)
                }
                [q] => {
                    Self::traverse_faces_corner(center, q)
                }
                _ => todo!("This case can only occur if there is an irregular node directly at the boundary. \
                    Should probably be implemented sometime.")
            }
        } else {
            Self::traverse_faces_irregular(center, msh.valence(center), faces)
        }
    }

    /// Traverses the given faces `q1`, `q2`, `q3` and `q4` of a **regular** patch around the `center` node.
    ///
    /// The traversal order is described in [`QuadNodesOneRing::Regular`].
    fn traverse_faces_regular(center: Node, q1: &QuadNodes, q2: &QuadNodes, q3: &QuadNodes, q4: &QuadNodes) -> QuadNodesOneRing {
        // In the interior case, there is no preferred orientation (the patch is rotationally symmetric)
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
                Some(_) => { // edge 4 -> 3 == top face
                    let q_top = other.sorted_by_node(center, 1);
                    [nodes[3], nodes[4], nodes[7], nodes[6]] = q_top.nodes();
                }
            }
        }

        QuadNodesOneRing::Regular(nodes)
    }

    /// Traverses the given faces `q1`, and `q2` of a **boundary** patch around the `center` node.
    ///
    /// The traversal order is described in [`QuadNodesOneRing::Boundary`].
    fn traverse_faces_boundary(center: Node, mut q1: QuadNodes, mut q2: QuadNodes) -> QuadNodesOneRing {
        // Get edge 0 -> 4
        let shared_edge = q1.shared_edge(q2).unwrap();

        // If edge is 4 -> 0, change q1 and q2 to fix orientation
        if shared_edge.end() == center {
            std::mem::swap(&mut q1, &mut q2);
        }

        // Sort nodes
        let [n0, n1, n4, n3] = q1.sorted_by_node(center, 1).nodes();
        let [_, n2, n5, _] = q2.sorted_by_origin(center).nodes();
        QuadNodesOneRing::Boundary([n0, n1, n2, n3, n4, n5])
    }

    /// Traverses the given face `q1` of a **corner** patch around the `center` node.
    /// In this case, essentially no traversal but only sorting of the nodes is required.
    ///
    /// The traversal order is described in [`QuadNodesOneRing::Corner`].
    fn traverse_faces_corner(center: Node, q1: &QuadNodes) -> QuadNodesOneRing {
        let [a, b, c, d] = q1.sorted_by_origin(center).nodes();
        QuadNodesOneRing::Corner([a, b, d, c])
    }

    /// Traverses the given `faces` of an **irregular** patch of valence `n` around the `center` node.
    ///
    /// The traversal order is described in [`QuadNodesOneRing::Irregular`].
    fn traverse_faces_irregular(center: Node, n: usize, mut faces: Vec<&QuadNodes>) -> Self {
        // Initialize nodes vector
        let mut nodes = Vec::with_capacity(2*n);
        nodes.push(center);

        // Traverse 1-ring neighborhood
        // As there is no preferred orientation, just start with the first face (index = 0)
        let mut next_face_idx = 0;
        loop {
            // Remove the face from the vector and append coordinates
            let next_face = faces.swap_remove(next_face_idx).sorted_by_origin(center);
            nodes.push(next_face.nodes()[3]);
            nodes.push(next_face.nodes()[2]);

            // Traversal is finished if there are no faces left
            if faces.is_empty() { break }

            // Otherwise, find the next face
            let next_edge = next_face.edges()[0].reversed();
            next_face_idx = faces.iter()
                .position(|face| face.edges().contains(&next_edge))
                .expect("No face contains given edge. Check faces argument.");
        }

        QuadNodesOneRing::Irregular(nodes, n)
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
            QuadNodes::new(2, 5, 4, 1),
            QuadNodes::new(6, 3, 4, 7),
            QuadNodes::new(0, 1, 4, 3),
            QuadNodes::new(7, 4, 5, 8),
        ];

        QuadVertexMesh::from_coords_matrix(SMatrix::<f64, 9, 2>::zeros(), faces)
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
            QuadNodes::new(1, 0, 3, 2),
            QuadNodes::new(0, 5, 4, 3),
            QuadNodes::new(7, 6, 5, 0),
            QuadNodes::new(9, 8, 7, 0),
            QuadNodes::new(9, 0, 1, 10),
        ];

        QuadVertexMesh::from_coords_matrix(SMatrix::<f64, 11, 2>::zeros(), faces)
    }

    #[test]
    fn find_regular() {
        let msh = setup_regular();

        // Regular case (test against 4 alignments, because this case is rotationally symmetric)
        let patch = QuadNodesOneRing::find(&msh, 4);
        let nodes_exp_bottom_align = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let nodes_exp_right_align = [2, 5, 8, 1, 4, 7, 0, 3, 6];
        let nodes_exp_top_align = [8, 7, 6, 5, 4, 3, 2, 1, 0];
        let nodes_exp_left_align = [6, 3, 0, 7, 4, 1, 8, 5, 2];
        assert!(
            patch == QuadNodesOneRing::Regular(nodes_exp_bottom_align)
            || patch == QuadNodesOneRing::Regular(nodes_exp_right_align)
            || patch == QuadNodesOneRing::Regular(nodes_exp_top_align)
            || patch == QuadNodesOneRing::Regular(nodes_exp_left_align)
        );

        // Boundary case
        let patch = QuadNodesOneRing::find(&msh, 1);
        let nodes_exp_bottom_bnd = [0, 1, 2, 3, 4, 5];
        assert_eq!(patch, QuadNodesOneRing::Boundary(nodes_exp_bottom_bnd));

        let patch = QuadNodesOneRing::find(&msh, 3);
        let nodes_exp_left_bnd = [6, 3, 0, 7, 4, 1];
        assert_eq!(patch, QuadNodesOneRing::Boundary(nodes_exp_left_bnd));

        let patch = QuadNodesOneRing::find(&msh, 5);
        let nodes_exp_right_bnd = [2, 5, 8, 1, 4, 7];
        assert_eq!(patch, QuadNodesOneRing::Boundary(nodes_exp_right_bnd));

        let patch = QuadNodesOneRing::find(&msh, 7);
        let nodes_exp_top_bnd = [8, 7, 6, 5, 4, 3];
        assert_eq!(patch, QuadNodesOneRing::Boundary(nodes_exp_top_bnd));

        // Corner case
        let patch = QuadNodesOneRing::find(&msh, 0);
        let nodes_exp_bottom_left_corner = [0, 1, 3, 4];
        assert_eq!(patch, QuadNodesOneRing::Corner(nodes_exp_bottom_left_corner));

        let patch = QuadNodesOneRing::find(&msh, 2);
        let nodes_exp_bottom_right_corner = [2, 5, 1, 4];
        assert_eq!(patch, QuadNodesOneRing::Corner(nodes_exp_bottom_right_corner));

        let patch = QuadNodesOneRing::find(&msh, 6);
        let nodes_exp_top_left_corner = [6, 3, 7, 4];
        assert_eq!(patch, QuadNodesOneRing::Corner(nodes_exp_top_left_corner));

        let patch = QuadNodesOneRing::find(&msh, 8);
        let nodes_exp_top_right_corner = [8, 7, 5, 4];
        assert_eq!(patch, QuadNodesOneRing::Corner(nodes_exp_top_right_corner));
    }

    #[test]
    pub fn find_irregular() {
        let msh = setup_irregular();

        // Irregular case (test against 5 alignments, because of rotational symmetry)
        let patch = QuadNodesOneRing::find(&msh, 0);
        let nodes_exp_1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let nodes_exp_2 = vec![0, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8];
        assert!(
            patch == QuadNodesOneRing::Irregular(nodes_exp_1, 5)
            || patch == QuadNodesOneRing::Irregular(nodes_exp_2, 5)
        );
        // todo: implement the other 3 alignments as well
    }
}