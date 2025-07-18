use itertools::Itertools;
use nalgebra::RealField;
use crate::cells::node::NodeIdx;
use crate::mesh::face_vertex::QuadVertexMesh;

/// Nodes of 2-by-2 quadrilaterals.
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
                    todo!("implement regular case")
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
            todo!("implement irregular case")
        }
    }
    
    fn traverse_faces_regular() {
        
    }
}