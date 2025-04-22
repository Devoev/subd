use crate::subd::edge::{next_edge, reverse_edge};
use crate::subd::face::{are_touching, edges_of_face, sort_by_node, sort_by_origin};
use crate::subd::mesh::{Face, Node, QuadMesh};
use itertools::{izip, Itertools};
use nalgebra::{Dyn, OMatrix, RealField, U2};
use std::collections::HashSet;
use std::iter::once;

/// Connectivity of the ordered nodes of a [`Patch`].
#[derive(Debug, Clone)]
pub enum NodeConnectivity {
    /// The regular interior case of valence `n=4`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  12 -- 13 -- 14 -- 15
    ///   |     |     |     |
    ///   8 --- 9 -- 10 -- 11
    ///   |     |  p  |     |
    ///   4 --- 5 --- 6 --- 7
    ///   |     |     |     |
    ///   0 --- 1 --- 2 --- 3
    /// ```
    /// where `p` is the center face of the patch.
    Regular([Node; 16]),

    /// The regular boundary case of valence `n=3`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  |     |     |     |
    ///  8 --- 9 -- 10 -- 11
    ///  |     |     |     |
    ///  4 --- 5 --- 6 --- 7
    ///  |     |  p  |     |
    ///  0 --- 1 --- 2 --- 3
    /// ```
    /// where `p` is the center face of the patch.
    Boundary([Node; 12]),

    /// The regular corner case of valence `n=2`.
    /// The nodes are ordered in lexicographical order
    /// ```text
    ///  |     |     |
    ///  6 --- 7 --- 8 ---
    ///  |     |     |
    ///  3 --- 4 --- 5 ---
    ///  |  p  |     |
    ///  0 --- 1 --- 2 ---
    /// ```
    /// where `p` is the center face of the patch.
    Corner([Node; 9]),

    /// The irregular interior case of valence `n≠4`.
    /// The nodes are ordered in the following order
    /// ```text
    /// 2N+7--2N+6--2N+5--2N+1
    ///   |     |     |     |
    ///   2 --- 3 --- 4 --2N+2
    ///   |     |  p  |     |
    ///   1 --- 0 --- 5 --2N+3
    ///  ╱    ╱ |     |     |
    /// 2N   ╱  7 --- 6 --2N+4
    ///  ╲  ╱  ╱
    ///   ○ - 8
    /// ```
    /// where `p` is the center face of the patch and node `0` is the irregular node.
    Irregular(Vec<Node>)

    // todo: add IrregularBoundary/Corner case of valence = 4
}

impl NodeConnectivity {

    /// Finds the nodes of the `msh` making up the patch of the `center_face`.
    pub fn find<T: RealField + Copy>(msh: &QuadMesh<T>, center_face: Face) -> NodeConnectivity {

        // Find all faces in the 1-ring neighborhood
        let faces = msh.faces.iter()
            .filter(|other| are_touching(&center_face, other))
            .collect_vec();

        // Sort the center face, depending on the patch type
        let center_sorted = match faces.len() {
            8 => center_face,
            5 => { // todo: 5 is not enough to check. Could this also be an interior irregular face?
                let node_irr = center_face.into_iter().enumerate()
                    .find_map(|(idx, node)| {
                        let next_idx = (idx + 1) % 4;
                        let next_node = center_face[next_idx];
                        (msh.valence(node) == 4 && msh.valence(next_node) != 4).then_some(next_node)
                    }).unwrap();
                sort_by_origin(center_face, node_irr)
            },
            3 => { // todo: 3 is maybe also not enough to check
                let node_irr = center_face.into_iter()
                    .find(|&n| msh.valence(n) == 2)
                    .unwrap();
                sort_by_origin(center_face, node_irr)
            },
            _ => {
                let node_irr = msh.irregular_node_of_face(center_face).unwrap();
                sort_by_origin(center_face, node_irr)
            }
        };

        if msh.is_regular(center_sorted) || msh.is_boundary_face(center_sorted) {
            let faces_sorted = Self::traverse_faces_regular(center_sorted, faces);
            if faces_sorted.len() == 9 {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (0, 3), (0, 2), (1, 2), (2, 2),
                    (6, 0), (6, 1), (7, 1), (8, 1),
                    (6, 3), (6, 2), (7, 2), (8, 2),
                ];
                let nodes = pick.map(|(face, node)| faces_sorted[face][node]);
                return NodeConnectivity::Regular(nodes);
            } else if faces_sorted.len() == 6 {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (3, 0), (3, 1), (4, 1), (5, 1),
                    (3, 3), (3, 2), (4, 2), (5, 2),
                ];
                let nodes = pick.map(|(face, node)| faces_sorted[face][node]);
                return NodeConnectivity::Boundary(nodes);
            } else if faces_sorted.len() == 4 {
                let pick = [
                    (0, 0), (0, 1), (1, 1),
                    (2, 0), (2, 1), (3, 1),
                    (2, 3), (2, 2), (3, 2),
                ];
                let nodes = pick.map(|(face, node)| faces_sorted[face][node]);
                return NodeConnectivity::Corner(nodes);
            }
        } else {
            let faces_sorted = Self::traverse_faces_irregular(center_sorted, faces);
            // Get faces at irregular node
            let node_irr = center_sorted[0];
            let n = msh.valence(node_irr);
            let mut inner_faces = vec![faces_sorted[1], faces_sorted[0]];
            inner_faces.extend_from_slice(&faces_sorted[7..n+5]);

            // Get nodes of inner faces by setting their uv origin to the irregular node
            let nodes_it = inner_faces.iter().flat_map(|&face| {
                let sorted = sort_by_origin(face, node_irr);
                once(sorted[3]).chain(once(sorted[2]))
            });

            let mut nodes = vec![node_irr];
            nodes.extend(nodes_it);

            // Get faces away from irregular node
            let outer_faces = &faces_sorted[2..=6].iter().enumerate().map(|(i, &face)| {
                sort_by_origin(face, nodes[i + 2])
            }).collect_vec();

            let pick = [
                (2, 2), (2, 1), (3, 1), (4, 1),
                (1, 2), (0, 2), (0, 3)
            ];
            let outer_nodes = pick.map(|(face, node)| outer_faces[face][node]);

            // Combine both
            nodes.extend_from_slice(&outer_nodes);
            return NodeConnectivity::Irregular(nodes)
        };

        unreachable!()
    }

    /// Traverses the given `faces` of a **regular** patch in lexicographical order around the already **sorted** `center_face`.
    /// Returns the sorted faces in a vector.
    /// The traversal order is as follows
    /// ```text
    /// +---+---+---+
    /// | 6 | 7 | 8 |
    /// +---+---+---+
    /// | 3 | p | 5 |
    /// +---+---+---+
    /// | 0 | 1 | 2 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face.
    fn traverse_faces_regular(center_face: Face, faces: Vec<&Face>) -> Vec<Face> {
        let center_edges = edges_of_face(center_face);
        let mut faces_sorted: Vec<Option<Face>> = vec![None; 9];
        faces_sorted[4] = Some(center_face);

        for &face in faces {
            // Check if faces are connected by an edge
            let edges = edges_of_face(face);
            let edge_res = center_edges.iter().find_position(|&&edge| edges.contains(&reverse_edge(edge)));
            if let Some((i, edge)) = edge_res {
                match i {
                    0 => faces_sorted[1] = Some(sort_by_node(face, edge[1], 2)),
                    1 => faces_sorted[5] = Some(sort_by_node(face, edge[1], 3)),
                    2 => faces_sorted[7] = Some(sort_by_node(face, edge[1], 0)),
                    3 => faces_sorted[3] = Some(sort_by_node(face, edge[1], 1)),
                    _ => unreachable!()
                }
                continue
            }

            // Check if faces are connected by a node
            let node_res = center_face.iter().find_position(|&&node| face.contains(&node));
            if let Some((i, &node)) = node_res {
                match i {
                    0 => faces_sorted[0] = Some(sort_by_node(face, node, 2)),
                    1 => faces_sorted[2] = Some(sort_by_node(face, node, 3)),
                    2 => faces_sorted[8] = Some(sort_by_node(face, node, 0)),
                    3 => faces_sorted[6] = Some(sort_by_node(face, node, 1)),
                    _ => unreachable!()
                }
                continue
            }

            panic!("Face {face:?} is not connected to the center face {center_face:?}");
        }

        faces_sorted.into_iter().flatten().collect()
    }

    /// Traverses the given `faces` of an **irregular** patch in clockwise order around the already **sorted** `center_face`.
    /// Returns the sorted faces in a vector.
    ///
    /// The traversal order is as follows
    /// ```text
    /// +---+---+---+
    /// | 2 | 3 | 4 |
    /// +---+---+---+
    /// | 1 | p | 5 |
    /// +---+---+---+
    /// | 8 | 7 | 6 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face.
    // todo: update and describe how faces are sorted
    fn traverse_faces_irregular(center_face: Face, mut faces: Vec<&Face>) -> Vec<Face> {
        // Get node of parametric origin
        let uv_origin = center_face[0];

        // Find face 1
        let (mut idx, mut found_face, mut found_edge) = faces.iter().enumerate()
            .find_map(|(i, other)| {
                let edges = edges_of_face(**other);
                // Find edge that is included in face and starts with start
                let edge_irr = edges.iter().find(|edge| edge[0] == uv_origin && center_face.contains(&edge[1]));
                edge_irr.map(|edge| (i, *other, *edge))
            }).unwrap();

        let found_sorted = sort_by_node(*found_face, uv_origin, 1);
        let mut faces_sorted = vec![center_face, found_sorted];

        while faces.len() > 1 {
            // Remove already visited face
            faces.swap_remove(idx);

            // Find next face
            let next_e = next_edge(found_edge, *found_face);
            let next_next_e = next_edge(next_e, *found_face);
            let next_inv = reverse_edge(next_e);
            let next_next_inv = reverse_edge(next_next_e);

            (idx, found_face, found_edge) = faces.iter().enumerate()
                .find_map(|(i, other)| {
                    let edges = edges_of_face(**other);
                    let found = edges.iter().find(|edge| **edge == next_inv || **edge == next_next_inv);
                    found.map(|edge| (i, *other, *edge))
                }).unwrap();

            // Save found face
            let found_sorted = sort_by_origin(*found_face, found_edge[0]); // todo: update sorting
            faces_sorted.push(found_sorted);
        }

        faces_sorted
    }

    /// Returns a slice containing the nodes.
    pub fn as_slice(&self) -> &[Node] {
        match self {
            NodeConnectivity::Regular(val) => val.as_slice(),
            NodeConnectivity::Boundary(val) => val.as_slice(),
            NodeConnectivity::Corner(val) => val.as_slice(),
            NodeConnectivity::Irregular(val) => val.as_slice(),
        }
    }
}

/// A patch of a quadrilateral mesh.
#[derive(Debug, Clone)]
pub struct Patch<'a, T: RealField> {
    pub(crate) nodes: NodeConnectivity,
    msh: &'a QuadMesh<T>
}

impl<'a, T: RealField + Copy> Patch<'a, T> {
    /// Finds the patch with the `center` face.
    ///
    /// The node `start` is the node between the faces `f`, `0`, `7` and `6`,
    /// i.e. the node of the center face closest to the parametric origin.
    pub fn find(msh: &'a QuadMesh<T>, center: Face) -> Self {
        Patch { nodes: NodeConnectivity::find(msh, center), msh }
    }
}

impl<'a, T: RealField + Copy> Patch<'a, T> {

    // todo: add docs
    pub fn faces(&self) -> Vec<Face> {
        todo!("implement function to calculate faces of a patch")
    }
    /// Finds and returns the irregular node index and its valence.
    /// Returns `None` if there is no irregular node.
    pub fn irregular_node(&self) -> Option<(usize, usize)> {
        let node_irr = match &self.nodes {
            NodeConnectivity::Irregular(val) => val[0],
            _ => return None
        };
        let n = self.msh.valence(node_irr);
        Some((node_irr, n))
    }

    /// Returns a matrix `(c1,...,cN)` over the coordinates of control points of this patch.
    pub fn coords(&self) -> OMatrix<T, U2, Dyn> {
        let points = self.nodes
            .as_slice()
            .iter()
            .map(|&n| self.msh.node(n).coords)
            .collect_vec();
        OMatrix::from_columns(&points)
    }
}

// /// An extended patch of a quadrilateral mesh.
// /// Consist of 3 regular patches and one irregular patch.
// pub struct ExtendedPatch<'a, T: RealField> {
//     /// The irregular patch.
//     patch_irr: Patch<'a, T>,
//     /// The 3 regular patches surrounding the irregular one.
//     patches_reg: [Patch<'a, T>; 3]
// }
//
// impl <'a, T: RealField + Copy> ExtendedPatch<'a, T> {
//
//     /// Finds the extended patch with the center `face`.
//     pub fn find(msh: &'a QuadMesh<T>, face: Face) -> Self {
//         // Find the irregular node
//         let node_irr = msh.irregular_node_of_face(face).expect("Face must be irregular!");
//
//         let [_, a, b, c] = sort_by_origin(face, node_irr);
//         let mut patch_faces = [Face::default(); 3];
//
//         for face in &msh.faces {
//             if face.contains(&a) && face.contains(&b) { patch_faces[0] = *face }
//             else if face.contains(&b) && face.contains(&c) { patch_faces[2] = *face }
//             else if face.contains(&b) { patch_faces[1] = *face }
//         }
//
//         // Create irregular patch and find nodes
//         let patch_irr = Patch::find(msh, face, node_irr);
//         let nodes_irr = patch_irr.nodes_irregular();
//
//         // Node indices for the orientation of regular patches
//         let starts = [5, 4, 3];
//         let origins = [7, 0, 1];
//
//         // Create and sort regular patches
//         let patches_reg = izip!(patch_faces, starts, origins)
//             .map(|(face, start_idx, origin_idx)| {
//                 let patch = Patch::find(msh, face, nodes_irr[start_idx]);
//                 patch.sort_faces_regular(nodes_irr[origin_idx])
//             })
//             .collect_array().unwrap();
//
//         ExtendedPatch {
//             patch_irr,
//             patches_reg
//         }
//     }
//
//     /// Returns the nodes of this extended patch in the following order
//     /// ```text
//     /// 2N+16-2N+15-2N+14-2N+13-2N+8
//     ///   |     |     |     |     |
//     /// 2N+7--2N+6--2N+5--2N+1--2N+9
//     ///   |     |     |     |     |
//     ///   2 --- 3 --- 4 --2N+2-2N+10
//     ///   |     |     |     |     |
//     ///   1 --- 0 --- 5 --2N+3-2N+11
//     ///  ╱    ╱ |     |     |     |
//     /// 2N   ╱  7 --- 6 --2N+4-2N+12
//     ///  ╲  ╱  ╱
//     ///   ○ - 8
//     /// ```
//     /// where `N` is the valence of the irregular node (`0` in the graphic).
//     pub fn nodes(&self) -> Vec<Node> {
//         let mut nodes = self.patch_irr.nodes_irregular();
//         let nodes1 = &self.patches_reg[0].nodes_regular();
//         let nodes2 = &self.patches_reg[1].nodes_regular();
//         let nodes3 = &self.patches_reg[2].nodes_regular();
//
//         nodes.extend_from_slice(&[nodes2[15], nodes2[11], nodes2[7], nodes2[3]]);
//         nodes.push(nodes1[3]);
//         nodes.extend_from_slice(&[nodes3[15], nodes3[14], nodes3[13], nodes3[12]]);
//
//         nodes
//     }
// }