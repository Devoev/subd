use crate::subd_legacy::edge::{next_edge, reverse_edge};
use crate::subd_legacy::face::{are_touching, edges_of_face, sort_by_node, sort_by_origin};
use crate::subd_legacy::mesh::{Face, Node, QuadMesh};
use itertools::Itertools;
use nalgebra::{Dyn, OMatrix, RealField, U2};
use std::iter::once;

/// A patch of a quadrilateral mesh.
#[derive(Debug, Clone)]
pub struct Patch<'a, T: RealField> {
    /// Connectivity of the nodes of this patch.
    pub(crate) connectivity: NodeConnectivity,
    /// The mesh of this patch.
    msh: &'a QuadMesh<T>
}

impl<'a, T: RealField + Copy> Patch<'a, T> {
    /// Finds the patch with the `center` face in the given `msh`.
    pub fn find(msh: &'a QuadMesh<T>, center: Face) -> Self {
        Patch { connectivity: NodeConnectivity::find(msh, center), msh }
    }
}

impl<'a, T: RealField + Copy> Patch<'a, T> {
    /// Returns a matrix `(c1,...,cN)` over the coordinates of control points of this patch.
    pub fn coords(&self) -> OMatrix<T, U2, Dyn> {
        let points = self.connectivity
            .as_slice()
            .iter()
            .map(|&n| self.msh.node(n).coords)
            .collect_vec();
        OMatrix::from_columns(&points)
    }
}

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
    Irregular(Vec<Node>, usize)

    // todo: add IrregularBoundary/Corner case of valence = 4
}

impl NodeConnectivity {

    /// Finds the nodes of the `msh` making up the patch of the `center_face`.
    pub fn find<T: RealField + Copy>(msh: &QuadMesh<T>, center_face: Face) -> NodeConnectivity {
        match FaceConnectivity::find(msh, center_face) {
            FaceConnectivity::Regular(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (0, 3), (0, 2), (1, 2), (2, 2),
                    (6, 0), (6, 1), (7, 1), (8, 1),
                    (6, 3), (6, 2), (7, 2), (8, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face][node]);
                NodeConnectivity::Regular(nodes)
            }
            FaceConnectivity::Boundary(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1), (2, 1),
                    (3, 0), (3, 1), (4, 1), (5, 1),
                    (3, 3), (3, 2), (4, 2), (5, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face][node]);
                NodeConnectivity::Boundary(nodes)
            }
            FaceConnectivity::Corner(faces) => {
                let pick = [
                    (0, 0), (0, 1), (1, 1),
                    (2, 0), (2, 1), (3, 1),
                    (2, 3), (2, 2), (3, 2),
                ];
                let nodes = pick.map(|(face, node)| faces[face][node]);
                NodeConnectivity::Corner(nodes)
            }
            FaceConnectivity::Irregular(faces) => {
                // Get faces at irregular node
                let node_irr = faces[0][0];
                let mut nodes = vec![node_irr];

                let n = faces.len() - 5;
                let faces_irr = &faces[..n];
                let faces_reg = &faces[n..];

                // Get nodes around irregular faces
                let nodes_irr = faces_irr.iter()
                    .flat_map(|&face| [face[3], face[2]]);
                nodes.extend(nodes_irr);

                // Get faces away from irregular node
                let pick = [
                    (0, 2), (0, 1), (1, 1), (2, 1),
                    (3, 2), (4, 2), (4, 3)
                ];
                let nodes_reg = pick.map(|(face, node)| faces_reg[face][node]);

                // Combine both
                nodes.extend_from_slice(&nodes_reg);
                NodeConnectivity::Irregular(nodes, n)
            }
        }
    }

    /// Returns a slice containing the nodes.
    pub fn as_slice(&self) -> &[Node] {
        match self {
            NodeConnectivity::Regular(val) => val.as_slice(),
            NodeConnectivity::Boundary(val) => val.as_slice(),
            NodeConnectivity::Corner(val) => val.as_slice(),
            NodeConnectivity::Irregular(val, _) => val.as_slice(),
        }
    }

    /// Returns the valence of the (possibly) irregular node.
    /// For the regular case, the valence is 4.
    pub fn valence(&self) -> usize {
        match self {
            NodeConnectivity::Regular(_) => 4,
            NodeConnectivity::Boundary(_) => 3,
            NodeConnectivity::Corner(_) => 2,
            NodeConnectivity::Irregular(_, valence) => *valence
        }
    }
}

/// Connectivity of the ordered faces of a [`Patch`].
#[derive(Debug, Clone)]
pub enum FaceConnectivity {
    /// The regular interior case of valence `n=4`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// +---+---+---+
    /// | 6 | 7 | 8 |
    /// +---+---+---+
    /// | 3 | p | 5 |
    /// +---+---+---+
    /// | 0 | 1 | 2 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Regular([Face; 9]),

    /// The regular boundary case of valence `n=3`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// |   |   |   |
    /// +---+---+---+
    /// | 3 | 4 | 5 |
    /// +---+---+---+
    /// | 0 | p | 2 |
    /// +---+---+---+
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Boundary([Face; 6]),

    /// The regular corner case of valence `n=2`.
    /// The faces are ordered in lexicographical order
    /// ```text
    /// |   |   |
    /// +---+---+---
    /// | 2 | 3 |
    /// +---+---+---
    /// | p | 1 |
    /// +---+---+---
    /// ```
    /// where `p` is the center face of the patch.
    /// Each face is sorted such that the first node is the lower left one.
    Corner([Face; 4]),

    /// The irregular interior case of valence `n≠4`.
    /// The faces are ordered in the following order
    /// ```text
    ///   +-----+-----+-----+
    ///   | n+4 | n+3 |  n  |
    ///   +-----+-----+-----+
    ///   |  0  |  p  | n+1 |
    ///   +-----+-----+-----+
    ///  /     /|  2  | n+2 |
    /// + n-1 / +-----+-----+
    ///  \   / /
    ///   ○---+
    /// ```
    /// where `p` is the center face of the patch. The orientation of each face is as
    /// - Face `p`: Sorted such that the irregular node is the lower left one.
    /// - Faces `0..n-1`: Sorted such that the first node is the irregular one.
    /// - Faces `n..n+4`: Sorted such that the first node is the lower left one.
    Irregular(Vec<Face>)
}

impl FaceConnectivity {

    /// Finds the faces of the `msh` making up the patch of the `center_face`.
    pub fn find<T: RealField + Copy>(msh: &QuadMesh<T>, center_face: Face) -> FaceConnectivity {
        // Find all faces in the 1-ring neighborhood
        let faces = msh.faces.iter()
            .filter(|other| are_touching(&center_face, other))
            .collect_vec();

        if msh.is_regular(center_face) || msh.is_boundary_face(center_face) {
            match faces.len() {
                8 => {
                    let faces: [Face; 9] = Self::traverse_faces_regular(center_face, faces).try_into().unwrap();
                    FaceConnectivity::Regular(faces)
                },
                5 => {
                    let node_irr = center_face.into_iter().enumerate()
                        .find_map(|(idx, node)| {
                            let next_idx = (idx + 1) % 4;
                            let next_node = center_face[next_idx];
                            (msh.valence(node) == 4 && msh.valence(next_node) != 4).then_some(next_node)
                        }).unwrap();
                    let center_face = sort_by_origin(center_face, node_irr);
                    let faces: [Face; 6] = Self::traverse_faces_regular(center_face, faces).try_into().unwrap();
                    FaceConnectivity::Boundary(faces)
                },
                3 => {
                    let node_irr = center_face.into_iter()
                        .find(|&n| msh.valence(n) == 2)
                        .unwrap();
                    let center_face = sort_by_origin(center_face, node_irr);
                    let faces: [Face; 4] = Self::traverse_faces_regular(center_face, faces).try_into().unwrap();
                    FaceConnectivity::Corner(faces)
                },
                _ => panic!("Possibly add more options for faces.len()")
            }
        } else {
            let node_irr = msh.irregular_node_of_face(center_face).unwrap();
            let center_face = sort_by_origin(center_face, node_irr);
            let faces = Self::traverse_faces_irregular(center_face, faces);
            FaceConnectivity::Irregular(faces)
        }
    }

    /// Traverses the given `faces` of a **regular** patch in lexicographical order around the already **sorted** `center_face`
    /// and returns them in a vector.
    ///
    /// The traversal order is compatible with [`FaceConnectivity::Regular`], [`FaceConnectivity::Boundary`] and [`FaceConnectivity::Corner`].
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

    /// Traverses the given `faces` of an **irregular** patch around the already **sorted** `center_face`
    /// and returns them in a vector.
    ///
    /// The traversal order is compatible with [`FaceConnectivity::Irregular`].
    fn traverse_faces_irregular(center_face: Face, mut faces: Vec<&Face>) -> Vec<Face> {

        // Find irregular faces 0..n-1
        let mut faces_irregular = vec![center_face];
        let node_irr = center_face[0];

        let mut found_face = center_face;
        while faces.len() > 5 {
            // Find next face
            let next_edge = edges_of_face(found_face)[0];
            let face_idx = faces.iter()
                .position(|&&face| edges_of_face(face).contains(&reverse_edge(next_edge)))
                .expect("No face contains given edge. Check faces argument.");

            found_face = sort_by_origin(*faces.swap_remove(face_idx), node_irr);
            faces_irregular.push(found_face);
        }

        faces_irregular.rotate_right(1);

        // Find regular faces n..n+4
        let mut faces_regular = vec![Face::default(); 5];

        // Find faces connected to edges 1 and 2
        for (edge_dix, &edge) in edges_of_face(center_face)[1..=2].iter().enumerate() {
            let face_idx = faces.iter()
                .position(|&&face| edges_of_face(face).contains(&reverse_edge(edge)))
                .expect("No face contains given edge. Check faces argument.");
            let face = *faces.swap_remove(face_idx);
            match edge_dix {
                0 => faces_regular[1] = sort_by_node(face, edge[0], 0),
                1 => faces_regular[3] = sort_by_node(face, edge[0], 1),
                _ => unreachable!("Iterating over only 2 edges!")
            };
        }

        // Find faces connected to nodes 1, 2, 3
        for (node_idx, node) in center_face[1..=3].iter().enumerate() {
            let face_idx = faces.iter()
                .position(|&&face| face.contains(node))
                .expect("No face contains given node. Check faces argument.");
            let face = *faces.swap_remove(face_idx);
            match node_idx {
                0 => faces_regular[2] = sort_by_node(face, *node, 3),
                1 => faces_regular[0] = sort_by_node(face, *node, 0),
                2 => faces_regular[4] = sort_by_node(face, *node, 1),
                _ => unreachable!("Iterating over only 3 nodes!")
            }
        }

        // Combine faces
        faces_irregular.extend_from_slice(&faces_regular);
        faces_irregular
    }

    /// Returns a slice containing the faces.
    pub fn as_slice(&self) -> &[Face] {
        match self {
            FaceConnectivity::Regular(val) => val.as_slice(),
            FaceConnectivity::Boundary(val) => val.as_slice(),
            FaceConnectivity::Corner(val) => val.as_slice(),
            FaceConnectivity::Irregular(val) => val.as_slice(),
        }
    }
}