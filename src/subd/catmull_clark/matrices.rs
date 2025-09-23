use std::collections::{HashMap, HashSet, VecDeque};
use nalgebra::{dmatrix, dvector, matrix, DMatrix, DVector, Dyn, RealField, RowDVector, SMatrix, Schur};
use std::iter::once;
use std::sync::LazyLock;
use itertools::Itertools;
use nalgebra_sparse::CooMatrix;
use crate::cells::line_segment::UndirectedEdge;
use crate::cells::node::NodeIdx;
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use crate::subd::patch::quad_nodes_edge_one_ring::QuadNodesEdgeOneRing;
use crate::subd::patch::quad_nodes_one_ring::QuadNodesOneRing;

/// The `7✕8` matrix `S11` for extended catmull clark subdivision.
/// Taken from "Stam 1998" without trailing zeroes.
pub static S11: LazyLock<SMatrix<f64, 7, 8>> = LazyLock::new(|| {
    // a = 36, b = 6, c = 1, d = 24, e = 4, f = 16 (see "Stam 1998")
    // todo: implement special case n=3 (valence=3)
    matrix![
        1, 0, 0, 6, 36, 6, 0, 0;
        4, 0, 0, 4, 24, 24, 0, 0;
        6, 0, 0, 1, 6, 36, 6, 1;
        4, 0, 0, 0, 0, 24, 24, 4;
        4, 0, 0, 24, 24, 4, 0, 0;
        6, 1, 6, 36, 6, 1, 0, 0;
        4, 4, 24, 24, 0, 0, 0, 0;
   ].cast() / 64.0
});

/// The `7✕7` matrix `S12` for extended catmull clark subdivision.
/// Taken from "Stam 1998".
pub static S12: LazyLock<SMatrix<f64, 7, 7>> = LazyLock::new(|| {
    matrix![
        1, 6, 1, 0, 6, 1, 0;
        0, 4, 4, 0, 0, 0, 0;
        0, 1, 6, 1, 0, 0, 0;
        0, 0, 4, 4, 0, 0, 0;
        0, 0, 0, 0, 4, 4, 0;
        0, 0, 0, 0, 1, 6, 1;
        0, 0, 0, 0, 0, 4, 4;
    ].cast() / 64.0
});

/// The `9✕7 `matrix `S21` for extended catmull clark subdivision.
/// Taken from "Stam 1998" without trailing zeroes.
pub static S21: LazyLock<SMatrix<f64, 9, 7>> = LazyLock::new(|| {
    matrix![
        0, 0, 0, 0, 16, 0, 0;
        0, 0, 0, 0, 24, 4, 0;
        0, 0, 0, 0, 16, 16, 0;
        0, 0, 0, 0, 4, 24, 4;
        0, 0, 0, 0, 0, 16, 16;
        0, 0, 0, 4, 24, 0, 0;
        0, 0, 0, 16, 16, 0, 0;
        0, 0, 4, 24, 4, 0, 0;
        0, 0, 16, 16, 0, 0, 0
    ].cast() / 64.0
});

/// The `9✕7 `matrix `S22` for extended catmull clark subdivision. Taken from "Stam 1998".
pub static S22: LazyLock<SMatrix<f64, 9, 7>> = LazyLock::new(|| {
    matrix![
        16, 16, 0, 0, 16, 0, 0;
        4, 24, 4, 0, 4, 0, 0;
        0, 16, 16, 0, 0, 0, 0;
        0, 4, 24, 4, 0, 0, 0;
        0, 0, 16, 16, 0, 0, 0;
        4, 4, 0, 0, 24, 4, 0;
        0, 0, 0, 0, 16, 16, 0;
        0, 0, 0, 0, 4, 24, 4;
        0, 0, 0, 0, 0, 16, 16
    ].cast() / 64.0
});

/// The eigenvalue decomposition `V Λ V^-1` of the `2n+8 ✕ 2n+8` extended subdivision matrix for valence `5`.
pub static EV5: LazyLock<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>)> = LazyLock::new(|| {
    // todo: maybe parse from file?
    
    let v = dmatrix![
        0.0, -0.235702260395516, 0.0, -0.072266893173852, 0.0, 0.0, 0.0, -0.007709415184152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        0.0, -0.235702260395516, -0.243961933121501, 0.056025279561625, 0.017976449422152, 0.134014507715682, -0.043118897405735, 0.024860897076523, 0.0, 0.0, -0.060441593731306, -0.028839053569709, 0.0, 0.0, 0.032721873537340, -0.039563575783392, 0.0, 0.0;
        0.0, -0.235702260395516, -0.266135594621165, 0.137233347622761, 0.175212912723591, 0.071011084660949, -0.229564282059851, -0.060896512385331, 0.0, 0.0, 0.043569873185784, 0.128073342263455, 0.0, -0.0, -0.043793851975447, 0.135167991857878, 0.0, 0.0;
        0.0, -0.235702260395516, -0.075388383315107, 0.056025279561626, 0.192270884091081, -0.108420014234779, -0.039622854092398, 0.024860897076523, 0.0, 0.0, 0.048898276495734, -0.005092447971809, 0.0, 0.0, -0.008834739411311, -0.034163118841809, 0.0, 0.0;
        0.0, -0.235702260395516, 0.101654751529122, 0.137233347622762, 0.244279795923152, -0.185909433220358, 0.187573875283400, -0.060896512385331, 0.0, 0.0, -0.114067408885909, -0.120731467285000, 0.0, 0.0, 0.086198954993002, 0.028808613047201, 0.0, 0.0;
        0.0, -0.235702260395516, 0.197369349875858, 0.056025279561625, 0.100853491993127, 0.041412760376938, 0.107230022058511, 0.024860897076523, 0.0, 0.0, -0.018677479630081, 0.037078807474036, 0.0, 0.0, -0.038182042775278, 0.018449607177452, 0.0, 0.0;
        0.0, -0.235702260395516, 0.328961686184087, 0.137233347622761, -0.024239696078195, 0.229796697118821, -0.073936623550225, -0.060896512385331, 0.0, 0.0, 0.140995071400248, 0.067274275315321, 0.0, 0.0, 0.097067735955844, -0.117363289825964, 0.0, 0.0;
        0.0, -0.235702260395516, 0.197369349875858, 0.056025279561625, -0.129939998155213, 0.041412760376938, -0.133878966212675, 0.024860897076523, 0.0, 0.0, -0.018677479630079, -0.054902322783495, 0.0, 0.0, -0.014763060783713, 0.045565603156559, 0.0, 0.0;
        0.0, -0.235702260395516, 0.101654751529120, 0.137233347622762, -0.259260751976445, -0.185909433220358, -0.067941905365732, -0.060896512385331, 0.0, 0.0, -0.114067408885910, 0.011879403256292, 0.0, 0.0, -0.026207794961290, -0.101343115191152, 0.0, 0.0;
        0.0, -0.235702260395516, -0.075388383315109, 0.056025279561626, -0.181160827351147, -0.108420014234779, 0.109390695652295, 0.024860897076523, 0.0, 0.0, 0.048898276495733, 0.051755016850977, 0.0, 0.0, 0.029057969432963, 0.009711484291191, 0.0, 0.0;
        0.0, -0.235702260395516, -0.266135594621167, 0.137233347622761, -0.135992260592105, 0.071011084660948, 0.183868935692406, -0.060896512385331, 0.0, 0.0, 0.043569873185787, -0.086495553550068, 0.0, 0.0, -0.113265044012111, 0.054729800112037, 0.0, 0.0;
        1.0, -0.235702260395516, 0.186609380259204, 0.518455489312809, 0.448428633599144, -0.539307283378539, 0.544135686713489, -0.903837258714507, -0.978539384070722,-0.978157182557307, -0.632627337315649, -0.669586759484602, 0.775388285894988, -0.312573906298448, 0.636700898532374, 0.212792252691612, -0.632455532033677, 0.059056565114616;
        0.0, -0.235702260395516, 0.266700052497139, 0.340948865236273, 0.313824130466218, -0.192505745526441, 0.459463311269496, -0.166426827071982, -0.049943293469981,-0.130088015074872, -0.439479030521399, -0.153747581136452, -0.055384877532114, 0.390330626598835, -0.134978711386621, 0.534934247718751, -0.316227766012683, 0.435992878246394;
        0.0, -0.235702260395516, 0.361454490753610, 0.274897952593563, 0.184699131917028, 0.120343775708881, 0.311606026896339, 0.159051835071534, 0.024971646734990, 0.065044007537436, -0.105127167717458, 0.208700000711471, -0.0, 0.0, -0.389709057840771, 0.188307866946095, -0.316227766012682, 0.435992878246393;
        0.0, -0.235702260395516, 0.476525493191247, 0.340948865236272, 0.065955666489682, 0.402775051491243, 0.084986807766509, -0.166426827071979, -0.049943293469981, -0.130088015074872, 0.314444836957764, 0.401967302567749, 0.055384877532115, -0.390330626598835, -0.062523453050636, -0.439500973705362, -0.316227766012682, 0.435992878246393;
        0.0, -0.235702260395515, 0.027808898800859, 0.340948865236274, 0.393890810925447, -0.459197977606878, 0.198075097445044, -0.166426827071983, -0.127972958179242, -0.047758745390093, -0.069303403263185, -0.384758994576905, 0.443079020479608, -0.546617579748059, 0.410440246677290, -0.442872055314906, -0.316227766020992, -0.376936313131779;
        0.0, -0.235702260395515, -0.138063330081594, 0.274897952593565, 0.352117360368267, -0.315064095140345, -0.115142381778942, 0.159051835071534, 0.063986479089621, 0.023879372695047, 0.275226498225310, -0.028663108868423, 0.0, -0.0, -0.090172702191829, -0.348689485659587, -0.316227766020993, -0.376936313131781;
        0.0, -0.235702260395516, -0.311695795946635, 0.340948865236273, 0.330135654371542, -0.091294212199822, -0.399256326479776, -0.166426827071979, -0.127972958179242, -0.047758745390093, 0.396647171768708, 0.350668199778535, -0.443079020479608, 0.546617579748059, -0.456139220046598, 0.266158358908019, -0.316227766020994, -0.376936313131782;
    ];

    let v_inv = v.clone().try_inverse().expect("Matrix V of eigenvectors is invertible.");

    let lambda = dvector![
        1.562500000000000e-02,
        9.999999999999970e-01,
        5.499883545182982e-01,
        3.224744871391586e-01,
        5.499883545182962e-01,
        3.401073881743499e-01,
        3.401073881743501e-01,
        7.752551286084074e-02,
        3.125000000000001e-02,
        3.125000000000003e-02,
        1.837654875287817e-01,
        1.837654875287818e-01,
        6.250000000000000e-02,
        6.249999999999999e-02,
        1.136387697785712e-01,
        1.136387697785714e-01,
        1.250000000000001e-01,
        1.250000000000000e-01,
    ];
    let lambda = DMatrix::from_diagonal(&lambda);

    (v, lambda, v_inv)
});

/// Builds the `2n+1 ✕ 2n+1` subdivision matrix.
///
/// The ordering of nodes is taken from "Andersson 2016".
pub fn build_mat<T: RealField>(n: usize) -> DMatrix<T> {
    let weight = 1.0 / 16.0;
    let n_inv_squared = 1.0 / (n as f64).powi(2);

    // Assemble sub matrices
    // Faces to faces
    let ff = DMatrix::<f64>::from_diagonal_element(n, n, 4.0);

    // Edges to faces
    let mut ef = DMatrix::<f64>::from_element(n, n, 4.0);
    ef.fill_lower_triangle(0.0, 1);
    ef.fill_upper_triangle(0.0, 2);
    ef[(n - 1, 0)] = 4.0;

    // Vertex to faces
    let vf = DVector::from_element(n, 4.0);

    // Faces to edges
    let mut fe = DMatrix::<f64>::from_element(n, n, 1.0);
    fe.fill_lower_triangle(0.0, 2);
    fe.fill_upper_triangle(0.0, 1);
    fe[(0, n - 1)] = 1.0;

    // Edges to edges
    let mut ee = DMatrix::<f64>::from_element(n, n, 6.0);
    ee.fill_lower_triangle(1.0, 1);
    ee.fill_lower_triangle(0.0, 2);
    ee[(n - 1, 0)] = 1.0;
    ee.fill_upper_triangle_with_lower_triangle();

    // Vertex to edges
    let ve = DVector::from_element(n, 6.0);

    // Faces to vertex
    let fv = RowDVector::from_element(n, 4.0 * n_inv_squared);

    // Edges to vertex
    let ev = RowDVector::from_element(n, 24.0 * n_inv_squared);

    // Vertex to vertex
    let vv = (16.0 * (n as f64) - 28.0) / (n as f64);

    // Assemble total matrix
    let mut s = DMatrix::<f64>::zeros(2 * n + 1, 2 * n + 1);
    s.view_mut((0, 0), (n, n)).copy_from(&ff);
    s.view_mut((0, n), (n, n)).copy_from(&ef);
    s.view_mut((0, 2 * n), (n, 1)).copy_from(&vf);
    s.view_mut((n, 0), (n, n)).copy_from(&fe);
    s.view_mut((n, n), (n, n)).copy_from(&ee);
    s.view_mut((n, 2 * n), (n, 1)).copy_from(&ve);
    s.view_mut((2 * n, 0), (1, n)).copy_from(&fv);
    s.view_mut((2 * n, n), (1, n)).copy_from(&ev);
    s[(2 * n, 2 * n)] = vv;

    (s * weight).cast()
}

/// Reorders the columns and rows of the subdivision matrix to match the ordering of "Stam 1998".
/// The DOFs get reordered as
/// ```text
/// (F1,...,Fn,E1,...,En,V) -> (V,E1,F1,...,En,Fn)
/// ```
pub fn permute_matrix<T: RealField>(s: &DMatrix<T>) -> DMatrix<T> {
    // todo: possibly remove this method and directly build correct ordering
    let (r, _) = s.shape();
    let n = (r - 1) / 2;
    let face_edge_it = (0..n).flat_map(|i| once(i + n).chain(once(i)));
    let indices = once(2*n).chain(face_edge_it);

    // Permute columns
    let mut tmp = DMatrix::<T>::zeros(r, r);
    for (idx_new, idx_old) in indices.clone().enumerate() {
        tmp.set_column(idx_new, &s.column(idx_old));
    }

    // Permute rows
    let mut mat = DMatrix::<T>::zeros(r, r);
    for (idx_new, idx_old) in indices.enumerate() {
        mat.set_row(idx_new, &tmp.row(idx_old));
    }

    mat
}

/// Builds the extended `2n+8 ✕ 2n+8` and `2n+17 ✕ 2n+8` subdivision matrices.
///
/// The ordering of nodes is taken from "Stam 1998".
pub fn build_extended_mats<T: RealField>(n: usize) -> (DMatrix<T>, DMatrix<T>) {
    let s = permute_matrix(&build_mat(n));
    let mut a = DMatrix::<T>::zeros(2*n + 8, 2*n + 8);
    a.view_mut((0, 0), (2*n + 1, 2*n + 1)).copy_from(&s);
    a.fixed_view_mut::<7, 8>(2*n + 1, 0).copy_from(&S11.cast());
    a.fixed_view_mut::<7, 7>(2*n + 1, 2*n + 1).copy_from(&S12.cast());

    let mut a_bar = DMatrix::<T>::zeros(2*n + 17, 2*n + 8);
    a_bar.view_mut((0, 0), (2*n + 8, 2*n + 8)).copy_from(&a);
    a_bar.fixed_view_mut::<9, 7>(2*n + 8, 0).copy_from(&S21.cast());
    a_bar.fixed_view_mut::<9, 7>(2*n + 8, 2*n + 1).copy_from(&S22.cast());

    (a, a_bar)
}

/// Edge to midpoint index map.
type EdgeMidpoints = HashMap<UndirectedEdge, NodeIdx>;

/// Face to midpoint index map.
type FaceMidpoints = HashMap<QuadNodes, NodeIdx>;

/// Assembles the global subdivision matrix for the given `quad_msh`.
pub fn assemble_global_mat<T: RealField, const M: usize>(quad_msh: &QuadVertexMesh<T, M>) -> (CooMatrix<f64>, FaceMidpoints, EdgeMidpoints) {
    let edges = quad_msh.edges().collect_vec();
    let num_nodes = quad_msh.num_nodes();
    let mut mat = CooMatrix::new(quad_msh.num_elems() + edges.len() + num_nodes, num_nodes);
    let mut edge_midpoints: EdgeMidpoints = HashMap::new();
    let mut face_midpoints: FaceMidpoints = HashMap::new();

    // Apply node smoothing stencil
    for node_idx in 0..num_nodes {
        let one_ring = QuadNodesOneRing::find(quad_msh, NodeIdx(node_idx));
        match one_ring {
            QuadNodesOneRing::Regular([n0, n1, n2, n3, n4, n5, n6, n7, n8]) => {
                mat.push(node_idx, n0.0, 0.0625);
                mat.push(node_idx, n2.0, 0.0625);
                mat.push(node_idx, n6.0, 0.0625);
                mat.push(node_idx, n8.0, 0.0625);
                mat.push(node_idx, n1.0, 0.125);
                mat.push(node_idx, n3.0, 0.125);
                mat.push(node_idx, n5.0, 0.125);
                mat.push(node_idx, n7.0, 0.125);
                mat.push(node_idx, n4.0, 0.25);
            }
            QuadNodesOneRing::Boundary([n0, n1, n2, ..]) => {
                // Interpolatory smoothing
                mat.push(node_idx, n0.0, 0.125);
                mat.push(node_idx, n1.0, 0.75);
                mat.push(node_idx, n2.0, 0.125);
            }
            QuadNodesOneRing::Corner(_) => {
                // No smoothing
                mat.push(node_idx, node_idx, 1.0)
            }
            QuadNodesOneRing::Irregular(nodes, n) => {
                let beta = 3.0 / (2.0 * n as f64);
                let gamma = 1.0 / (4.0 * n as f64);
                let mut nodes = VecDeque::from(nodes);

                // Set weight for vertex node
                let v = nodes.pop_front().unwrap();
                mat.push(node_idx, v.0, 1.0 - beta - gamma);

                // Set weights for edge nodes
                for e in nodes.iter().step_by(2) {
                    mat.push(node_idx, e.0, beta / (n as f64));
                }

                // Set weights for face nodes
                for f in nodes.iter().skip(1).step_by(2) {
                    mat.push(node_idx, f.0, gamma / (n as f64));
                }
            }
        }
    }

    // Apply face-midpoint stencil
    let mut idx_offset = num_nodes;
    for (face_idx, face) in quad_msh.elems.iter().enumerate() {
        let [NodeIdx(a), NodeIdx(b), NodeIdx(c), NodeIdx(d)] = face.nodes();
        mat.push(face_idx + idx_offset, a, 0.25);
        mat.push(face_idx + idx_offset, b, 0.25);
        mat.push(face_idx + idx_offset, c, 0.25);
        mat.push(face_idx + idx_offset, d, 0.25);
        face_midpoints.insert(*face, NodeIdx(face_idx + idx_offset));
    }

    // Apply edge-midpoint stencil
    idx_offset += quad_msh.num_elems();
    for (edge_idx, edge) in edges.into_iter().enumerate() {
        let one_ring = QuadNodesEdgeOneRing::find(quad_msh, edge);
        match one_ring {
            QuadNodesEdgeOneRing::Regular([n0, n1, n2, n3, n4, n5]) => {
                mat.push(edge_idx + idx_offset, n0.0, 0.0625);
                mat.push(edge_idx + idx_offset, n1.0, 0.0625);
                mat.push(edge_idx + idx_offset, n2.0, 0.375);
                mat.push(edge_idx + idx_offset, n3.0, 0.375);
                mat.push(edge_idx + idx_offset, n4.0, 0.0625);
                mat.push(edge_idx + idx_offset, n5.0, 0.0625);
            }
            QuadNodesEdgeOneRing::Boundary([a, b]) => {
                mat.push(edge_idx + idx_offset, a.0, 0.5);
                mat.push(edge_idx + idx_offset, b.0, 0.5);
            }
        }
        edge_midpoints.insert(edge.into(), NodeIdx(edge_idx + idx_offset));
    }

    (mat, face_midpoints, edge_midpoints)
}