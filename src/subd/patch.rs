use crate::cells;
use crate::cells::node::NodeIdx;
use crate::cells::quad::QuadTopo;
use crate::mesh::elem_vertex_topo::QuadVertex;
use nalgebra::{Const, DimNameSub, U2};

/// Topology of a Catmull-Clark surface patch.
#[derive(Clone, Debug)]
pub enum CatmullClarkPatchTopology {
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
    Regular([NodeIdx; 16]),

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
    Boundary([NodeIdx; 12]),

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
    Corner([NodeIdx; 9]),

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
    Irregular(Vec<NodeIdx>, usize)

    // todo: add IrregularBoundary/Corner case of valence = 4
}

impl CatmullClarkPatchTopology {
    /// Finds the [`CatmullClarkPatchTopology`] in the given quad-vertex topology `msh`.
    /// The center face `p` is given by `quad`.
    pub fn find(msh: &QuadVertex, quad: &QuadTopo) -> Self {
        todo!("Copy from subd_legacy::patch or re-implement")    
    }
    
    /// Returns a slice containing the nodes.
    pub fn as_slice(&self) -> &[NodeIdx] {
        match self {
            CatmullClarkPatchTopology::Regular(val) => val.as_slice(),
            CatmullClarkPatchTopology::Boundary(val) => val.as_slice(),
            CatmullClarkPatchTopology::Corner(val) => val.as_slice(),
            CatmullClarkPatchTopology::Irregular(val, _) => val.as_slice(),
        }
    }
}

impl cells::topo::Cell<U2> for CatmullClarkPatchTopology {
    fn nodes(&self) -> &[NodeIdx] {
        self.as_slice()
    }

    fn is_connected<const M: usize>(&self, other: &Self) -> bool
    where
        U2: DimNameSub<Const<M>>
    {
        todo!()
    }
}