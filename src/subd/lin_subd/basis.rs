//! Piecewise-linear basis functions.
//! todo: move this elsewhere. For example a separate FE module?

use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::local::MeshBasis;
use crate::basis::traits::Basis;
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use nalgebra::{matrix, Dyn, OMatrix, RealField, U1, U2, U4};
use numeric_literals::replace_float_literals;
use crate::basis::space::Space;

/// Space spanned by piecewise-linear basis functions on a quad mesh.
pub type PlSpaceQuad<'a, T, const M: usize> = Space<T, PlBasisQuad<'a, T, M>, 2>;

/// Piecewise-linear basis functions on a quadrilateral mesh.
#[derive(Debug, Clone)]
pub struct PlBasisQuad<'a, T: RealField, const M: usize>(pub &'a QuadVertexMesh<T, M>);

impl <'a, T: RealField, const M: usize> Basis for PlBasisQuad<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U1;
    type Coord<_T> = (_T, _T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.num_nodes())
    }
}

impl <'a, T: RealField + Copy, const M: usize> MeshBasis<T> for PlBasisQuad<'a, T, M> {
    type Cell = &'a QuadNodes;
    type LocalBasis = LinBasisQuad;
    type GlobalIndices = impl Iterator<Item = usize> + Clone;

    fn local_basis(&self, _elem: &Self::Cell) -> Self::LocalBasis {
        LinBasisQuad
    }

    fn global_indices(&self, elem: &Self::Cell) -> Self::GlobalIndices {
        elem.0.into_iter().map(|n| n)
    }
}

/// Linear basis functions on a quadrilateral.
#[derive(Debug, Clone)]
pub struct LinBasisQuad;

impl Basis for LinBasisQuad {
    type NumBasis = U4;
    type NumComponents = U1;
    type Coord<T> = (T, T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        U4
    }
}

impl <T: RealField + Copy> EvalBasis<T> for LinBasisQuad {
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    fn eval(&self, (u, v): (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        matrix![
            (1.0 - u) * (1.0 - v), // (1-u)(1-v)
            u         * (1.0 - v), //     u(1-v)
            u         * v,         //     uv
            (1.0 - u) * v          // (1-u)v
        ]
    }
}

impl <T: RealField + Copy> EvalGrad<T, 2> for LinBasisQuad {
    fn eval_grad(&self, (u, v): (T, T)) -> OMatrix<T, U2, Self::NumBasis> {
        matrix![
            -(T::one() - v), (T::one() - v), v, -v; // u-derivatives
            -(T::one() - u), -u, u, (T::one() - u); // v-derivatives
        ]
    }
}