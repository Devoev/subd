use crate::basis::eval::{EvalBasis, EvalGrad};
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use crate::cells::topo::Cell;
use crate::subd::mesh::CatmarkMesh;
use crate::subd::patch::{CatmarkPatch, CatmarkPatchNodes};
use itertools::Itertools;
use nalgebra::{Dyn, Matrix, OMatrix, RealField, U1, U2};
use std::vec;

/// Basis functions for Catmull-Clark subdivision.
pub struct CatmarkBasis<'a, T: RealField, const M: usize>(&'a CatmarkMesh<T, M>);

impl <'a, T: RealField, const M: usize> Basis for CatmarkBasis<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.topology.num_nodes)
    }
}

impl <'a, T: RealField + Copy, const M: usize> LocalBasis<T, (T, T)> for CatmarkBasis<'a, T, M> {
    type Elem = CatmarkPatchNodes;
    type ElemBasis = CatmarkPatchBasis;
    type GlobalIndices = vec::IntoIter<usize>;

    fn find_elem(&self, x: (T, T)) -> Self::Elem {
        todo!()
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        // todo: move this to `elem` function on CellTopo or else
        let patch = CatmarkPatch::from_msh(self.0, elem);
        patch.basis()
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        // todo: possibly remove allocation
        let indices = elem.nodes().iter().map(|node| node.0).collect_vec();
        indices.into_iter()
    }
}

/// Basis functions on a Catmull-Clark patch.
pub enum CatmarkPatchBasis {
    Regular,
    Boundary,
    Corner,
    Irregular(usize) // todo: valence parameter
}

impl CatmarkPatchBasis {
    /// Returns a pair of [`CubicBspline`] for both parametric directions.
    fn bases(&self) -> (CubicBspline, CubicBspline) {
        match self {
            CatmarkPatchBasis::Regular => {
                (CubicBspline::Smooth, CubicBspline::Smooth)
            }
            CatmarkPatchBasis::Boundary => {
                (CubicBspline::Smooth, CubicBspline::Interpolating)
            }
            CatmarkPatchBasis::Corner => {
                (CubicBspline::Interpolating, CubicBspline::Interpolating)
            }
            CatmarkPatchBasis::Irregular(_) => {
                todo!()
            }
        }
    }
}

impl Basis for CatmarkPatchBasis {
    type NumBasis = Dyn;
    type NumComponents = U1;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkPatchBasis::Regular => Dyn(16),
            CatmarkPatchBasis::Boundary => Dyn(12),
            CatmarkPatchBasis::Corner => Dyn(9),
            CatmarkPatchBasis::Irregular(n) => todo!("dependent on valence")
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmarkPatchBasis {
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkPatchBasis::Irregular(n) => { todo!() },
            _ => {
                let (bu, bv) = self.bases();
                bv.eval(v).kronecker(&bu.eval(u))
            }
        }
    }
}

impl <T: RealField + Copy> EvalGrad<T, (T, T), 2> for CatmarkPatchBasis {
    fn eval_grad(&self, x: (T, T)) -> OMatrix<T, U2, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkPatchBasis::Irregular(n) => { todo!() },
            _ => {
                let (basis_u, basis_v) = self.bases();
                let bu = basis_u.eval(u);
                let bu_du = basis_u.eval_grad(u);
                let bv = basis_v.eval(v);
                let bv_dv = basis_v.eval_grad(v);
                let b_du = bv.kronecker(&bu_du);
                let b_dv = bv_dv.kronecker(&bu);
                Matrix::from_rows(&[b_du, b_dv])
            }
        }
    }
}