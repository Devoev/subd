// todo: this is work in progress

use crate::basis::eval::EvalBasis;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::bspline::cubic::CubicBspline;
use crate::mesh::traits::MeshTopology;
use crate::subd::mesh::CatmarkMesh;
use crate::subd::patch::{CatmarkPatch, CatmarkPatchNodes};
use nalgebra::{stack, Dyn, OMatrix, RealField, RowDVector, U2};
use std::vec;
use itertools::Itertools;
use crate::cells::topo::Cell;

/// Edge basis functions for Catmull-Clark subdivision.
pub struct CatmarkEdgeBasis<'a, T: RealField, const M: usize>(pub(crate) &'a CatmarkMesh<T, M>);

impl <'a, T: RealField, const M: usize> Basis for CatmarkEdgeBasis<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U2;

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.num_nodes() * 2) // todo: replace with num_edges
    }
}

impl <'a, T: RealField + Copy, const M: usize> LocalBasis<T, (T, T)> for CatmarkEdgeBasis<'a, T, M> {
    type Elem = &'a CatmarkPatchNodes; // todo: separate EdgePatch struct is required
    type ElemBasis = CatmarkPatchEdgeBasis;
    type GlobalIndices = vec::IntoIter<usize>;

    fn find_elem(&self, x: (T, T)) -> Self::Elem {
        todo!()
    }

    fn elem_basis(&self, elem: &Self::Elem) -> Self::ElemBasis {
        let patch = CatmarkPatch::from_msh(self.0, elem);
        match patch {
            CatmarkPatch::Regular(_) => CatmarkPatchEdgeBasis::Regular,
            CatmarkPatch::Boundary(_) => CatmarkPatchEdgeBasis::Boundary,
            CatmarkPatch::Corner(_) => CatmarkPatchEdgeBasis::Corner,
            _ => todo!("implement irregular")
        }
    }

    fn global_indices(&self, elem: &Self::Elem) -> Self::GlobalIndices {
        // todo: In order to give global indices for edge basis functions,
        //  the edges need a global ordering. This isn't implemented yet.
        //  The code below works, but should probably be updated, 
        //  because there aren't num_nodes * 2 edges
        
        let num_nodes = self.0.num_nodes();
        let idx_x = elem.nodes().iter().map(|node| node.0);
        let idx_y = elem.nodes().iter().map(|node| node.0 + num_nodes);
        let indices = idx_x.chain(idx_y).collect_vec();
        indices.into_iter()
    }
}

/// Edge basis functions on a Catmull-Clark patch.
pub enum CatmarkPatchEdgeBasis {
    Regular,
    Boundary,
    Corner
}

impl Basis for CatmarkPatchEdgeBasis {
    type NumBasis = Dyn;
    type NumComponents = U2;

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkPatchEdgeBasis::Regular => Dyn(32), // todo: should be 24?
            CatmarkPatchEdgeBasis::Boundary => Dyn(24),
            CatmarkPatchEdgeBasis::Corner => Dyn(18),
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T, (T, T)> for CatmarkPatchEdgeBasis {
    #[allow(clippy::toplevel_ref_arg)]
    fn eval(&self, x: (T, T)) -> OMatrix<T, Self::NumComponents, Self::NumBasis> {
        let (u, v) = x;
        match self {
            CatmarkPatchEdgeBasis::Regular => {
                let bu_du = CubicBspline::eval_smooth_deriv(u); // replace with Curry-Schoenberg like basis
                let bu = CubicBspline::eval_smooth(u);
                let bv_dv = CubicBspline::eval_smooth_deriv(v); // replace with Curry-Schoenberg like basis
                let bv = CubicBspline::eval_smooth(v);
                let b_du = RowDVector::from_row_slice(bv.kronecker(&bu_du).as_slice());
                let b_dv = RowDVector::from_row_slice(bv_dv.kronecker(&bu).as_slice());
                stack![
                    b_du, 0;
                    0, b_dv
                ]
            },
            CatmarkPatchEdgeBasis::Boundary => {
                let bu_du = CubicBspline::eval_smooth_deriv(u); // replace with Curry-Schoenberg like basis
                let bu = CubicBspline::eval_smooth(u);
                let bv_dv = CubicBspline::eval_interpolating_deriv(v); // replace with Curry-Schoenberg like basis
                let bv = CubicBspline::eval_interpolating(v);
                let b_du = RowDVector::from_row_slice(bv.kronecker(&bu_du).as_slice());
                let b_dv = RowDVector::from_row_slice(bv_dv.kronecker(&bu).as_slice());

                stack![
                    b_du, 0;
                    0, b_dv
                ]
            },
            CatmarkPatchEdgeBasis::Corner => {
                let bu_du = CubicBspline::eval_interpolating_deriv(u); // replace with Curry-Schoenberg like basis
                let bu = CubicBspline::eval_interpolating(u);
                let bv_dv = CubicBspline::eval_interpolating_deriv(v); // replace with Curry-Schoenberg like basis
                let bv = CubicBspline::eval_interpolating(v);
                let b_du = RowDVector::from_row_slice(bv.kronecker(&bu_du).as_slice());
                let b_dv = RowDVector::from_row_slice(bv_dv.kronecker(&bu).as_slice());

                stack![
                    b_du, 0;
                    0, b_dv
                ]
            }
        }
    }
}