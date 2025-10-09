// todo: this is work in progress

use crate::space::eval_basis::EvalBasis;
use crate::space::local::MeshBasis;
use crate::space::basis::BasisFunctions;
use crate::bspline::cubic::CubicBspline;
use crate::mesh::cell_topology::CellTopology;
use crate::subd::catmull_clark::mesh::CatmarkMesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};
use itertools::Itertools;
use nalgebra::{stack, Dyn, OMatrix, RealField, RowDVector, U2};
use std::vec;
use crate::cells::traits::Cell;

/// Edge basis functions for Catmull-Clark subdivision.
pub struct CatmarkEdgeBasis<'a, T: RealField, const M: usize>(pub(crate) &'a CatmarkMesh<T, M>);

impl <'a, T: RealField, const M: usize> BasisFunctions for CatmarkEdgeBasis<'a, T, M> {
    type NumBasis = Dyn;
    type NumComponents = U2;
    type ParametricDim = U2;
    type Coord<_T> = (_T, _T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        Dyn(self.0.num_nodes() * 2) // todo: replace with num_edges
    }
}

impl <'a, T: RealField + Copy, const M: usize> MeshBasis<T> for CatmarkEdgeBasis<'a, T, M> {
    type Cell = CatmarkPatchNodes; // todo: separate EdgePatch struct is required
    type LocalBasis = CatmarkPatchEdgeBasis;
    type GlobalIndices = vec::IntoIter<usize>;

    fn local_basis(&self, cell: &Self::Cell) -> Self::LocalBasis {
        match cell {
            CatmarkPatchNodes::Regular(_) => CatmarkPatchEdgeBasis::Regular,
            CatmarkPatchNodes::Boundary(_) => CatmarkPatchEdgeBasis::Boundary,
            CatmarkPatchNodes::Corner(_) => CatmarkPatchEdgeBasis::Corner,
            _ => todo!("implement irregular")
        }
    }

    fn global_indices(&self, cell: &Self::Cell) -> Self::GlobalIndices {
        // todo: In order to give global indices for edge basis functions,
        //  the edges need a global ordering. This isn't implemented yet.
        //  The code below works, but should probably be updated, 
        //  because there aren't num_nodes * 2 edges
        
        let num_nodes = self.0.num_nodes();
        let mut idx_x = cell.nodes().to_vec();
        let idx_y = cell.nodes().iter().map(|node| node + num_nodes);
        idx_x.extend(idx_y);
        idx_x.into_iter()
    }
}

/// Edge basis functions on a Catmull-Clark patch.
pub enum CatmarkPatchEdgeBasis {
    Regular,
    Boundary,
    Corner
}

impl BasisFunctions for CatmarkPatchEdgeBasis {
    type NumBasis = Dyn;
    type NumComponents = U2;
    type ParametricDim = U2;
    type Coord<T> = (T, T);

    fn num_basis_generic(&self) -> Self::NumBasis {
        match self {
            CatmarkPatchEdgeBasis::Regular => Dyn(32), // todo: should be 24?
            CatmarkPatchEdgeBasis::Boundary => Dyn(24),
            CatmarkPatchEdgeBasis::Corner => Dyn(18),
        }
    }
}

impl <T: RealField + Copy> EvalBasis<T> for CatmarkPatchEdgeBasis {
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