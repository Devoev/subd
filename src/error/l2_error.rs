use std::borrow::Borrow;
use crate::element::traits::Element;
use crate::mesh::cell_topology::VolumetricElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{Mesh, MeshAllocator};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::{Quadrature, QuadratureOnMesh};
use crate::space::eval_basis::EvalBasisAllocator;
use crate::space::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::space::local::MeshElemBasis;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, OPoint, OVector, RealField};
use std::iter::{zip, Product, Sum};

/// L2-norm on a mesh.
pub struct L2Norm<'a, T, Verts, Cells> {
    /// Mesh defining the geometry discretization.
    pub(crate) msh: &'a Mesh<T, Verts, Cells>,
}

impl<'a, T, Verts, Cells> L2Norm<'a, T, Verts, Cells> {
    /// Constructs a new [`L2Norm`] on the given `msh`. 
    pub fn new(msh: &'a Mesh<T, Verts, Cells>) -> Self {
        L2Norm { msh }
    }

    /// Calculates the squared L2 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<Quadrature, U, N>(&self, u: U, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, N>,
          N: DimName,
          DefaultAllocator: MeshAllocator<T, Verts, Cells> + Allocator<N>
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_iter()
            .map(|elem| {
                // Evaluate function at quadrature nodes of element
                let u = quad.nodes_elem(&elem).map(&u);

                // Calculate L2 error on element
                let u_norm_squared = u.map(|u| u.norm_squared());
                quad.integrate_elem(&elem, u_norm_squared)
            })
            .sum::<T>()
    }

    /// Calculates the L2 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<Quadrature, U, N>(&self, u: U, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, N>,
          N: DimName,
          DefaultAllocator: MeshAllocator<T, Verts, Cells> + Allocator<N>
    {
        self.norm_squared(u, quad).sqrt()
    }

    /// Calculates the squared L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error_squared<Basis, Quadrature, U>(&self, uh: &LinCombination<T, Basis>, u: &U, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, Cells>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis> + MeshAllocator<T, Verts, Cells>
    {
        // Iterate over every element and calculate error element-wise
        self.msh.elem_cell_iter()
            .map(|(elem, cell)| {
                // Get geometrical and reference element
                let parametric_elem = elem.parametric_element();

                // Evaluate functions at quadrature nodes of element
                let uh = quad.nodes_ref(&parametric_elem).map(|x| uh.eval_on_elem(cell.borrow(), x));
                let u = quad.nodes_elem(&elem).map(u);

                // Calculate L2 error on element
                let du_norm_squared = zip(uh, u).map(|(uh, u)| (uh - u).norm_squared());
                quad.integrate_elem(&elem, du_norm_squared)
            })
            .sum::<T>()
    }

    /// Calculates the L2 error between the given discrete solution `uh` and the exact one `u`
    /// using the quadrature rule `quad`.
    pub fn error<Basis, Quadrature, U>(&self, uh: &LinCombination<T, Basis>, u: &U, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, Cells>,
          Quadrature: QuadratureOnMesh<T, Verts, Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>,
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis> + MeshAllocator<T, Verts, Cells>
    {
        self.error_squared(uh, u, quad).sqrt()
    }
}