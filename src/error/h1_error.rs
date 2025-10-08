use crate::error::l2_error::L2Norm;
use crate::mesh::cell_topology::VolumetricElementTopology;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{Mesh, MeshAllocator};
use crate::quadrature::pullback::PullbackQuad;
use crate::quadrature::traits::QuadratureOnMesh;
use crate::space::eval_basis::{EvalBasisAllocator, EvalGradAllocator};
use crate::space::grad::{GradBasis, GradBasisPullback};
use crate::space::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::space::local::{MeshElemBasis, MeshGradBasis};
use crate::space::Space;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, OPoint, OVector, RealField, U1};
use std::iter::{Product, Sum};

/// H1-norm on a mesh.
pub struct H1Norm<'a, T, Verts, Cells>(L2Norm<'a, T, Verts, Cells>);

impl<'a, T, Verts, Cells> H1Norm<'a, T, Verts, Cells> {
    /// Constructs a new [`H1Norm`] for the given `msh`.
    pub fn new(msh: &'a Mesh<T, Verts, Cells>) -> Self {
        H1Norm(L2Norm::new(msh))
    }

    /// Calculates the squared H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm_squared<Quadrature, U, UGrad>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          &'a Cells: VolumetricElementTopology<T, Verts>,
          Quadrature: QuadratureOnMesh<T, Verts, &'a Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, U1>,
          UGrad: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Verts::GeoDim>,
          DefaultAllocator: MeshAllocator<T, Verts, &'a Cells> + Allocator<Verts::GeoDim> + Allocator<U1>
    {
        // Calculate ||u||^2 + ||grad u||^2
        self.0.norm_squared(u, quad) + self.0.norm_squared(u_grad, quad)
    }

    /// Calculates the H1 norm of the given exact solution `u`
    /// using the quadrature rule `quad`.
    pub fn norm<Quadrature, U, UGrad>(&self, u: U, u_grad: UGrad, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          &'a Cells: VolumetricElementTopology<T, Verts>,
          Quadrature: QuadratureOnMesh<T, Verts, &'a Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, U1>,
          UGrad: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Verts::GeoDim>,
          DefaultAllocator: MeshAllocator<T, Verts, &'a Cells> + Allocator<Verts::GeoDim>
    {
       self.norm_squared(u, u_grad, quad).sqrt()
    }

    /// Calculates the squared H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`, using the quadrature rule `quad`.
    pub fn error_squared<Basis, Quadrature, U, UGrad>(&self, uh: &LinCombination<T, Basis>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          &'a Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, &'a Cells> + MeshGradBasis<T> + Clone, // todo: remove clone
          Basis::Coord<T>: Copy,
          Quadrature: QuadratureOnMesh<T, Verts, &'a Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>,
          UGrad: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Verts::GeoDim>,
        // todo: there are way to many allocator bounds. Fix when refactoring GradBasis
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalGradAllocator<Basis::LocalBasis>
          + EvalGradAllocator<GradBasis<Basis::LocalBasis>> + SelectCoeffsAllocator<GradBasis<Basis::LocalBasis>>
          + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis>
          + EvalFunctionAllocator<GradBasis<Basis>>
          + MeshAllocator<T, Verts, &'a Cells> + Allocator<Verts::GeoDim>
    {
        // Compute gradient of uh (todo: possibly add as input argument?)
        // todo: very ugly. When GradBasisPullback gets refactored, update this code!
        let grad_space_parametric = uh.space.clone().grad();
        let grad_basis = GradBasisPullback { msh: self.0.msh, grad_basis: grad_space_parametric.basis };
        let grad_space = Space::new(grad_basis);
        // let uh_grad = uh.clone().grad(&grad_space);
        let uh_grad = LinCombination::new(uh.coeffs.clone(), &grad_space).unwrap();

        // Calculate ||u - uh||^2 + ||grad u - grad uh||^2
        self.0.error_squared(uh, u, quad) + self.0.error_squared(&uh_grad, u_grad, quad)
    }

    /// Calculates the H1 error between the given discrete solution `uh` and the exact one `u`,
    /// with gradient `u_grad`,
    /// using the quadrature rule `quad`.
    pub fn error<Basis, Quadrature, U, UGrad>(&self, uh: &LinCombination<T, Basis>, u: &U, u_grad: &UGrad, quad: &PullbackQuad<Quadrature>) -> T
    where T: RealField + Copy + Product<T> + Sum<T>,
          Verts: VertexStorage<T>,
          &'a Cells: VolumetricElementTopology<T, Verts>,
          Basis: MeshElemBasis<T, Verts, &'a Cells> + MeshGradBasis<T> + Clone, // todo: remove clone
          Basis::Coord<T>: Copy,
          Quadrature: QuadratureOnMesh<T, Verts, &'a Cells>,
          U: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Basis::NumComponents>,
          UGrad: Fn(OPoint<T, Verts::GeoDim>) -> OVector<T, Verts::GeoDim>,
    // todo: there are way to many allocator bounds. Fix when refactoring GradBasis
          DefaultAllocator: EvalBasisAllocator<Basis::LocalBasis> + EvalGradAllocator<Basis::LocalBasis>
          + EvalGradAllocator<GradBasis<Basis::LocalBasis>> + SelectCoeffsAllocator<GradBasis<Basis::LocalBasis>>
          + EvalFunctionAllocator<Basis> + SelectCoeffsAllocator<Basis::LocalBasis>
          + EvalFunctionAllocator<GradBasis<Basis>>
          + MeshAllocator<T, Verts, &'a Cells> + Allocator<Verts::GeoDim>
    {
        self.error_squared(uh, u, u_grad, quad).sqrt()
    }
}