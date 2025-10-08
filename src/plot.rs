use std::borrow::Borrow;
use crate::cells::node::Node;
use crate::cells::quad::QuadNodes;
use crate::cells::traits::{CellConnectivity, ElemOfCell, ToElement};
use crate::diffgeo::chart::Chart;
use crate::element::traits::{ElemAllocator, ElemCoord, Element};
use crate::mesh::cell_topology::{CellOfMesh, CellTopology, VolumetricElementTopology};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::vertex_storage::VertexStorage;
use crate::mesh::{ElemOfMesh, Mesh, MeshAllocator};
use itertools::Itertools;
use nalgebra::{DefaultAllocator, Point, Scalar, U2};
use plotly::common::{ColorScale, ColorScalePalette};
use plotly::layout::Annotation;
use plotly::{Layout, Plot, Scatter, Surface};
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::iter::zip;

/// Plots the given `faces` of a 2D quad-vertex `msh`.
pub fn plot_faces(msh: &QuadVertexMesh<f64, 2>, faces: impl Iterator<Item=QuadNodes>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, face) in faces.enumerate() {
        for edge in face.edges() {
            let line = edge.to_element(&msh.coords);
            let [pi, pj] = line.vertices;
            let edge = Scatter::new(vec![pi.x, pj.x], vec![pi.y, pj.y]);
            plot.add_trace(edge)
        }
        let quad = face.to_element(&msh.coords);
        let center = quad.centroid();
        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(center.x)
            .y(center.y);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot
}

/// Plots the given `nodes` of a 2D quad-vertex `msh`.
pub fn plot_nodes(msh: &QuadVertexMesh<f64, 2>, nodes: impl Iterator<Item=Node>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, node) in nodes.enumerate() {
        let pos = msh.coords.vertex(node);
        let pos_trace = Scatter::new(vec![pos.x], vec![pos.y]);
        plot.add_trace(pos_trace);

        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(pos.x)
            .y(pos.y + 0.05);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot
}

/// Plots the function `f` on the given `elem`/`patch`
/// using `num` evaluation points per parametric direction.
pub fn plot_fn_elem<Cell, F, Discretize>(elem: &ElemOfCell<f64, Cell, U2>, cell: &Cell, f: &F, num: usize, mesh_grid: &Discretize) -> Plot
    where Cell: ToElement<f64, U2>,
          ElemCoord<f64, ElemOfCell<f64, Cell, U2>>: From<(f64, f64)>,
          F: Fn(&Cell, ElemCoord<f64, ElemOfCell<f64, Cell, U2>>) -> f64,
          Discretize: Fn(&ElemOfCell<f64, Cell, U2>, usize) -> (Vec<f64>, Vec<f64>),
          DefaultAllocator: ElemAllocator<f64, ElemOfCell<f64, Cell, U2>>,
{
    let mut plot = Plot::new();
    let phi = elem.geo_map();
    let (u_range, v_range) = mesh_grid(elem, num);

    let mut x = vec![vec![0.0; num]; num];
    let mut y = vec![vec![0.0; num]; num];
    let mut z = vec![vec![0.0; num]; num];

    for (i, &u) in u_range.iter().enumerate() {
        for (j, &v) in v_range.iter().enumerate() {
            // Evaluate patch
            let pos = phi.eval((u, v).into());

            // Set coordinates
            x[i][j] = pos[0];
            y[i][j] = pos[1];
            z[i][j] = f(cell, (u, v).into());
        }
    }

    let surface = Surface::new(z).x(x).y(y)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
        .show_scale(false);
    plot.add_trace(surface.clone());
    plot
}

/// Plots the function `f` on the entire `msh`
/// using `num` evaluation points per parametric direction per element.
pub fn plot_fn_msh<Verts, Cells, F, Discretize>(msh: &Mesh<f64, Verts, Cells>, f: &F, num: usize, mesh_grid: Discretize) -> Plot
    where Verts: VertexStorage<f64, GeoDim = U2> + Clone,  // todo: remove cloning
          Cells: VolumetricElementTopology<f64, Verts> + Clone,
          <Cells as CellTopology>::Cell: ToElement<f64, U2>, // fixme: why is this bound required? should VolumetricElementTopology enforce this?
          ElemCoord<f64, ElemOfMesh<f64, Verts, Cells>>: From<(f64, f64)>,
          F: Fn(&CellOfMesh<Cells>, ElemCoord<f64, ElemOfMesh<f64, Verts, Cells>>) -> f64,
          Discretize: Fn(&ElemOfMesh<f64, Verts, Cells>, usize) -> (Vec<f64>, Vec<f64>),
          DefaultAllocator: MeshAllocator<f64, Verts, Cells>
{
    let mut plot = Plot::new();

    for (elem, cell) in msh.elem_cell_iter() {
        let elem_plt = plot_fn_elem(&elem, cell.borrow(), f, num, &mesh_grid);
        plot.add_traces(elem_plt.data().iter().cloned().collect_vec());
    }

    plot
}

// #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
// pub fn eval_at_vertices<'a, T, Msh, B>(msh: &'a Msh, fh: LinCombination<T, B, 2>)
//     where T: RealField,
//           Msh: Mesh<'a, T, 2, 2>,
//           Msh::GeoElem: HasBasisCoord<T, B>,
//           B: MeshBasis<T, Cell= Msh::Elem, Coord<T> = (T, T)>,
//           // B::ElemBasis: Basis<Coord<T> = (T, T)>,
//           DefaultAllocator: CellAllocator<T, Msh::GeoElem> + EvalBasisAllocator<B::LocalBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::LocalBasis>
// {
//     let vertices_to_err = msh.elem_iter()
//         .flat_map(|elem| {
//             let patch = msh.geo_elem(&elem);
//             let phi = patch.geo_map();
//             [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)].map(|uv| {
//                 let p = phi.eval(uv.clone());
//                 let err = fh.eval_on_elem(&elem, uv).norm();
//                 (p, err)
//             })
//         })
//         .collect_vec();
//
//     todo!("")
// }

/// Writes the coordinates of control points `coords` into a `file`.
pub fn write_coords<T: Scalar + Display, const D: usize>(coords: impl Iterator<Item = Point<T, D>>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for p in coords {
        let str = p.coords.iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{str}")?;
    }
    Ok(())
}

/// Writes the element connectivity of `elems` into a `file`.
pub fn write_connectivity<C: CellConnectivity<Node = Node>>(elems: impl Iterator<Item = C>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for elem in elems {
        let str = elem.nodes().iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{str}")?;
    }
    Ok(())
}

/// Writes all `(x,f(x))` pairs into a `file`,
/// where `x` are the coordinates `coords` and `f` a function.
pub fn write_coords_with_fn<T: Scalar + Display, const D: usize>(coords: impl Iterator<Item = Point<T, D>>, f: impl Iterator<Item = T>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for (p, f) in zip(coords, f) {
        let coord = p.coords.iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{coord} {f}")?;
    }
    Ok(())
}