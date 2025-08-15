//! This example visualizes the quadrature rule in the irregular subdivision setting,
//! by plotting the quadrature nodes in a truncated series of segments (see [SubdUnitSquare]).

use plotly::layout::Annotation;
use plotly::{Layout, Plot, Scatter};
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::quadrature::traits::Quadrature;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::patch::subd_unit_square::SubdUnitSquare;

fn main() {
    // Define quadrature
    let p = 2;
    let m_max = 1;
    let gauss_quad = GaussLegendreMulti::<f64, 2>::with_degrees([p, p]);
    let subd_quad = SubdUnitSquareQuad::new(gauss_quad.clone(), m_max);

    // Get irregular nodes
    let nodes_irr = subd_quad.nodes_elem(&SubdUnitSquare::Irregular);
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, (x, y)) in nodes_irr.enumerate() {
        let pos_trace = Scatter::new(vec![x], vec![y]);
        plot.add_trace(pos_trace);

        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(x)
            .y(y + 0.05);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot.show();
}