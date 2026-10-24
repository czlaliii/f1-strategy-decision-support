"""Dark dashboard theme: shared colours, DataTable styles and the Plotly template.

The same tokens are mirrored as CSS variables in assets/dashboard.css.
"""
import plotly.graph_objects as go
import plotly.io as pio

BACKGROUND = "#0b0c10"
SURFACE = "#15161c"
SURFACE_RAISED = "#1c1e26"
LINE = "#2a2c36"
TEXT = "#f2f3f5"
TEXT_MUTED = "#9aa0ab"
ACCENT = "#e10600"
GRID = "#262833"
FONT_FAMILY = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"

TABLE_STYLE = {"overflowX": "auto"}
TABLE_CELL = {
    "padding": "8px 10px",
    "textAlign": "left",
    "fontFamily": FONT_FAMILY,
    "fontSize": "13px",
    "backgroundColor": SURFACE,
    "color": TEXT,
    "border": "none",
    "borderBottom": f"1px solid {LINE}",
    "whiteSpace": "normal",
}
TABLE_HEADER = {
    "fontWeight": "700",
    "backgroundColor": SURFACE_RAISED,
    "color": TEXT_MUTED,
    "border": "none",
    "borderBottom": f"1px solid {LINE}",
    "fontSize": "11px",
    "textTransform": "uppercase",
    "letterSpacing": ".06em",
}
TABLE_ZEBRA = {"if": {"row_index": "odd"}, "backgroundColor": "#181920"}

TEMPLATE_NAME = "f1dark"


def register_plotly_template():
    """Make every Plotly figure dark by default; figures may still override."""
    template = go.layout.Template(pio.templates["plotly_dark"])
    template.layout.update(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_FAMILY, "color": TEXT},
        colorway=[ACCENT, "#ffffff", "#f5c400", "#3fa7ff", "#2ecc71", "#b38cff"],
        xaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": LINE},
        yaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": LINE},
        hoverlabel={"bgcolor": SURFACE_RAISED, "bordercolor": LINE, "font": {"color": TEXT}},
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME
