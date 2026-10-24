"""Race progression graph: every driver's position lap by lap, up to the replay lap.

Lap 0 is the starting grid. Lines use team colours (the second driver of a
team is dashed), pit stops are dots on the in-lap, retirements end with a
cross, and SC/VSC/red-flag laps are shaded in the background.

When race control sends the field through the pit lane under a safety car,
OpenF1 records each pass as a pit stop and a new stint; a pass without a
compound change is therefore not shown as a stop.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.theme import TEXT_MUTED

STATUS_BAND_COLORS = {"SC": "rgba(255, 214, 0, .16)", "VSC": "rgba(255, 176, 32, .14)", "RED": "rgba(225, 6, 0, .22)"}
STATUS_BAND_LABELS = {"SC": "SC", "VSC": "VSC", "RED": "PIROS"}
DEFAULT_COLOR = "#808080"


def progression_data(laps, grid=None, stints=None, retirements=None, track_status=None, lap_of=None):
    """Position per driver and lap (lap 0 = grid), pit laps, retirements and status bands.

    laps: replay lap table with Driver, LapNumber, Position (at lap end) and
    TimeSeconds (lap end), so the graph shows only laps finished by the replay time.
    grid: Driver -> grid slot. lap_of: converts replay seconds to a race lap.
    """
    rows = laps.dropna(subset=["Position", "LapNumber", "TimeSeconds"])[["Driver", "LapNumber", "Position", "TimeSeconds"]]
    rows = rows.rename(columns={"LapNumber": "Lap"}).astype({"Lap": int, "Position": int})
    if grid:
        start = pd.DataFrame([
            {"Driver": d, "Lap": 0, "Position": int(p), "TimeSeconds": 0.0} for d, p in grid.items() if pd.notna(p)
        ])
        rows = pd.concat([start, rows], ignore_index=True)
    positions = rows.sort_values(["Driver", "Lap"]).drop_duplicates(["Driver", "Lap"]).reset_index(drop=True)

    lap_ends = {
        (driver, int(lap)): end
        for driver, lap, end in laps[["Driver", "LapNumber", "TimeSeconds"]].dropna().itertuples(index=False)
    }
    pit_lane_windows = [
        (period["pit_lane_from"], period["end"] if period["end"] is not None else float("inf"))
        for period in (track_status or [])
        if period.get("pit_lane_from") is not None
    ]

    def pit_lane_pass(driver, lap, old, new):
        end = lap_ends.get((driver, lap))
        return old == new and end is not None and any(start <= end <= stop for start, stop in pit_lane_windows)

    pit_laps = {}
    if stints is not None and not stints.empty:
        for driver, group in stints.sort_values("stint_number").groupby("Driver"):
            compounds = group["compound"].tolist()
            # A stint ends on the in-lap of the following pit stop.
            pit_laps[driver] = [
                int(lap) for lap, old, new in zip(group["lap_end"].iloc[:-1], compounds[:-1], compounds[1:])
                if pd.notna(lap) and not pit_lane_pass(driver, int(lap), old, new)
            ]

    retired = {driver: int(info["lap"]) for driver, info in (retirements or {}).items()}

    bands = []
    if track_status and lap_of is not None:
        for period in track_status:
            end = period["end"] if period["end"] is not None else period["start"]
            bands.append({"status": period["status"], "start": lap_of(period["start"]), "end": lap_of(end)})
    return {"positions": positions, "pit_laps": pit_laps, "retired": retired, "bands": bands}


def _line_styles(drivers, team_of):
    """Dash the second driver of each team so teammates stay distinguishable."""
    seen, styles = {}, {}
    for driver in sorted(drivers):
        team = team_of.get(driver, driver)
        styles[driver] = "dash" if seen.get(team) else "solid"
        seen[team] = True
    return styles


def build_progression_figure(data, current_seconds, colors, team_of=None, highlight=None):
    """Laps finished by current_seconds; the highlighted driver is drawn bold on top."""
    positions = data["positions"]
    visible = positions[positions["TimeSeconds"] <= current_seconds]
    figure = go.Figure()
    if visible.empty:
        figure.add_annotation(text="Még nincs pozícióadat.", showarrow=False, font={"color": TEXT_MUTED})
        figure.update_layout(xaxis={"visible": False}, yaxis={"visible": False})
        return figure
    current_lap = int(visible["Lap"].max())

    for band in data["bands"]:
        if band["start"] > current_lap:
            continue
        figure.add_vrect(
            x0=band["start"] - 0.5, x1=min(band["end"], current_lap) + 0.5, line_width=0,
            fillcolor=STATUS_BAND_COLORS[band["status"]], layer="below",
            annotation_text=STATUS_BAND_LABELS[band["status"]], annotation_position="top left",
            annotation_font={"size": 9, "color": TEXT_MUTED},
        )

    styles = _line_styles(visible["Driver"].unique(), team_of or {})
    field_size = int(max(20, visible["Position"].max()))
    # Highlighted driver last, so its line is on top.
    order = sorted(visible["Driver"].unique(), key=lambda d: d == highlight)
    for driver in order:
        mine = visible[visible["Driver"] == driver].sort_values("Lap")
        color = colors.get(driver, DEFAULT_COLOR)
        bold = driver == highlight
        faded = highlight is not None and not bold
        figure.add_trace(go.Scatter(
            x=mine["Lap"], y=mine["Position"], mode="lines", name=driver,
            line={"color": color, "width": 4 if bold else 2, "dash": styles[driver], "shape": "linear"},
            opacity=0.35 if faded else 1.0,
            customdata=np.full(len(mine), driver),
            hovertemplate="%{customdata} · %{x}. kör · P%{y}<extra></extra>",
        ))
        pits = mine[mine["Lap"].isin(data["pit_laps"].get(driver, []))]
        if len(pits):
            figure.add_trace(go.Scatter(
                x=pits["Lap"], y=pits["Position"], mode="markers", hoverinfo="skip", showlegend=False,
                marker={"size": 7 if not bold else 9, "color": "#0b0c10", "line": {"width": 2, "color": color}},
                opacity=0.35 if faded else 1.0,
            ))
        last = mine.iloc[-1]
        out = data["retired"].get(driver)
        if out is not None and out <= current_lap:
            figure.add_trace(go.Scatter(
                x=[last["Lap"]], y=[last["Position"]], mode="markers", hoverinfo="skip", showlegend=False,
                marker={"symbol": "x", "size": 10, "color": color}, opacity=0.35 if faded else 1.0,
            ))
        figure.add_annotation(
            x=last["Lap"], y=last["Position"], text=driver, showarrow=False, xanchor="left", xshift=8,
            font={"size": 11 if bold else 10, "color": color}, opacity=0.45 if faded else 1.0,
        )

    figure.update_layout(
        showlegend=False,
        margin={"l": 10, "r": 50, "t": 20, "b": 10},
        hovermode="closest",
        xaxis={"title": "Kör (0 = rajtrács)", "range": [-0.5, max(current_lap, 5) + 2.5],
               "gridcolor": "#1f2129", "zeroline": False, "dtick": 5},
        # Position 1 at the top, as on timing screens.
        yaxis={"title": "Pozíció", "range": [field_size + 0.7, 0.3], "tickvals": list(range(1, field_size + 1)),
               "gridcolor": "#1f2129", "zeroline": False},
    )
    return figure
