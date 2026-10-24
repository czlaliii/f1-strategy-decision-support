"""Small render helpers: driver detail card and the per-lap weather strip."""
import pandas as pd
from dash import html

from src.dashboard.replay_player import format_lap_time
from src.dashboard.theme import COMPOUND_COLORS, COMPOUND_NAMES, COMPOUND_TEXT

DEFAULT_TEAM_COLOR = "#808080"


def _number(value):
    return None if value is None or pd.isna(value) else int(value)


def compound_pill(compound):
    compound = str(compound).upper() if compound is not None and not pd.isna(compound) else "–"
    return html.Span(
        COMPOUND_NAMES.get(compound, compound.title()),
        className="compound-pill",
        style={"backgroundColor": COMPOUND_COLORS.get(compound, "#697586"), "color": COMPOUND_TEXT.get(compound, "#ffffff")},
    )


def _stat(label, value, sub=None, class_name=""):
    return html.Div(
        [html.Div(label, className="driver-stat-label"),
         html.Div(value, className=f"driver-stat-value {class_name}"),
         html.Div(sub, className="driver-stat-sub") if sub else None],
        className="driver-stat",
    )


def driver_card(driver, last_lap, best_lap=None, team_color=None, is_fastest=False):
    """Card for the driver's last completed lap; best_lap is the personal best row (or None)."""
    position = _number(last_lap.get("Position"))
    tyre_age = _number(last_lap.get("TyreLife"))
    stint = _number(last_lap.get("Stint"))
    best = None
    if best_lap is not None:
        best = _stat(
            "Egyéni legjobb",
            format_lap_time(best_lap["LapTime"]),
            f"{int(best_lap['LapNumber'])}. kör" + (" · a verseny leggyorsabb köre" if is_fastest else ""),
            "is-fastest" if is_fastest else "is-personal-best",
        )
    return html.Div(
        className="driver-card",
        style={"--team-color": team_color or DEFAULT_TEAM_COLOR},
        children=[
            html.Div(
                [
                    html.Span(f"P{position}" if position else "P–", className="driver-card-position"),
                    html.Span(driver, className="driver-card-name"),
                    html.Span(f"{int(last_lap['LapNumber'])}. kör után", className="driver-card-lap"),
                ],
                className="driver-card-header",
            ),
            html.Div(
                [
                    _stat("Utolsó kör", format_lap_time(last_lap.get("LapTime"))),
                    best,
                    _stat("Gumi", compound_pill(last_lap.get("Compound")),
                          f"{tyre_age} kör használt" if tyre_age is not None else None),
                    _stat("Stint", str(stint) if stint is not None else "–"),
                ],
                className="driver-card-stats",
            ),
        ],
    )


def driver_card_message(text):
    return html.Div(text, className="driver-card driver-card-empty")


# Compass labels for the direction the wind blows from (meteorological convention).
WIND_DIRECTIONS = ["É", "ÉK", "K", "DK", "D", "DNy", "Ny", "ÉNy"]


def _trend(current, previous, key, digits=1):
    if previous is None or pd.isna(current.get(key)) or pd.isna(previous.get(key)):
        return None
    change = float(current[key]) - float(previous[key])
    if round(change, digits) == 0:
        return html.Span("= előző körhöz", className="weather-trend")
    arrow = "▲" if change > 0 else "▼"
    return html.Span(f"{arrow} {abs(change):.{digits}f} az előző körhöz", className="weather-trend")


def _tile(label, value, sub=None, class_name=""):
    return html.Div(
        [html.Div(label, className="weather-label"), html.Div(value, className="weather-value"),
         html.Div(sub, className="weather-sub") if sub is not None else None],
        className=f"weather-tile {class_name}",
    )


def weather_strip(current, previous=None, lap=None):
    """Weather tiles at the start of `lap`, with the change since the previous lap."""
    title = f"IDŐJÁRÁS · {lap}. KÖR" if lap else "IDŐJÁRÁS"
    if current is None:
        return [html.Div(title, className="weather-title"),
                html.Div("Ehhez a sessionhöz nincs időjárási adat.", className="weather-empty")]

    def value(key, unit, digits=1):
        number = current.get(key)
        return "–" if number is None or pd.isna(number) else f"{float(number):.{digits}f} {unit}"

    raining = bool(current.get("rainfall")) and not pd.isna(current.get("rainfall"))
    direction = current.get("wind_direction")
    if direction is None or pd.isna(direction):
        wind_sub = None
    else:
        compass = WIND_DIRECTIONS[int(((float(direction) % 360) + 22.5) // 45) % 8]
        # The arrow points where the wind blows to.
        wind_sub = html.Span([
            html.Span("➤", className="wind-arrow", style={"transform": f"rotate({float(direction) + 90:.0f}deg)"}),
            f" {compass} felől",
        ])
    return [
        html.Div(title, className="weather-title"),
        html.Div(
            [
                _tile("Levegő", value("air_temperature", "°C"), _trend(current, previous, "air_temperature")),
                _tile("Pályafelület", value("track_temperature", "°C"), _trend(current, previous, "track_temperature")),
                _tile("Páratartalom", value("humidity", "%", 0), _trend(current, previous, "humidity", 0)),
                _tile("Szél", value("wind_speed", "m/s"), wind_sub),
                _tile("Légnyomás", value("pressure", "hPa", 0)),
                _tile("Csapadék", "Eső" if raining else "Száraz", None, "is-rain" if raining else "is-dry"),
            ],
            className="weather-grid",
        ),
    ]
