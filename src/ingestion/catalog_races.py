import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"

OUTPUT_PATH = Path("data/metadata/race_catalog.csv")

YEARS = range(2023, 2027)

# OpenF1 free tier miatt óvatos request rate.
REQUEST_INTERVAL_SECONDS = 2.5

# 429 esetén minimum ennyit várunk,
# ha a szerver nem ad Retry-After fejlécet.
RATE_LIMIT_FALLBACK_SECONDS = 65

MAX_RETRIES = 5


class OpenF1Client:
    """Rate limitet és retry-t kezelő OpenF1 kliens."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        request_interval: float = REQUEST_INTERVAL_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_interval = request_interval

        self.session = requests.Session()
        self._last_request_time = 0.0

    def _wait_before_request(self) -> None:
        """Biztosítja a kérések közötti minimális várakozást."""
        elapsed = time.monotonic() - self._last_request_time

        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)

    def get_json(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> list[dict]:
        """Lekér egy OpenF1 endpointot retry + rate limit kezeléssel."""

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(1, MAX_RETRIES + 1):
            self._wait_before_request()

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=30,
                )

                self._last_request_time = time.monotonic()

            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise

                wait_seconds = min(5 * attempt, 30)

                print(f"  Network error: {exc}")
                print(
                    f"  Retry {attempt}/{MAX_RETRIES} "
                    f"in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)
                continue

            # -----------------------------------------------------
            # SUCCESS
            # -----------------------------------------------------
            if response.status_code == 200:
                return response.json()

            # -----------------------------------------------------
            # 404
            # -----------------------------------------------------
            if response.status_code == 404:
                response.raise_for_status()

            # -----------------------------------------------------
            # 429 - Rate limit
            # -----------------------------------------------------
            if response.status_code == 429:

                retry_after_header = response.headers.get(
                    "Retry-After"
                )

                if retry_after_header:
                    try:
                        wait_seconds = float(retry_after_header)
                    except ValueError:
                        wait_seconds = RATE_LIMIT_FALLBACK_SECONDS
                else:
                    wait_seconds = (
                        RATE_LIMIT_FALLBACK_SECONDS
                        + (attempt - 1) * 10
                    )

                print(
                    f"  Rate limit reached (429). "
                    f"Waiting {wait_seconds:.0f}s..."
                )

                time.sleep(wait_seconds)
                continue

            # -----------------------------------------------------
            # 5xx - Server error
            # -----------------------------------------------------
            if 500 <= response.status_code < 600:

                if attempt == MAX_RETRIES:
                    response.raise_for_status()

                wait_seconds = min(5 * attempt, 30)

                print(
                    f"  Server error {response.status_code}. "
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)
                continue

            # -----------------------------------------------------
            # Egyéb HTTP hiba
            # -----------------------------------------------------
            response.raise_for_status()

        raise RuntimeError(
            f"Failed to fetch {url} after {MAX_RETRIES} attempts."
        )


def get_race_sessions(client: OpenF1Client) -> list[dict]:
    """Lekéri a Race sessionöket 2023 és 2026 között."""

    all_sessions = []

    for year in YEARS:
        print(f"Loading race sessions for {year}...")

        sessions = client.get_json(
            "sessions",
            params={
                "year": year,
                "session_name": "Race",
            },
        )

        all_sessions.extend(sessions)

    return all_sessions


def get_meetings(client: OpenF1Client) -> dict[int, dict]:
    """
    Lekéri a meeting metaadatokat évenként.

    A meeting_key alapján dictionary-t építünk.
    """

    meetings_by_key = {}

    for year in YEARS:
        print(f"Loading meeting metadata for {year}...")

        meetings = client.get_json(
            "meetings",
            params={
                "year": year,
            },
        )

        for meeting in meetings:
            meeting_key = meeting.get("meeting_key")

            if meeting_key is not None:
                meetings_by_key[int(meeting_key)] = meeting

    return meetings_by_key


def analyze_weather(
    client: OpenF1Client,
    meeting_key: int,
    session_key: int,
) -> tuple[bool | None, bool]:
    """
    Megnézi, hogy a KONKRÉT Race session alatt volt-e rainfall.

    A weather endpoint meeting_key alapján elérhető,
    de session_key alapján is szűrhető.

    Visszatérés:
        (has_rain, weather_available)

    has_rain:
        True  -> volt rainfall a Race session alatt
        False -> volt weather adat, de nem volt rainfall
        None  -> nincs elérhető weather adat
    """

    try:
        weather = client.get_json(
            "weather",
            params={
                "meeting_key": meeting_key,
                "session_key": session_key,
            },
        )

    except requests.HTTPError as exc:

        if (
            exc.response is not None
            and exc.response.status_code == 404
        ):
            print(
                f"  Race weather unavailable for "
                f"session {session_key} (404)."
            )

            return None, False

        raise

    if not weather:
        print(
            f"  Race weather unavailable for "
            f"session {session_key} (empty response)."
        )

        return None, False

    has_rain = any(
        record.get("rainfall") == 1
        for record in weather
    )

    return has_rain, True


def analyze_stints(
    client: OpenF1Client,
    session_key: int,
) -> tuple[bool | None, bool | None, bool]:
    """
    Megvizsgálja a Race session stintjeit.

    Visszatérés:
        (has_intermediate, has_wet, stints_available)
    """

    try:
        stints = client.get_json(
            "stints",
            params={
                "session_key": session_key,
            },
        )

    except requests.HTTPError as exc:

        if (
            exc.response is not None
            and exc.response.status_code == 404
        ):
            print(
                f"  Stint data unavailable for "
                f"session {session_key} (404)."
            )

            return None, None, False

        raise

    if not stints:
        print(
            f"  Stint data unavailable for "
            f"session {session_key} (empty response)."
        )

        return None, None, False

    compounds = {
        str(stint.get("compound") or "").upper()
        for stint in stints
    }

    has_intermediate = "INTERMEDIATE" in compounds
    has_wet = "WET" in compounds

    return has_intermediate, has_wet, True

def analyze_race_control(
    client: OpenF1Client,
    session_key: int,
) -> tuple[bool | None, bool | None, bool]:
    """
    Megkeresi a VSC és Safety Car eseményeket.

    Visszatérés:
        (has_vsc, has_safety_car, race_control_available)
    """

    try:
        events = client.get_json(
            "race_control",
            params={
                "session_key": session_key,
            },
        )

    except requests.HTTPError as exc:

        if (
            exc.response is not None
            and exc.response.status_code == 404
        ):
            print(
                f"  Race control data unavailable for "
                f"session {session_key} (404)."
            )

            return None, None, False

        raise

    if not events:
        print(
            f"  Race control data unavailable for "
            f"session {session_key} (empty response)."
        )

        return None, None, False

    has_vsc = False
    has_safety_car = False

    for event in events:

        message = str(
            event.get("message") or ""
        ).upper()

        category = str(
            event.get("category") or ""
        ).upper()

        # ---------------------------------------------------------
        # Virtual Safety Car
        # ---------------------------------------------------------
        if (
            "VIRTUAL SAFETY CAR" in message
            or category == "VSC"
            or "VSC" in message
        ):
            has_vsc = True

        # ---------------------------------------------------------
        # Safety Car
        # ---------------------------------------------------------
        if (
            category == "SAFETYCAR"
            or "SAFETY CAR" in message
        ):
            has_safety_car = True

    return has_vsc, has_safety_car, True


def load_checkpoint() -> pd.DataFrame:
    """Betölti az eddigi eredményeket, ha létezik checkpoint."""

    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(OUTPUT_PATH)

        print(
            f"Loaded checkpoint: "
            f"{len(df)} completed races."
        )

        return df

    except Exception as exc:
        print(
            f"Warning: could not read existing checkpoint: "
            f"{exc}"
        )

        return pd.DataFrame()


def save_checkpoint(df: pd.DataFrame) -> None:
    """Elmenti az eddigi eredményeket."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = df.sort_values(
        ["year", "race_date"]
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def add_meeting_metadata(
    df: pd.DataFrame,
    meetings_by_key: dict[int, dict],
) -> pd.DataFrame:
    """
    A meeting_key alapján kiegészíti a katalógust
    a helyes meeting metaadatokkal.
    """

    meeting_names = []
    countries = []
    circuits = []

    for meeting_key in df["meeting_key"]:

        meeting = meetings_by_key.get(
            int(meeting_key),
            {},
        )

        meeting_names.append(
            meeting.get("meeting_name")
        )

        countries.append(
            meeting.get("country_name")
        )

        circuits.append(
            meeting.get("circuit_short_name")
        )

    df["meeting_name"] = meeting_names
    df["country_name"] = countries
    df["circuit_short_name"] = circuits

    return df


def build_catalog(
    client: OpenF1Client,
) -> pd.DataFrame:
    """Felépíti a teljes versenykatalógust."""

    sessions = get_race_sessions(client)

    print(
        f"\nFound {len(sessions)} race sessions."
    )

    # ---------------------------------------------------------
    # MEETING METAADATOK
    # ---------------------------------------------------------

    meetings_by_key = get_meetings(client)

    print(
        f"Loaded metadata for "
        f"{len(meetings_by_key)} meetings."
    )

    # ---------------------------------------------------------
    # CHECKPOINT
    # ---------------------------------------------------------

    catalog = load_checkpoint()

    completed_sessions: set[int] = set()

    if not catalog.empty:
        completed_sessions = set(
            catalog["session_key"]
            .dropna()
            .astype(int)
        )

    # ---------------------------------------------------------
    # VERSENYEK FELDOLGOZÁSA
    # ---------------------------------------------------------

    for index, session in enumerate(
        sessions,
        start=1,
    ):

        meeting_key = int(
            session["meeting_key"]
        )

        session_key = int(
            session["session_key"]
        )

        # -----------------------------------------------------
        # Már feldolgozott session
        # -----------------------------------------------------

        if session_key in completed_sessions:
            print(
                f"[{index}/{len(sessions)}] "
                f"{session.get('year')} - "
                f"{session.get('country_name')} - "
                f"{session.get('circuit_short_name')} "
                f"→ SKIP (already processed)"
            )

            continue

        # -----------------------------------------------------
        # MEETING METAADAT
        # -----------------------------------------------------

        meeting = meetings_by_key.get(
            meeting_key,
            {},
        )

        meeting_name = meeting.get(
            "meeting_name"
        )

        country_name = meeting.get(
            "country_name"
        )

        circuit_short_name = meeting.get(
            "circuit_short_name"
        )

        print(
            f"[{index}/{len(sessions)}] "
            f"{session.get('year')} - "
            f"{meeting_name} - "
            f"{circuit_short_name}"
        )

        # -----------------------------------------------------
        # WEATHER
        #
        # FONTOS:
        # Most már a konkrét RACE SESSION-re szűrünk.
        # -----------------------------------------------------

        has_rain, weather_available = analyze_weather(
            client,
            meeting_key,
            session_key,
        )

        # -----------------------------------------------------
        # RACE CONTROL
        # -----------------------------------------------------

        (
            has_vsc,
            has_safety_car,
            race_control_available,
        ) = analyze_race_control(
            client,
            session_key,
        )

        (
            has_intermediate,
            has_wet,
            stints_available,
        ) = analyze_stints(
            client,
            session_key,
        )

        row = {
            "year": session.get("year"),
            "meeting_name": meeting_name,
            "country_name": country_name,
            "circuit_short_name": circuit_short_name,
            "meeting_key": meeting_key,
            "session_key": session_key,
            "race_date": session.get("date_start"),
            "has_rain": has_rain,
            "weather_available": weather_available,
            "has_intermediate": has_intermediate,
            "has_wet": has_wet,
            "stints_available": stints_available,
            "has_vsc": has_vsc,
            "has_safety_car": has_safety_car,
            "race_control_available": race_control_available,
        }

        new_row_df = pd.DataFrame([row])

        # -----------------------------------------------------
        # ÚJ SOR HOZZÁADÁSA
        # -----------------------------------------------------

        if catalog.empty:
            catalog = new_row_df
        else:
            catalog = pd.concat(
                [
                    catalog,
                    new_row_df,
                ],
                ignore_index=True,
            )

        completed_sessions.add(
            session_key
        )

        # -----------------------------------------------------
        # CHECKPOINT
        # -----------------------------------------------------

        save_checkpoint(catalog)

        print(
            f"  Checkpoint saved "
            f"({len(catalog)} races)."
        )

    # ---------------------------------------------------------
    # MEETING METAADATOK FRISSÍTÉSE
    # ---------------------------------------------------------

    catalog = add_meeting_metadata(
        catalog,
        meetings_by_key,
    )

    return catalog


def print_summary(df: pd.DataFrame) -> None:
    """Kiírja a katalógus összefoglalóját."""

    print("\n=== DONE ===")

    print(
        f"Races: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print("\n=== SUMMARY ===")

    summary_columns = [
        "year",
        "meeting_name",
        "country_name",
        "circuit_short_name",
        "has_rain",
        "has_intermediate",
        "has_wet",
        "weather_available",
        "has_vsc",
        "has_safety_car",
        "race_control_available",
    ]

    print(
        df[summary_columns].to_string(
            index=False
        )
    )

    print(
        f"Races with Intermediate: "
        f"{int((df['has_intermediate'] == True).sum())}"
    )

    print(
        f"Races with Wet: "
        f"{int((df['has_wet'] == True).sum())}"
    )

    print(
        f"Stints available: "
        f"{int(df['stints_available'].sum())} / "
        f"{len(df)}"
    )

    print("\n=== DATA AVAILABILITY ===")

    print(
        f"Weather available: "
        f"{int(df['weather_available'].sum())} / "
        f"{len(df)}"
    )

    print(
        f"Race control available: "
        f"{int(df['race_control_available'].sum())} / "
        f"{len(df)}"
    )

    print(
        f"Races with rain: "
        f"{int((df['has_rain'] == True).sum())}"
    )

    print(
        f"Races with VSC: "
        f"{int((df['has_vsc'] == True).sum())}"
    )

    print(
        f"Races with Safety Car: "
        f"{int((df['has_safety_car'] == True).sum())}"
    )

    print("\n=== UNKNOWN DATA ===")

    unknown_weather = int(
        df["weather_available"].eq(False).sum()
    )

    unknown_race_control = int(
        df["race_control_available"].eq(False).sum()
    )

    print(
        f"Weather unavailable: "
        f"{unknown_weather}"
    )

    print(
        f"Race control unavailable: "
        f"{unknown_race_control}"
    )


def main() -> None:
    print("=== F1 RACE CATALOG ===")

    client = OpenF1Client()

    df = build_catalog(client)

    df = df.sort_values(
        ["year", "race_date"]
    ).reset_index(drop=True)

    save_checkpoint(df)

    print_summary(df)


if __name__ == "__main__":
    main()