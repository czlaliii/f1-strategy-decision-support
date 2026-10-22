
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/processed/multi_race")

TRAIN_PATH = DATA_DIR / "train_context.parquet"
VALIDATION_PATH = DATA_DIR / "validation_context.parquet"

TARGET = "next_lap_time"

BASE_FEATURES = [
    "driver_number", "lap_number", "lap_duration", "compound",
    "tyre_age", "pit_stop", "is_pit_out_lap", "position",
    "gap_to_leader", "laps_down", "interval",
    "previous_lap_time", "rolling_3_lap_avg", "rolling_5_lap_avg",
    "lap_time_delta", "position_change", "gap_change",
    "interval_change", "laps_since_pit", "track_air_temp_difference",
    "air_temperature", "track_temperature", "rainfall", "humidity",
    "wind_speed",
]

MISSING_FEATURES = [
    "position_missing", "interval_missing",
    "interval_change_missing", "gap_to_leader_missing",
    "gap_change_missing", "previous_lap_time_missing",
    "rolling_3_lap_avg_missing", "rolling_5_lap_avg_missing",
    "lap_time_delta_missing", "position_change_missing",
]

PRACTICE_FEATURES = (
    [
        f"{session}_{feature}"
        for session in ("fp1", "fp2")
        for feature in (
            "available", "best_lap", "avg_top5_lap", "lap_count",
            "best_soft", "best_medium", "best_hard",
            "best_intermediate",
        )
    ]
    + [
        f"fp3_{feature}"
        for feature in (
            "available", "best_lap", "avg_top5_lap", "lap_count",
            "best_soft", "best_medium", "best_hard",
            "best_intermediate", "best_wet",
        )
    ]
)

SC_FEATURES = [
    "sc_duration_in_lap",
    "sc_in_lap",
    "sc_lap_fraction",
    "sc_active_at_end",
]

FEATURES = BASE_FEATURES + MISSING_FEATURES + PRACTICE_FEATURES + SC_FEATURES


def load_data(path):
    df = pd.read_parquet(path)

    # A context datasetből hiányzó missing-indicator oszlopok
    # létrehozása az eredeti oszlopok alapján.
    missing_indicator_sources = {
        "position_missing": "position",
        "interval_missing": "interval",
        "interval_change_missing": "interval_change",
        "gap_to_leader_missing": "gap_to_leader",
        "gap_change_missing": "gap_change",
        "previous_lap_time_missing": "previous_lap_time",
        "rolling_3_lap_avg_missing": "rolling_3_lap_avg",
        "rolling_5_lap_avg_missing": "rolling_5_lap_avg",
        "lap_time_delta_missing": "lap_time_delta",
        "position_change_missing": "position_change",
    }

    for indicator, source in missing_indicator_sources.items():
        if indicator not in df.columns and source in df.columns:
            df[indicator] = df[source].isna().astype("int8")

        print(f"\nMissing-indicator ellenőrzés: {path}")
    for col in missing_indicator_sources:
        if col in df.columns:
            print(f"{col}: {df[col].sum()} jelzett hiány")

    missing = [
        c for c in FEATURES + [TARGET]
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Hiányzó oszlopok ({path}): {missing}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    return df


def build_model(train_df):
    # Azokat a feature-öket elhagyjuk, amelyek a tanítóhalmazon
    # teljes egészében hiányoznak.
    features = [
        c for c in FEATURES
        if not train_df[c].isna().all()
    ]

    categorical = ["compound"]
    numeric = [c for c in features if c not in categorical]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric,
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ])

    return model, features


def add_sc_phase(df):
    df = df.copy()

    # Az SC-jellemzők legyenek numerikusak.
    for col in SC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # SC-vel érintett kör.
    affected = (
        (df["sc_in_lap"] > 0)
        | (df["sc_duration_in_lap"] > 0)
    )

    # SC alatt a kör végén is aktív volt.
    active_at_end = df["sc_active_at_end"] > 0

    # Az előző kör végén még aktív volt az SC,
    # a jelenlegi körben viszont már nem érintett.
    df = df.sort_values(
        ["race_id", "driver_number", "lap_number"]
        if "race_id" in df.columns
        else ["driver_number", "lap_number"]
    ).copy()

    group_cols = (
        ["race_id", "driver_number"]
        if "race_id" in df.columns
        else ["driver_number"]
    )

    previous_active = (
        df.groupby(group_cols)["sc_active_at_end"]
        .shift(1)
        .fillna(0)
        .gt(0)
    )

    df["sc_phase"] = "Normal versenykör"
    df.loc[previous_active & ~affected, "sc_phase"] = (
        "SC utáni első megfigyelt kör"
    )
    df.loc[affected, "sc_phase"] = "SC-vel érintett kör"
    df.loc[affected & active_at_end, "sc_phase"] = (
        "SC a kör végén is aktív"
    )

    return df

def inspect_lap_alignment(df, label):
    print(f"\n=== IDŐBELI ILLESZTÉS: {label} ===")

    df = df.copy()

    required = ["race", "driver_number", "lap_number",
                "lap_duration", "next_lap_time"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        print("Hiányzó oszlopok:", missing)
        return

    for col in [
        "lap_number", "lap_duration", "next_lap_time",
        "sc_duration_in_lap", "sc_lap_fraction",
        "sc_active_at_end",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # A következő sor köridője ugyanahhoz a versenyzőhöz
    # és versenyhez tartozzon.
    df = df.sort_values(
        ["race", "driver_number", "lap_number"]
    )

    df["following_lap_duration_check"] = (
        df.groupby(["race", "driver_number"])["lap_duration"]
        .shift(-1)
    )

    df["target_alignment_diff"] = (
        df["next_lap_time"] - df["following_lap_duration_check"]
    )

    print("\nA next_lap_time és a következő sor lap_duration eltérése:")
    valid = df["target_alignment_diff"].dropna()

    if not valid.empty:
        print("Összehasonlítható sorok:", len(valid))
        print("Átlagos abszolút eltérés:",
              round(valid.abs().mean(), 4), "s")
        print("Legnagyobb abszolút eltérés:",
              round(valid.abs().max(), 4), "s")
        print("Legfeljebb 0,01 s eltérésű sorok:",
              int(valid.abs().le(0.01).sum()), "/", len(valid))

    # Silverstone 2025, a problémás körök környezete
    silverstone = df[
        (df["race"] == "silverstone_2025")
        & df["lap_number"].between(13, 22)
        & df["driver_number"].isin([1, 87])
    ].copy()

    columns = [
        "race", "driver_number", "lap_number",
        "lap_duration", "next_lap_time",
        "following_lap_duration_check",
        "target_alignment_diff",
        "sc_duration_in_lap", "sc_lap_fraction",
        "sc_active_at_end",
    ]
    columns = [col for col in columns if col in silverstone.columns]

    print("\nSilverstone 2025 – 1-es és 87-es autó, 13–22. kör:")
    print(silverstone[columns].to_string(index=False))

def main():
    train_df = load_data(TRAIN_PATH)

    
    print("\n=== SC-MEGFIGYELÉSEK A TANÍTÓADATOKBAN ===")

    train_sc = train_df[
        pd.to_numeric(
            train_df["sc_active_at_end"], errors="coerce"
        ).fillna(0).gt(0)
    ].copy()

    print(f"SC-aktív tanítósorok: {len(train_sc)} / {len(train_df)}")

    if not train_sc.empty:
        print(
            train_sc.groupby("race").agg(
                rows=("next_lap_time", "size"),
                mean_lap_time=("next_lap_time", "mean"),
                median_lap_time=("next_lap_time", "median"),
                max_lap_time=("next_lap_time", "max"),
                mean_sc_duration=("sc_duration_in_lap", "mean"),
            ).round(3).to_string()
        )

        print("\nSC-aktív köridő-percentilisek:")
        print(
            train_sc["next_lap_time"]
            .quantile([0.25, 0.50, 0.75, 0.90, 0.95])
            .round(3)
            .to_string()
        )
    val_df = load_data(VALIDATION_PATH)

    inspect_lap_alignment(train_df, "TRAIN")
    inspect_lap_alignment(val_df, "VALIDATION")

    output_dir = DATA_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, features = build_model(train_df)

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_val = val_df[features]
    y_val = val_df[TARGET]

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    val_df = val_df.copy()
    val_df["actual"] = y_val.to_numpy()
    val_df["predicted"] = predictions
    val_df["error"] = val_df["predicted"] - val_df["actual"]
    val_df["absolute_error"] = val_df["error"].abs()

    val_df = add_sc_phase(val_df)

    
    print("\n=== SC-IDŐTARTAM ÉS KÖRIDŐ KAPCSOLATA ===")

    for label, data in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
    ]:
        data = data.copy()

        data["sc_duration_in_lap"] = pd.to_numeric(
            data["sc_duration_in_lap"], errors="coerce"
        )
        data["next_lap_time"] = pd.to_numeric(
            data["next_lap_time"], errors="coerce"
        )

        data = data[
            (data["sc_active_at_end"] > 0)
            & data["sc_duration_in_lap"].notna()
            & data["next_lap_time"].notna()
        ]

        print(f"\n{label}: {len(data)} SC-aktív sor")

        if len(data) > 1:
            print(
                data[
                    ["sc_duration_in_lap", "next_lap_time"]
                ].describe().round(3)
            )

            print(
                "Korreláció:",
                round(
                    data["sc_duration_in_lap"].corr(
                        data["next_lap_time"]
                    ),
                    3,
                ),
            )

        
    print("\n=== SC-IDŐTARTAM SZERINTI ÖSSZEHASONLÍTÁS ===")

    for label, data in [("TRAIN", train_df), ("VALIDATION", val_df)]:
        data = data.copy()

        for col in [
            "sc_duration_in_lap",
            "next_lap_time",
            "sc_active_at_end",
        ]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data[
            (data["sc_active_at_end"] > 0)
            & data["sc_duration_in_lap"].notna()
            & data["next_lap_time"].notna()
        ].copy()

        data["sc_duration_group"] = pd.cut(
            data["sc_duration_in_lap"],
            bins=[0, 60, 100, 130, 150, 180, float("inf")],
            labels=[
                "0–60 s",
                "60–100 s",
                "100–130 s",
                "130–150 s",
                "150–180 s",
                "180+ s",
            ],
            right=False,
        )

        print(f"\n--- {label} ---")

        print(
            data.groupby("sc_duration_group", observed=False)
            .agg(
                rows=("next_lap_time", "size"),
                mean_sc_duration=("sc_duration_in_lap", "mean"),
                mean_actual_lap=("next_lap_time", "mean"),
                median_actual_lap=("next_lap_time", "median"),
                max_actual_lap=("next_lap_time", "max"),
            )
            .round(2)
            .to_string()
        )

    print("\n=== TELJES VALIDÁCIÓ ===")
    print(f"Sorok száma: {len(val_df)}")
    print(f"MAE:  {mean_absolute_error(y_val, predictions):.3f} s")
    print(
        f"RMSE: "
        f"{np.sqrt(mean_squared_error(y_val, predictions)):.3f} s"
    )
    print(f"R²:   {r2_score(y_val, predictions):.3f}")

    print("\n=== HIBA SC-FÁZISONKÉNT ===")
    summary = (
        val_df.groupby("sc_phase", dropna=False)
        .agg(
            rows=("actual", "size"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("error", lambda x: np.sqrt(np.mean(x ** 2))),
            bias=("error", "mean"),
        )
        .sort_values("rows", ascending=False)
        .round(3)
    )

    print(summary.to_string())

    
    # Részletes diagnosztika az SC alatt nagyot tévedett körökre.
    sc_rows = val_df[
        val_df["sc_active_at_end"].gt(0)
    ].copy()

    sc_rows["absolute_error"] = (
        sc_rows["predicted"] - sc_rows["actual"]
    ).abs()

    # Azonosítsuk a versenyt, ha az elérhető az adatokban.
    race_col = next(
        (
            col for col in ("race_id", "race_name", "race", "meeting_key")
            if col in sc_rows.columns
        ),
        None,
    )

    detail_cols = [
        col for col in (
            race_col,
            "driver_number",
            "lap_number",
            "compound",
            "tyre_age",
            "actual",
            "predicted",
            "error",
            "sc_duration_in_lap",
            "sc_in_lap",
            "sc_lap_fraction",
            "sc_active_at_end",
        )
        if col is not None and col in sc_rows.columns
    ]

    print("\n=== SC AKTÍV A KÖR VÉGÉN: LEGNAGYOBB HIBÁK ===")
    print(
        sc_rows.sort_values("absolute_error", ascending=False)[
            detail_cols
        ].head(25).to_string(index=False)
    )

    if race_col is not None:
        print("\n=== SC-HIBA VERSENYENKÉNT ===")
        print(
            sc_rows.groupby(race_col).agg(
                rows=("actual", "size"),
                actual_mean=("actual", "mean"),
                predicted_mean=("predicted", "mean"),
                MAE=("absolute_error", "mean"),
                bias=("error", "mean"),
            ).round(3).to_string()
        )

    sc_rows[detail_cols + ["absolute_error"]].to_csv(
        output_dir / "sc_active_detailed_errors.csv",
        index=False,
    )

    output_dir = DATA_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "sc_phase_error_summary.csv")
    val_df.to_csv(
        output_dir / "sc_phase_validation_predictions.csv",
        index=False,
    )

    print(f"\nEredmények mentve ide: {output_dir}")


if __name__ == "__main__":
    main()
