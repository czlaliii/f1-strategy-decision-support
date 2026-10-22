from src.dashboard.replay_player import load_local_session_data

replay = load_local_session_data(
    race_name="australia_2025",
    session_key=9693,
)

laps = replay["laps"]
frames = replay["frames"]
track = replay["track"]

print("Helyi replay betöltési teszt")
print("=" * 60)

print("Köradatok:", len(laps))
print("Telemetriai képkockák:", len(frames))
print("Pályapontok:", len(track))
print("Versenyzők:", frames["Driver"].nunique())
print("Idővonal:", frames["TimeSeconds"].min(), "-", frames["TimeSeconds"].max())

print("\nKépkockák oszlopai:")
print(frames.columns.tolist())

print("\nElső 5 képkocka:")
print(
    frames[
        ["TimeSeconds", "Driver", "LapNumber", "X", "Y"]
    ].head()
)

print("\nKöradatok oszlopai:")
print(laps.columns.tolist())

print("\nPályaadatok:")
print(track.head())
