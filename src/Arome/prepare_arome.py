from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("Datasetfinal")
OUTPUT_FILE = Path("Datasetfinal.csv")


# Noms des colonnes
columns = [
    "station",
    "datetime",
    "longitude",
    "latitude",
    "u10",
    "v10",
    "t2m",
    "rh2m",
    "u850",
    "v850",
    "u950",
    "v950",
    "psurf",
    "u_gust60",
    "v_gust60",
    "tke20m",
    "edr20m",
    "pblh",
]


rows = []
invalid_rows = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line_number, line in enumerate(f, start=1):

        values = line.strip().split()

        if len(values) == len(columns):
            rows.append(values)
        else:
            invalid_rows.append((line_number, len(values)))

# Création du DataFrame
df = pd.DataFrame(rows, columns=columns)

print(f"Valid rows   : {len(df):,}")
print(f"Invalid rows : {len(invalid_rows):,}")

# Conversion de la date
df["datetime"] = pd.to_datetime(
    df["datetime"],
    format="%Y%m%d%H",
    errors="coerce",
)

# Conversion des colonnes numériques
numeric_columns = [
    "longitude",
    "latitude",
    "u10",
    "v10",
    "t2m",
    "rh2m",
    "u850",
    "v850",
    "u950",
    "v950",
    "psurf",
    "u_gust60",
    "v_gust60",
    "tke20m",
    "edr20m",
    "pblh",
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

# Calcul de la vitesse de rafale
df["raf_arome"] = np.hypot(
    df["u_gust60"],
    df["v_gust60"],
)

# Sauvegarde
df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print(f"\nDataset saved to: {OUTPUT_FILE}")
print(f"Final shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())