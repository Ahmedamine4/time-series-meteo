"""
Feature engineering pour la prediction des rafales de vent
à partir du dataset fusionné AROME + METAR.

Entrée attendue :
    Dataset_AROME_METAR.csv

Sorties :
    Dataset_AROME_METAR_features.csv
    Dataset_AROME_METAR_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration
# =============================================================================

INPUT_PATH = Path("Dataset_AROME_METAR.csv")
OUTPUT_CSV = Path("Dataset_AROME_METAR_features.csv")
OUTPUT_PARQUET = Path("Dataset_AROME_METAR_features.parquet")

REQUIRED_COLUMNS = [
    "station",
    "indicatif",
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
    "wind_mean",
    "wind_final",
    "has_gust",
]


# =============================================================================
# 2. Fonctions de feature engineering
# =============================================================================

def add_wind_speed(data: pd.DataFrame, u_col: str, v_col: str, output: str) -> None:
    """Calcule la magnitude du vent à partir de ses composantes u et v."""
    data[output] = np.hypot(data[u_col], data[v_col])


def add_wind_direction(
    data: pd.DataFrame,
    u_col: str,
    v_col: str,
    suffix: str,
) -> None:
    """
    Calcule la direction météorologique du vent, puis son encodage cyclique.

    La direction indique d'où vient le vent et appartient à [0, 360[.
    Les versions sinus/cosinus évitent la discontinuité entre 359° et 0°.
    """
    direction = (
        np.degrees(np.arctan2(-data[u_col], -data[v_col])) + 360.0
    ) % 360.0

    data[f"dir_{suffix}"] = direction
    data[f"dir_{suffix}_sin"] = np.sin(np.radians(direction))
    data[f"dir_{suffix}_cos"] = np.cos(np.radians(direction))


def add_vector_shear(
    data: pd.DataFrame,
    u_lower: str,
    v_lower: str,
    u_upper: str,
    v_upper: str,
    output: str,
) -> None:
    """Calcule la différence vectorielle du vent entre deux niveaux."""
    data[output] = np.hypot(
        data[u_upper] - data[u_lower],
        data[v_upper] - data[v_lower],
    )


def validate_columns(data: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes nécessaires sont présentes."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


# =============================================================================
# 3. Chargement et validation
# =============================================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {INPUT_PATH.resolve()}")

df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
validate_columns(df)

df = df.sort_values(["datetime", "indicatif"]).reset_index(drop=True)

print(f"Dataset chargé : {len(df):,} lignes")
print(f"Nombre de stations : {df['indicatif'].nunique()}")
print(f"Période : {df['datetime'].min()} -> {df['datetime'].max()}")


# =============================================================================
# 4. Magnitude du vent
# =============================================================================

add_wind_speed(df, "u10", "v10", "speed_10")
add_wind_speed(df, "u850", "v850", "speed_850")
add_wind_speed(df, "u950", "v950", "speed_950")
add_wind_speed(df, "u_gust60", "v_gust60", "speed_gust60")


# =============================================================================
# 5. Direction du vent et encodage cyclique
# =============================================================================

add_wind_direction(df, "u10", "v10", "10")
add_wind_direction(df, "u850", "v850", "850")
add_wind_direction(df, "u950", "v950", "950")


# =============================================================================
# 6. Cisaillement vectoriel entre niveaux
# =============================================================================

add_vector_shear(
    df,
    "u10",
    "v10",
    "u950",
    "v950",
    "shear_10_950",
)

add_vector_shear(
    df,
    "u950",
    "v950",
    "u850",
    "v850",
    "shear_950_850",
)

add_vector_shear(
    df,
    "u10",
    "v10",
    "u850",
    "v850",
    "shear_10_850",
)


# =============================================================================
# 7. Facteur de rafale AROME
# =============================================================================

# Un plancher évite une division instable lorsque le vent moyen est proche de 0.
speed_10_safe = df["speed_10"].clip(lower=0.5)
df["gust_ratio_arome"] = df["speed_gust60"] / speed_10_safe

# Indique les observations pour lesquelles le dénominateur a été protégé.
df["speed_10_was_floored"] = (df["speed_10"] < 0.5).astype("int8")


# =============================================================================
# 8. Variables temporelles
# =============================================================================

df["hour"] = df["datetime"].dt.hour.astype("int8")
df["month"] = df["datetime"].dt.month.astype("int8")
df["day_of_year"] = df["datetime"].dt.dayofyear.astype("int16")
df["year"] = df["datetime"].dt.year.astype("int16")

df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
df["doy_sin"] = np.sin(2.0 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"] = np.cos(2.0 * np.pi * df["day_of_year"] / 365.25)

# Saison météorologique dans l'hémisphère Nord.
season_map = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
df["season"] = df["month"].map(season_map).astype("category")


# =============================================================================
# 9. Encodage catégoriel de la station
# =============================================================================

# station_idx est un identifiant catégoriel compact.
# Il ne doit pas être interprété comme une variable numérique ordonnée.
station_categories = sorted(df["indicatif"].dropna().unique())
station_mapping = {station: index for index, station in enumerate(station_categories)}
df["station_idx"] = df["indicatif"].map(station_mapping).astype("Int16")


# =============================================================================
# 10. Contrôles de qualité des nouvelles features
# =============================================================================

ENGINEERED_FEATURES = [
    "speed_10",
    "speed_850",
    "speed_950",
    "speed_gust60",
    "dir_10",
    "dir_850",
    "dir_950",
    "dir_10_sin",
    "dir_10_cos",
    "dir_850_sin",
    "dir_850_cos",
    "dir_950_sin",
    "dir_950_cos",
    "shear_10_950",
    "shear_950_850",
    "shear_10_850",
    "gust_ratio_arome",
    "speed_10_was_floored",
    "hour",
    "month",
    "day_of_year",
    "year",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "season",
    "station_idx",
]

# Remplacer les valeurs infinies éventuelles par NaN avant export.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nNouvelles features créées :")
for feature in ENGINEERED_FEATURES:
    print(f"- {feature}")

print("\nValeurs manquantes dans les nouvelles features :")
print(df[ENGINEERED_FEATURES].isna().sum().sort_values(ascending=False))

print("\nAperçu des nouvelles features :")
print(df[["indicatif", "datetime", *ENGINEERED_FEATURES[:12]]].head())


# =============================================================================
# 11. Export
# =============================================================================

df.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
df.to_parquet(OUTPUT_PARQUET, index=False)

print("\nFichiers créés :")
print(f"- {OUTPUT_CSV}")
print(f"- {OUTPUT_PARQUET}")
