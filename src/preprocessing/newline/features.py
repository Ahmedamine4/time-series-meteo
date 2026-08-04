import polars as pl
import numpy as np


# =====================================================
# 1. PARAMÈTRES
# =====================================================

INPUT_FILE = "AROME_METAR_merged_2021_2025.csv"
OUTPUT_PARQUET = "Dataset_features.parquet"
OUTPUT_CSV = "Dataset_features.csv"

TARGET = "has_gust"


# =====================================================
# 2. COLONNES À GARDER
# =====================================================

# Variables AROME utilisées pour entraîner le modèle.
features_arome = [
    "lon",
    "lat",
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
    "distance_km",
    "elevation_m"
]

# Variables créées à partir de datetime.
features_temporelles = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos"
]

# Colonnes utilisées seulement pour identifier les lignes.
colonnes_identification = [
    "datetime",
    "icao"
]


# =====================================================
# 3. LIRE LE DATASET
# =====================================================

df = pl.read_csv(
    INPUT_FILE,
    try_parse_dates=False
)

print("Nombre de lignes initial :", df.height)
print("Nombre de colonnes initial :", len(df.columns))


# =====================================================
# 4. VÉRIFIER LES COLONNES OBLIGATOIRES
# =====================================================

colonnes_obligatoires = [
    "datetime",
    TARGET
] + features_arome

colonnes_manquantes = [
    colonne
    for colonne in colonnes_obligatoires
    if colonne not in df.columns
]

if colonnes_manquantes:
    raise ValueError(
        "Colonnes manquantes : "
        + ", ".join(colonnes_manquantes)
    )


# =====================================================
# 5. CONVERTIR DATETIME
# =====================================================

df = df.with_columns(
    pl.col("datetime")
    .cast(pl.Utf8)
    .str.strip_chars()
    .str.strptime(
        pl.Datetime,
        format="%Y-%m-%d %H:%M:%S",
        strict=False
    )
    .alias("datetime")
)

df = df.drop_nulls([
    "datetime",
    TARGET
])


# =====================================================
# 6. CRÉER LES VARIABLES TEMPORELLES
# =====================================================

df = df.with_columns([
    pl.col("datetime").dt.year().alias("year"),
    pl.col("datetime").dt.hour().alias("hour"),
    pl.col("datetime").dt.ordinal_day().alias("day_of_year")
])

df = df.with_columns([
    (
        2 * np.pi * pl.col("hour") / 24
    ).sin().alias("hour_sin"),

    (
        2 * np.pi * pl.col("hour") / 24
    ).cos().alias("hour_cos"),

    (
        2 * np.pi * pl.col("day_of_year") / 365.25
    ).sin().alias("doy_sin"),

    (
        2 * np.pi * pl.col("day_of_year") / 365.25
    ).cos().alias("doy_cos")
])


# =====================================================
# 7. CONVERTIR LES VARIABLES EN NUMÉRIQUE
# =====================================================

df = df.with_columns([
    pl.col(colonne)
    .cast(pl.Float64, strict=False)
    .alias(colonne)
    for colonne in features_arome
])

df = df.with_columns(
    pl.col(TARGET)
    .cast(pl.Int8, strict=False)
    .alias(TARGET)
)


# =====================================================
# 8. GARDER UNIQUEMENT LES COLONNES UTILES
# =====================================================

colonnes_identification_disponibles = [
    colonne
    for colonne in colonnes_identification
    if colonne in df.columns
]

colonnes_finales = (
    colonnes_identification_disponibles
    + ["year"]
    + features_arome
    + features_temporelles
    + [TARGET]
)

df_final = df.select(
    colonnes_finales
)


# =====================================================
# 9. VÉRIFIER LE SPLIT PAR ANNÉE
# =====================================================

train = df_final.filter(
    pl.col("year").is_between(2021, 2023)
)

validation = df_final.filter(
    pl.col("year") == 2024
)

test = df_final.filter(
    pl.col("year") == 2025
)

print()
print("Train 2021-2023 :", train.shape)
print("Validation 2024 :", validation.shape)
print("Test 2025 :", test.shape)


# =====================================================
# 10. ENREGISTRER LE DATASET FINAL
# =====================================================

df_final.write_parquet(
    OUTPUT_PARQUET
)

df_final.write_csv(
    OUTPUT_CSV
)

print()
print("Préparation terminée.")
print("Nombre de lignes final :", df_final.height)
print("Nombre de colonnes final :", len(df_final.columns))
print("Colonnes finales :")
print(df_final.columns)
print("Fichier créé :", OUTPUT_FILE)
