
import polars as pl


# ==========================================================
# 1. Correspondance code numérique -> code OACI
# ==========================================================

code_OACI = {
    60033: "GMML",
    60060: "GMMF",
    60096: "GMMH",
    60101: "GMTT",
    60107: "GMTA",
    60115: "GMFO",
    60120: "GMMP",
    60135: "GMME",
    60136: "GMSL",
    60141: "GMFF",
    60150: "GMFM",
    60155: "GMMC",
    60156: "GMMN",
    60160: "GMFI",
    60191: "GMMD",
    60200: "GMFB",
    60210: "GMFK",
    60220: "GMMI",
    60230: "GMMX",
    60250: "GMAA",
    60252: "GMAD",
    60265: "GMMZ",
    60280: "GMAG",
    60285: "GMAT",
    60340: "GMMW"
}


# ==========================================================
# 2. Lire les fichiers
# ==========================================================

df_arome = pl.read_csv(
    "Arome_clean_final.csv",
    try_parse_dates=False
)

df_metar = pl.read_csv(
    "Rafale_METAR.csv",
    try_parse_dates=False
)

print("Colonnes brutes AROME :", df_arome.columns)
print("Colonnes METAR :", df_metar.columns)


# ==========================================================
# 3. Corriger les noms de colonnes AROME
# ==========================================================

renommage = {}

if "" in df_arome.columns:
    renommage[""] = "station"

if "longitudestation" in df_arome.columns:
    renommage["longitudestation"] = "longitude"

if renommage:
    df_arome = df_arome.rename(renommage)

print("Colonnes corrigées AROME :", df_arome.columns)


# ==========================================================
# 4. Vérifier les colonnes nécessaires
# ==========================================================

colonnes_arome_requises = {
    "station",
    "datetime"
}

colonnes_metar_requises = {
    "indicatif",
    "time",
    "wind_mean",
    "wind_final",
    "has_gust"
}

manquantes_arome = colonnes_arome_requises - set(df_arome.columns)
manquantes_metar = colonnes_metar_requises - set(df_metar.columns)

if manquantes_arome:
    raise ValueError(
        f"Colonnes manquantes dans AROME : {sorted(manquantes_arome)}"
    )

if manquantes_metar:
    raise ValueError(
        f"Colonnes manquantes dans METAR : {sorted(manquantes_metar)}"
    )


# ==========================================================
# 5. Convertir station numérique en code OACI
# ==========================================================

df_arome = df_arome.with_columns(
    pl.col("station")
    .cast(pl.Int64, strict=False)
    .replace_strict(
        code_OACI,
        default=None,
        return_dtype=pl.Utf8
    )
    .alias("indicatif")
)

print("\nExemples de conversion station -> indicatif :")
print(
    df_arome.select([
        "station",
        "indicatif"
    ]).unique().sort("station")
)


# ==========================================================
# 6. Nettoyer les indicatifs
# ==========================================================

df_arome = df_arome.with_columns(
    pl.col("indicatif")
    .str.strip_chars()
    .str.to_uppercase()
)

df_metar = df_metar.with_columns(
    pl.col("indicatif")
    .cast(pl.Utf8)
    .str.strip_chars()
    .str.to_uppercase()
)


# ==========================================================
# 7. Convertir les dates
# ==========================================================

def convertir_datetime(colonne: str) -> pl.Expr:
    return (
        pl.col(colonne)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all("T", " ")
        .str.replace_all("Z", "")
        .str.slice(0, 19)
        .str.strptime(
            pl.Datetime,
            format="%Y-%m-%d %H:%M:%S",
            strict=False
        )
    )


df_arome = df_arome.with_columns(
    convertir_datetime("datetime").alias("datetime")
)

df_metar = df_metar.with_columns(
    convertir_datetime("time").alias("time")
)

print(
    "\nDates AROME non reconnues :",
    df_arome["datetime"].null_count()
)

print(
    "Dates METAR non reconnues :",
    df_metar["time"].null_count()
)


# ==========================================================
# 8. Vérifier les stations sans correspondance
# ==========================================================

stations_non_converties = (
    df_arome
    .filter(pl.col("indicatif").is_null())
    .select("station")
    .unique()
    .sort("station")
)

print("\nCodes AROME sans correspondance OACI :")
print(stations_non_converties)


# ==========================================================
# 9. Supprimer les clés manquantes
# ==========================================================

df_arome = df_arome.drop_nulls([
    "datetime",
    "indicatif"
])

df_metar = df_metar.drop_nulls([
    "time",
    "indicatif"
])


# ==========================================================
# 10. Remplir wind_final avec wind_mean sans rafale
# ==========================================================

df_metar = df_metar.with_columns([
    pl.col("wind_mean")
    .cast(pl.Float64, strict=False),

    pl.coalesce([
        pl.col("wind_final").cast(pl.Float64, strict=False),
        pl.col("wind_mean").cast(pl.Float64, strict=False)
    ]).alias("wind_final"),

    pl.col("has_gust")
    .cast(pl.Int8, strict=False)
])


# ==========================================================
# 11. Garder les colonnes METAR utiles
# ==========================================================

df_metar = df_metar.select([
    "time",
    "indicatif",
    "wind_mean",
    "wind_final",
    "has_gust"
])


# ==========================================================
# 12. Supprimer les doublons
# ==========================================================

df_arome = df_arome.unique(
    subset=["datetime", "indicatif"],
    keep="last"
)

df_metar = df_metar.unique(
    subset=["time", "indicatif"],
    keep="last"
)


# ==========================================================
# 13. Vérifier les stations communes
# ==========================================================

stations_arome = set(
    df_arome["indicatif"].unique().to_list()
)

stations_metar = set(
    df_metar["indicatif"].unique().to_list()
)

stations_communes = stations_arome & stations_metar

print("\nNombre de stations AROME :", len(stations_arome))
print("Nombre de stations METAR :", len(stations_metar))
print("Nombre de stations communes :", len(stations_communes))
print("Stations communes :", sorted(stations_communes))


# ==========================================================
# 14. Vérifier les périodes
# ==========================================================

print("\nPériode AROME :")
print(
    df_arome.select(
        pl.col("datetime").min().alias("debut"),
        pl.col("datetime").max().alias("fin")
    )
)

print("\nPériode METAR :")
print(
    df_metar.select(
        pl.col("time").min().alias("debut"),
        pl.col("time").max().alias("fin")
    )
)


# ==========================================================
# 15. Jointure AROME / METAR
# ==========================================================

df_final = df_arome.join(
    df_metar,
    left_on=[
        "datetime",
        "indicatif"
    ],
    right_on=[
        "time",
        "indicatif"
    ],
    how="inner"
)


# ==========================================================
# 16. Choisir les colonnes finales
# ==========================================================

colonnes_finales = [
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
    "has_gust"
]

colonnes_disponibles = [
    colonne
    for colonne in colonnes_finales
    if colonne in df_final.columns
]

df_final = df_final.select(colonnes_disponibles)


# ==========================================================
# 17. Trier
# ==========================================================

df_final = df_final.sort([
    "datetime",
    "indicatif"
])


# ==========================================================
# 18. Résultats
# ==========================================================

print("\nNombre de lignes AROME :", df_arome.height)
print("Nombre de lignes METAR :", df_metar.height)
print("Nombre de lignes après jointure :", df_final.height)

print("\nColonnes finales :")
print(df_final.columns)

print("\nAperçu :")
print(df_final.head(10))


# ==========================================================
# 19. Export
# ==========================================================

df_final.write_csv(
    "Dataset_AROME_METAR.csv",
    datetime_format="%Y-%m-%d %H:%M:%S"
)

df_final.write_parquet(
    "Dataset_AROME_METAR.parquet"
)

print("\nFichiers créés :")
print("- Dataset_AROME_METAR.csv")
print("- Dataset_AROME_METAR.parquet")
