import polars as pl

code_OACI = {
    60033: "GMML", 60060: "GMMF", 60096: "GMMH", 60101: "GMTT",
    60107: "GMTA", 60115: "GMFO", 60120: "GMMP", 60135: "GMME",
    60136: "GMSL", 60141: "GMFF", 60150: "GMFM", 60155: "GMMC",
    60156: "GMMN", 60160: "GMFI", 60191: "GMMD", 60200: "GMFB",
    60210: "GMFK", 60220: "GMMI", 60230: "GMMX", 60250: "GMAA",
    60252: "GMAD", 60265: "GMMZ", 60280: "GMAG", 60285: "GMAT",
    60340: "GMMW"
}

# 1. Lecture

df_arome = pl.read_csv("Arome_clean_final.csv", try_parse_dates=False)
df_metar = pl.read_csv("Rafale_METAR.csv", try_parse_dates=False)

print("Colonnes brutes AROME :", df_arome.columns)
print("Colonnes METAR :", df_metar.columns)

# 2. Renommage éventuel
renommage = {}
if "" in df_arome.columns:
    renommage[""] = "station"
if "longitudestation" in df_arome.columns:
    renommage["longitudestation"] = "longitude"
if renommage:
    df_arome = df_arome.rename(renommage)

# 3. Vérification des colonnes
colonnes_arome_requises = {"station", "datetime"}
colonnes_metar_requises = {"indicatif", "time", "wind_mean", "wind_final", "has_gust"}

manquantes_arome = colonnes_arome_requises - set(df_arome.columns)
manquantes_metar = colonnes_metar_requises - set(df_metar.columns)

if manquantes_arome:
    raise ValueError(f"Colonnes manquantes dans AROME : {sorted(manquantes_arome)}")
if manquantes_metar:
    raise ValueError(f"Colonnes manquantes dans METAR : {sorted(manquantes_metar)}")

# 4. Station numérique -> OACI

df_arome = df_arome.with_columns(
    pl.col("station").cast(pl.Int64, strict=False).alias("station")
)

df_arome = df_arome.with_columns(
    pl.col("station")
    .replace_strict(code_OACI, default=None, return_dtype=pl.Utf8)
    .alias("indicatif")
)

# 5. Nettoyage des indicatifs

df_arome = df_arome.with_columns(
    pl.col("indicatif").str.strip_chars().str.to_uppercase()
)

df_metar = df_metar.with_columns(
    pl.col("indicatif").cast(pl.Utf8).str.strip_chars().str.to_uppercase()
)

# 6. Conversion des dates

def convertir_datetime(colonne: str) -> pl.Expr:
    return (
        pl.col(colonne)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all("T", " ")
        .str.replace_all("Z", "")
        .str.slice(0, 19)
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    )


df_arome = df_arome.with_columns(convertir_datetime("datetime").alias("datetime"))
df_metar = df_metar.with_columns(convertir_datetime("time").alias("time"))

print("Dates AROME non reconnues :", df_arome["datetime"].null_count())
print("Dates METAR non reconnues :", df_metar["time"].null_count())

# 7. Codes non convertis
stations_non_converties = (
    df_arome.filter(pl.col("indicatif").is_null())
    .select("station").unique().sort("station")
)
print("Codes AROME sans correspondance OACI :")
print(stations_non_converties)

# 8. Suppression uniquement des clés manquantes

df_arome = df_arome.drop_nulls(["datetime", "indicatif"])
df_metar = df_metar.drop_nulls(["time", "indicatif"])

# 9. Préparation METAR — logique conservée

df_metar = df_metar.with_columns([
    pl.col("wind_mean").cast(pl.Float64, strict=False),
    pl.coalesce([
        pl.col("wind_final").cast(pl.Float64, strict=False),
        pl.col("wind_mean").cast(pl.Float64, strict=False)
    ]).alias("wind_final"),
    pl.col("has_gust").cast(pl.Int8, strict=False)
])

df_metar = df_metar.select([
    "time", "indicatif", "wind_mean", "wind_final", "has_gust"
])

# 10. CORRECTION PRINCIPALE
# Ne plus supprimer une ligne AROME selon datetime + indicatif.
# On retire uniquement les lignes entièrement identiques.
df_arome = df_arome.unique(maintain_order=True)

# METAR reste à une ligne par station et heure.
df_metar = df_metar.unique(
    subset=["time", "indicatif"],
    keep="last",
    maintain_order=True
)

# 11. Contrôle des répétitions AROME, sans suppression

doublons_arome = (
    df_arome.group_by(["station", "datetime"])
    .len()
    .filter(pl.col("len") > 1)
    .sort("len", descending=True)
)

print("Nombre de couples station/datetime répétés dans AROME :", doublons_arome.height)
if doublons_arome.height > 0:
    print(doublons_arome.head(20))

# 12. Sauvegarde de référence pour contrôler 60101
colonnes_arome_originales = list(df_arome.columns)
reference_60101 = (
    df_arome.filter(pl.col("station") == 60101)
    .select(colonnes_arome_originales)
)

print("Nombre de lignes AROME 60101 avant jointure :", reference_60101.height)

# 13. Stations et périodes
stations_arome = set(df_arome["indicatif"].unique().to_list())
stations_metar = set(df_metar["indicatif"].unique().to_list())
stations_communes = stations_arome & stations_metar

print("Nombre de stations AROME :", len(stations_arome))
print("Nombre de stations METAR :", len(stations_metar))
print("Nombre de stations communes :", len(stations_communes))
print("Stations communes :", sorted(stations_communes))

print("Période AROME :")
print(df_arome.select(pl.col("datetime").min().alias("debut"), pl.col("datetime").max().alias("fin")))
print("Période METAR :")
print(df_metar.select(pl.col("time").min().alias("debut"), pl.col("time").max().alias("fin")))

# 14. Jointure sûre
# m:1 = plusieurs lignes AROME possibles, une seule ligne METAR par clé.
df_final = df_arome.join(
    df_metar,
    left_on=["datetime", "indicatif"],
    right_on=["time", "indicatif"],
    how="inner",
    validate="m:1"
)

# 15. Vérification que 60101 n'a pas été modifiée
apres_60101 = (
    df_final.filter(pl.col("station") == 60101)
    .select([c for c in colonnes_arome_originales if c in df_final.columns])
)

cles_comparaison = [c for c in apres_60101.columns if c in reference_60101.columns]

lignes_60101_inconnues = apres_60101.join(
    reference_60101.select(cles_comparaison),
    on=cles_comparaison,
    how="anti"
)

if lignes_60101_inconnues.height > 0:
    raise ValueError("Des valeurs AROME de la station 60101 ont été modifiées pendant la jointure.")

print("Contrôle 60101 : aucune valeur AROME n'a été modifiée.")
print("Nombre de lignes 60101 après jointure :", apres_60101.height)

# 16. Colonnes finales
colonnes_finales = [
    "station", "indicatif", "datetime", "longitude", "latitude",
    "u10", "v10", "t2m", "rh2m", "u850", "v850", "u950", "v950",
    "psurf", "u_gust60", "v_gust60", "tke20m", "edr20m", "pblh",
    "wind_mean", "wind_final", "has_gust"
]

colonnes_disponibles = [c for c in colonnes_finales if c in df_final.columns]
df_final = df_final.select(colonnes_disponibles)

# 17. Tri et export

df_final = df_final.sort(["datetime", "indicatif"])

print("Nombre de lignes AROME :", df_arome.height)
print("Nombre de lignes METAR :", df_metar.height)
print("Nombre de lignes après jointure :", df_final.height)
print("Colonnes finales :", df_final.columns)
print(df_final.head(10))

df_final.write_csv(
    "Dataset_AROME_METAR.csv",
    datetime_format="%Y-%m-%d %H:%M:%S"
)

df_final.write_parquet("Dataset_AROME_METAR.parquet")

print("Fichiers créés :")
print("- Dataset_AROME_METAR.csv")
print("- Dataset_AROME_METAR.parquet")
