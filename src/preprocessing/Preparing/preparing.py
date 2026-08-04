#### Ce code permet d'extraire les données de rafales à partir de message METAR  ##########

import polars as pl
import re
from tqdm import tqdm
from datetime import date
file=input("donne le nom de ficher csv : ")
# Lecteure des messages METAR et la métadonnée
# Lire le fichier brut avec Python natif
with open(f"{file}.csv", encoding="utf-8") as f:
    raw_lines = f.readlines()

# Regrouper les lignes en enregistrements logiques
records = []
current = ""

for line in raw_lines:
    if re.match(r"^\d+;", line):  # Nouvelle entrée = ligne qui commence par un SID
        if current:
            records.append(current.strip())
        current = line.strip()
    else:
        current += " " + line.strip()

# Ajouter le dernier enregistrement
if current:
    records.append(current.strip())

# Convertir en DataFrame Polars
df_clean = pl.DataFrame({"raw": records})

# Séparer les colonnes
df = df_clean.select(
    pl.col("raw").str.split_exact(";", 8).alias("splitted")
).unnest("splitted")

# Renommer proprement
df = df.rename({
    "field_0": "sid",
    "field_1": "nom",
    "field_2": "taille",
    "field_3": "updated",           
    "field_4": "dated",
    "field_5": "message",
    "field_6": "source",
    "field_7": "oacis"
})

# Nettoyage optionnel du message (remplacer les multiples espaces/nouveaux lignes)
df = df.with_columns(
    pl.col("message").str.replace_all(r"\s+", " ").str.strip_chars()
)
# ###  Supprimer(si la priemier ligne repeter) et inverser l'odre des lignes
df=df.drop("field_8")
df = df.filter(pl.col("sid") != "sid")
 

# Extraire metar_data (1er bloc de texte) + message (la suite réelle du METAR)
df = df.with_columns([
    pl.col("message").str.extract(r"^(.*?)\s+METAR", 1).alias("metar_data"),
    pl.col("message").str.extract(r"(METAR.*?)(?:\s+transmet|$)", 1).alias("message_clean")
])

# Extraction du message METAR
df_met=df[["dated","message_clean","metar_data"]]

### Eliminer des observation manquantes (""METAR,,,")
df_met = df_met.filter(pl.col("message_clean").is_not_null())

# Extract Indicatif et l'heure de l'observation
df_met = df_met.with_columns([
    # Extraire l’indicatif (2e mot du message)
    pl.col("metar_data")
      .str.extract(r"^\S+\s+(\S+)", 1)
      .alias("indicatif"),

    # Extraire l'heure (les deux chiffres au milieu de l'heure brute)
    pl.col("metar_data")
      .str.extract(r"\b\d{2}(\d{2})\d{2}", 1)
      .alias("heure")
])

# Create time colonne
df_met = df_met.with_columns([
    # Créer une colonne string combinée "2023-02-14 06:00:00"
    (pl.col("dated").cast(pl.Utf8) + " " + pl.col("heure") + ":00:00")
    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    .alias("time")
])

df_met[['time','message_clean']]


# Extract Wind And Gust 
# Constante de conversion
MPS_TO_KT = 1.94384

# Étape 1 — Nettoyage des messages METAR
df_met = df_met.with_columns(
    pl.col("message_clean")
    .str.replace_all(r"VRC", "VRB")                                      # Corriger VRC → VRB
    .str.replace_all(r"(\d{5})T[Y]?", r"\1KT")                           # 04021T ou 04021TY → 04021KT
    .str.replace_all(r"(\d{3})V(\d{3})(\d{2,3})KT", r"\2\3KT")           # 210V31003KT → 31003KT
    .str.replace_all(r"\b(\d{2})KT", r"0\1KT") 
    .str.replace_all(r"\b(\d{3})K(\d{2,3})T\b", r"\1\2KT")         # 320K09T → 32009KT
    .str.replace_all(r"\bVRB(\d)KT\b", r"VRB0\1KT")                # VRB2KT → VRB02KT
    .str.replace_all(r"\b(\d{5})K\b", r"\1KT")                     # 36006K → 36006KT
    .str.replace_all(r"\s+(KT|MPS)\b", r"\1")                      # 00001 KT → 00001KT, VRB02 KT → VRB02KT
    .str.replace_all(r"KY\b", "KT")                                # 02018KY → 02018KT
    .str.replace_all(r"\b(VRB\d{2})T\b", r"\1KT")                  # VRB02T → VRB02KT                          # Ajouter zéro si vent direction à 2 chiffres
    .alias("message_clean")
)

msg = pl.col("message_clean")  # pour simplifier les appels

# Étape 2 — Extraction des champs du vent
df_met = df_met.with_columns([

    # wind_dir : 0 pour VRB, sinon extraire direction (2 à 3 chiffres)
    pl.when(msg.str.contains(r"\bVRB\d{2,3}(G\d{2,3})?(KT|MPS)"))
      .then(pl.lit(0))
      .when(msg.str.contains(r"\b(\d{2,3})\d{2,3}(G\d{2,3})?(KT|MPS)"))
      .then(
          msg.str.extract(r"\b(\d{2,3})\d{2,3}(G\d{2,3})?(KT|MPS)", 1).cast(pl.Int32)
      )
      .otherwise(None)
      .alias("wind_dir"),

    # wind_speed : gérer VRB et directionnel avec KT/MPS
    pl.when(msg.str.contains(r"\bVRB(\d{2,3})(G\d{2,3})?MPS"))
      .then(
          (msg.str.extract(r"\bVRB(\d{2,3})(G\d{2,3})?MPS", 1)
           .cast(pl.Float64) * MPS_TO_KT).round(1)
      )
      .when(msg.str.contains(r"\bVRB(\d{2,3})(G\d{2,3})?KT"))
      .then(
          msg.str.extract(r"\bVRB(\d{2,3})(G\d{2,3})?KT", 1).cast(pl.Int32)
      )
      .when(msg.str.contains(r"\b\d{2,3}(\d{2,3})(G\d{2,3})?MPS"))
      .then(
          (msg.str.extract(r"\b\d{2,3}(\d{2,3})(G\d{2,3})?MPS", 1)
           .cast(pl.Float64) * MPS_TO_KT).round(1)
      )
      .when(msg.str.contains(r"\b\d{2,3}(\d{2,3})(G\d{2,3})?KT"))
      .then(
          msg.str.extract(r"\b\d{2,3}(\d{2,3})(G\d{2,3})?KT", 1).cast(pl.Int32)
      )
      .otherwise(None)
      .alias("wind_speed"),

    # gust_speed : gérer rafales en KT ou MPS
    pl.when(msg.str.contains(r"G\d{2,3}KT"))
      .then(msg.str.extract(r"G(\d{2,3})KT", 1).cast(pl.Int32))
      .when(msg.str.contains(r"G\d{2,3}MPS"))
      .then(
          (msg.str.extract(r"G(\d{2,3})MPS", 1)
           .cast(pl.Float64) * MPS_TO_KT).round(1)
      )
      .otherwise(None)
      .alias("gust_speed")
])

# Étape 3 — Colonnes finales
df_met = df_met.with_columns([
    pl.col("wind_speed")
      .cast(pl.Float64)
      .alias("wind_mean"),

    # Rafale si elle existe, sinon vent moyen
    pl.coalesce([
        pl.col("gust_speed").cast(pl.Float64),
        pl.col("wind_speed").cast(pl.Float64)
    ]).alias("wind_final"),

    # 1 = vraie rafale, 0 = aucune rafale
    pl.col("gust_speed")
      .is_not_null()
      .cast(pl.Int8)
      .alias("has_gust")
])

# Colonnes finales
df_fin = df_met.select([
    "time",
    "indicatif",
    "wind_mean",
    "wind_final",
    "has_gust"
])

# Trier par date
df_fin = df_fin.sort("time")

# Garder les lignes où le vent moyen existe
df_fin = df_fin.drop_nulls(["time", "indicatif", "wind_mean"])

# Supprimer les valeurs aberrantes
df_fin = df_fin.filter(
    (pl.col("wind_mean") < 100) &
    (
        pl.col("wind_final").is_null() |
        (pl.col("wind_final") < 100)
    )
)

df_fin.write_parquet(f"{file}_extracted.parquet")
df_fin.write_csv(f"{file}_extracted.csv")