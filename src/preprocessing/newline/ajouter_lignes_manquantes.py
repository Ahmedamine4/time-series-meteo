#!/usr/bin/env python3
"""
Ajoute à un dataset de caractéristiques les lignes absentes du fichier
AROME–METAR fusionné, sans modifier les lignes déjà présentes.

Clé d'identification :
    (station, datetime) dans le dataset cible
    (id, datetime) dans le fichier source

Mapping demandé :
    station     <- id
    indicatif   <- icao
    longitude   <- lon
    latitude    <- lat
    wind_mean   <- wind_speed_ms
    wind_final  <- gust_speed_ms

Exemple :
python ajouter_lignes_manquantes.py \
    --source AROME_METAR_merged_2021_2025.csv \
    --target AROME_METAR_features.csv \
    --output AROME_METAR_features_complete.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


KEYS_TARGET = ["station", "datetime"]
KEYS_SOURCE = ["id", "datetime"]

COLUMN_MAPPING = {
    "id": "station",
    "icao": "indicatif",
    "datetime": "datetime",
    "lon": "longitude",
    "lat": "latitude",
    "u10": "u10",
    "v10": "v10",
    "t2m": "t2m",
    "rh2m": "rh2m",
    "u850": "u850",
    "v850": "v850",
    "u950": "u950",
    "v950": "v950",
    "psurf": "psurf",
    "u_gust60": "u_gust60",
    "v_gust60": "v_gust60",
    "tke20m": "tke20m",
    "edr20m": "edr20m",
    "pblh": "pblh",
    "wind_speed_ms": "wind_mean",
    "gust_speed_ms": "wind_final",
    "has_gust": "has_gust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajouter les lignes AROME–METAR absentes d'un dataset cible."
    )
    parser.add_argument("--source", required=True, help="CSV AROME–METAR fusionné")
    parser.add_argument("--target", required=True, help="CSV cible à compléter")
    parser.add_argument("--output", required=True, help="CSV de sortie")
    parser.add_argument(
        "--wind-final-fallback",
        action="store_true",
        help=(
            "Lorsque gust_speed_ms est vide, mettre wind_speed_ms dans wind_final. "
            "Sans cette option, wind_final reste vide en l'absence de rafale."
        ),
    )
    return parser.parse_args()


def check_required_columns(
    df: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Colonnes absentes dans {dataset_name}: {', '.join(missing)}"
        )


def normalize_keys(df: pd.DataFrame, station_col: str) -> pd.DataFrame:
    result = df.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")

    if result["datetime"].isna().any():
        count = int(result["datetime"].isna().sum())
        raise ValueError(f"{count} dates sont invalides dans la colonne datetime.")

    # Utilise le type numérique nullable pour éviter 60252.0 contre 60252.
    result[station_col] = pd.to_numeric(
        result[station_col], errors="coerce"
    ).astype("Int64")

    if result[station_col].isna().any():
        count = int(result[station_col].isna().sum())
        raise ValueError(f"{count} identifiants de station sont invalides.")

    return result


def add_basic_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule seulement les variables dont la définition est non ambiguë,
    si elles existent dans le schéma du dataset cible.
    """
    out = df.copy()

    vector_pairs = {
        "speed_10": ("u10", "v10"),
        "speed_850": ("u850", "v850"),
        "speed_950": ("u950", "v950"),
        "speed_gust60": ("u_gust60", "v_gust60"),
    }
    for new_col, (u_col, v_col) in vector_pairs.items():
        if u_col in out.columns and v_col in out.columns:
            out[new_col] = np.hypot(out[u_col], out[v_col])

    direction_levels = {
        "10": ("u10", "v10"),
        "850": ("u850", "v850"),
        "950": ("u950", "v950"),
    }
    for level, (u_col, v_col) in direction_levels.items():
        if u_col not in out.columns or v_col not in out.columns:
            continue

        # Direction météorologique : direction d'où vient le vent.
        radians = np.arctan2(-out[u_col], -out[v_col])
        degrees = (np.degrees(radians) + 360.0) % 360.0

        out[f"dir_{level}"] = degrees
        out[f"dir_{level}_sin"] = np.sin(np.radians(degrees))
        out[f"dir_{level}_cos"] = np.cos(np.radians(degrees))

    speed_pairs = {
        "shear_10_950": ("u10", "v10", "u950", "v950"),
        "shear_950_850": ("u950", "v950", "u850", "v850"),
        "shear_10_850": ("u10", "v10", "u850", "v850"),
    }
    for new_col, (u1, v1, u2, v2) in speed_pairs.items():
        if all(col in out.columns for col in (u1, v1, u2, v2)):
            out[new_col] = np.hypot(out[u2] - out[u1], out[v2] - out[v1])

    if "datetime" in out.columns:
        dt = out["datetime"]
        out["hour"] = dt.dt.hour
        out["month"] = dt.dt.month
        out["day_of_year"] = dt.dt.dayofyear
        out["year"] = dt.dt.year

        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

        month = out["month"]
        out["season"] = np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            ["winter", "spring", "summer", "autumn"],
            default=None,
        )

    return out


def main() -> None:
    args = parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    output_path = Path(args.output)

    for path in (source_path, target_path):
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    print("Lecture du dataset cible...")
    target = pd.read_csv(target_path, low_memory=False)
    check_required_columns(target, set(KEYS_TARGET), "le dataset cible")
    target = normalize_keys(target, "station")

    duplicate_target = target.duplicated(KEYS_TARGET, keep=False)
    if duplicate_target.any():
        raise ValueError(
            f"Le dataset cible contient {int(duplicate_target.sum())} lignes "
            "impliquées dans des doublons de clé (station, datetime)."
        )

    source_columns = list(COLUMN_MAPPING.keys())
    print("Lecture du fichier source...")
    source = pd.read_csv(
        source_path,
        usecols=lambda col: col in source_columns,
        low_memory=False,
    )
    check_required_columns(
        source,
        set(KEYS_SOURCE).union(COLUMN_MAPPING.keys()),
        "le fichier source",
    )
    source = normalize_keys(source, "id")

    duplicate_source = source.duplicated(KEYS_SOURCE, keep=False)
    if duplicate_source.any():
        raise ValueError(
            f"Le fichier source contient {int(duplicate_source.sum())} lignes "
            "impliquées dans des doublons de clé (id, datetime)."
        )

    # Création d'un MultiIndex compact pour détecter les clés absentes.
    target_keys = pd.MultiIndex.from_frame(target[KEYS_TARGET])
    source_key_frame = source[KEYS_SOURCE].rename(
        columns={"id": "station"}
    )
    source_keys = pd.MultiIndex.from_frame(source_key_frame[KEYS_TARGET])

    missing_mask = ~source_keys.isin(target_keys)
    missing_source = source.loc[missing_mask].copy()

    print(f"Lignes du fichier source : {len(source):,}")
    print(f"Lignes déjà présentes    : {len(source) - len(missing_source):,}")
    print(f"Lignes à ajouter         : {len(missing_source):,}")

    if missing_source.empty:
        print("Aucune ligne manquante. Copie du dataset cible.")
        result = target.copy()
    else:
        # Renommage vers le schéma cible.
        new_rows = missing_source.rename(columns=COLUMN_MAPPING)

        # Mapping demandé par l'utilisateur.
        new_rows["wind_mean"] = missing_source["wind_speed_ms"].to_numpy()
        new_rows["wind_final"] = missing_source["gust_speed_ms"].to_numpy()

        if args.wind_final_fallback:
            new_rows["wind_final"] = new_rows["wind_final"].fillna(
                new_rows["wind_mean"]
            )

        # Calcule les variables dérivées évidentes si le fichier cible les utilise.
        new_rows = add_basic_engineered_features(new_rows)

        # Les colonnes propres au dataset cible mais impossibles à reconstruire
        # automatiquement sont laissées à NaN.
        impossible_to_infer = {
            "station_idx",
            "arome_error",
            "correction_target",
            "speed_10_was_floored",
        }
        for col in impossible_to_infer.intersection(target.columns):
            if col not in new_rows.columns:
                new_rows[col] = np.nan

        # Garde exactement l'ordre et le schéma du fichier cible.
        for col in target.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        new_rows = new_rows[target.columns]

        result = pd.concat([target, new_rows], ignore_index=True)
        result = result.sort_values(
            ["datetime", "station"], kind="stable"
        ).reset_index(drop=True)

        final_duplicates = result.duplicated(KEYS_TARGET).sum()
        if final_duplicates:
            raise RuntimeError(
                f"La fusion a créé {final_duplicates} doublons inattendus."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Fichier écrit : {output_path}")
    print(f"Nombre final de lignes : {len(result):,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)
