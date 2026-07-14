from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "Datasetfinal.csv"
OUTPUT_FILE = PROJECT_ROOT / "stations_reference.csv"


REQUIRED_COLUMNS = {
    "station",
    "longitude",
    "latitude",
}


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE.resolve()}"
        )

    # Charger le dataset.
    # Les stations sont lues comme des chaînes de caractères
    # afin de conserver correctement les identifiants.
    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "station": "string",
        },
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing_columns)}"
        )

    # Conserver uniquement les lignes possédant
    # un identifiant de station.
    station_rows = df[
        df["station"].notna()
        & df["station"].str.strip().ne("")
    ].copy()

    # Convertir les coordonnées en valeurs numériques.
    station_rows["longitude"] = pd.to_numeric(
        station_rows["longitude"],
        errors="coerce",
    )

    station_rows["latitude"] = pd.to_numeric(
        station_rows["latitude"],
        errors="coerce",
    )

    # Supprimer les lignes ayant des coordonnées invalides.
    valid_rows = station_rows.dropna(
        subset=[
            "station",
            "longitude",
            "latitude",
        ]
    ).copy()

    # Compter le nombre d'apparitions de chaque couple
    # longitude/latitude pour chaque station.
    coordinate_counts = (
        valid_rows.groupby(
            [
                "station",
                "longitude",
                "latitude",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "n_occurrences",
            }
        )
    )

    # Calculer le nombre total d'observations par station.
    station_totals = (
        coordinate_counts.groupby(
            "station",
            as_index=False,
        )["n_occurrences"]
        .sum()
        .rename(
            columns={
                "n_occurrences": "n_total",
            }
        )
    )

    # Trier les coordonnées par fréquence décroissante,
    # puis conserver la coordonnée la plus fréquente
    # pour chaque station.
    references = (
        coordinate_counts.sort_values(
            by=[
                "station",
                "n_occurrences",
                "longitude",
                "latitude",
            ],
            ascending=[
                True,
                False,
                True,
                True,
            ],
        )
        .drop_duplicates(
            subset=["station"],
            keep="first",
        )
        .merge(
            station_totals,
            on="station",
            how="left",
            validate="one_to_one",
        )
    )

    # Calculer la pureté :
    # part des observations correspondant
    # à la coordonnée principale.
    references["purity"] = (
        references["n_occurrences"]
        / references["n_total"]
    ).round(4)

    # Renommer les coordonnées de référence.
    references = references.rename(
        columns={
            "longitude": "reference_longitude",
            "latitude": "reference_latitude",
        }
    )

    final_columns = [
        "station",
        "reference_longitude",
        "reference_latitude",
        "n_occurrences",
        "n_total",
        "purity",
    ]

    references = (
        references[final_columns]
        .sort_values("station")
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    references.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=== Reference coordinates report ===")
    print(f"Rows read                 : {len(df):,}")
    print(f"Rows with station ID      : {len(station_rows):,}")
    print(f"Valid coordinate rows     : {len(valid_rows):,}")
    print(
        f"Rows without station ID   : "
        f"{df['station'].isna().sum():,}"
    )
    print(
        f"Stations found            : "
        f"{references['station'].nunique():,}"
    )
    print(f"Output file               : {OUTPUT_FILE.resolve()}")

    print("\nReference coordinates:")
    print(references.to_string(index=False))


if __name__ == "__main__":
    main()