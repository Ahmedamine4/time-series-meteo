import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "../Arome/Datasetfinal.csv"
REFERENCE_FILE = PROJECT_ROOT / "../Arome/stations_reference.csv"

OUTPUT_FILE = PROJECT_ROOT / "Arome_filtered.csv"
REJECTED_FILE = PROJECT_ROOT / "Arome_rejected_distance.csv"


THRESHOLD_KM = 10.0
EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    """Calcule la distance entre deux positions en kilomètres."""
    lon1, lat1, lon2, lat2 = map(
        math.radians,
        [lon1, lat1, lon2, lat2],
    )

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return (
        2
        * EARTH_RADIUS_KM
        * math.asin(math.sqrt(a))
    )


def nearest_station(
    longitude: float,
    latitude: float,
    references: pd.DataFrame,
) -> tuple[str | None, float]:
    """Trouve la station de référence la plus proche."""
    best_station = None
    best_distance = float("inf")

    for reference in references.itertuples(index=False):
        distance = haversine_km(
            longitude,
            latitude,
            reference.reference_longitude,
            reference.reference_latitude,
        )

        if distance < best_distance:
            best_distance = distance
            best_station = str(reference.station)

    return best_station, best_distance


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE.resolve()}"
        )

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de références introuvable : "
            f"{REFERENCE_FILE.resolve()}"
        )

    # Charger le dataset.
    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "station": "string",
        },
    )

    # Charger les coordonnées de référence.
    references = pd.read_csv(
        REFERENCE_FILE,
        dtype={
            "station": "string",
        },
    )

    required_data_columns = {
        "station",
        "datetime",
        "longitude",
        "latitude",
    }

    required_reference_columns = {
        "station",
        "reference_longitude",
        "reference_latitude",
    }

    missing_data = required_data_columns - set(df.columns)
    missing_references = (
        required_reference_columns - set(references.columns)
    )

    if missing_data:
        raise ValueError(
            f"Colonnes manquantes dans les données : "
            f"{sorted(missing_data)}"
        )

    if missing_references:
        raise ValueError(
            f"Colonnes manquantes dans les références : "
            f"{sorted(missing_references)}"
        )

    # Nettoyer les stations vides.
    df["station"] = df["station"].str.strip()
    df.loc[df["station"].eq(""), "station"] = pd.NA

    # Conversion de la date.
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    # Conversion des coordonnées.
    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    references["reference_longitude"] = pd.to_numeric(
        references["reference_longitude"],
        errors="coerce",
    )

    references["reference_latitude"] = pd.to_numeric(
        references["reference_latitude"],
        errors="coerce",
    )

    references = references.dropna(
        subset=[
            "station",
            "reference_longitude",
            "reference_latitude",
        ]
    ).copy()

    # Vérifier qu’il n’existe qu’une référence par station.
    if references["station"].duplicated().any():
        duplicate_stations = (
            references.loc[
                references["station"].duplicated(keep=False),
                "station",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Plusieurs références trouvées pour les stations : "
            f"{duplicate_stations}"
        )

    # Dictionnaire :
    # station -> longitude/latitude de référence.
    reference_lookup = (
        references.set_index("station")[
            [
                "reference_longitude",
                "reference_latitude",
            ]
        ]
        .to_dict("index")
    )

    final_stations = []
    distances = []
    assignment_methods = []

    for row in df.itertuples(index=False):
        # Coordonnées invalides.
        if (
            pd.isna(row.longitude)
            or pd.isna(row.latitude)
        ):
            final_stations.append(None)
            distances.append(float("nan"))
            assignment_methods.append(
                "invalid_coordinates"
            )
            continue

        original_station = (
            str(row.station)
            if pd.notna(row.station)
            else None
        )

        # Cas 1 :
        # la ligne possède déjà une station.
        if original_station is not None:
            reference = reference_lookup.get(
                original_station
            )

            # Station présente dans les données,
            # mais absente du fichier de référence.
            if reference is None:
                final_stations.append(
                    original_station
                )
                distances.append(float("nan"))
                assignment_methods.append(
                    "missing_reference"
                )
                continue

            distance = haversine_km(
                row.longitude,
                row.latitude,
                reference["reference_longitude"],
                reference["reference_latitude"],
            )

            final_stations.append(
                original_station
            )
            distances.append(
                distance
            )
            assignment_methods.append(
                "original_station"
            )

        # Cas 2 :
        # la ligne de 17 champs ne possède pas de station.
        else:
            nearest_id, distance = nearest_station(
                row.longitude,
                row.latitude,
                references,
            )

            final_stations.append(
                nearest_id
            )
            distances.append(
                distance
            )
            assignment_methods.append(
                "nearest_station"
            )

    df["station"] = final_stations
    df["distance_km"] = distances
    df["assignment_method"] = assignment_methods

    # Garder seulement les lignes situées
    # à 10 km maximum de leur station de référence.
    keep_mask = (
        df["station"].notna()
        & df["datetime"].notna()
        & df["distance_km"].notna()
        & (df["distance_km"] <= THRESHOLD_KM)
    )

    filtered = (
        df.loc[keep_mask]
        .copy()
        .reset_index(drop=True)
    )

    rejected = (
        df.loc[~keep_mask]
        .copy()
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    filtered.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    rejected.to_csv(
        REJECTED_FILE,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print("=== Station assignment and distance report ===")
    print(f"Rows read                    : {len(df):,}")
    print(f"Distance threshold           : {THRESHOLD_KM} km")

    print(
        "Rows with original station   : "
        f"{(df['assignment_method'] == 'original_station').sum():,}"
    )

    print(
        "Rows assigned to nearest     : "
        f"{(df['assignment_method'] == 'nearest_station').sum():,}"
    )

    print(
        "Rows without reference       : "
        f"{(df['assignment_method'] == 'missing_reference').sum():,}"
    )

    print(
        "Rows with invalid coordinates: "
        f"{(df['assignment_method'] == 'invalid_coordinates').sum():,}"
    )

    print(f"Rows kept                    : {len(filtered):,}")
    print(f"Rows rejected                : {len(rejected):,}")
    print(f"Filtered file                : {OUTPUT_FILE.resolve()}")
    print(f"Rejected file                : {REJECTED_FILE.resolve()}")


if __name__ == "__main__":
    main()