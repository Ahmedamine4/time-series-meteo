from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster


# ============================================================
# File paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REFERENCE_FILE = PROJECT_ROOT / "../Arome/stations_reference.csv"
FILTERED_FILE = PROJECT_ROOT / "Arome_filtered.csv"
REJECTED_FILE = PROJECT_ROOT / "Arome_rejected_distance.csv"

OUTPUT_FILE = PROJECT_ROOT / "Arome_station_map.html"


# Maximum number of points displayed for each category.
# This prevents the HTML map from becoming too large.
MAX_FILTERED_POINTS = 5000
MAX_REJECTED_POINTS = 2000

RANDOM_STATE = 42


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Charge un CSV et vérifie la présence des colonnes requises."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    df = pd.read_csv(
        path,
        dtype={"station": "string"},
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {path.name} : "
            f"{sorted(missing_columns)}"
        )

    return df


def prepare_coordinates(
    df: pd.DataFrame,
    longitude_column: str,
    latitude_column: str,
) -> pd.DataFrame:
    """Convertit les coordonnées en nombres et retire les valeurs invalides."""
    df = df.copy()

    df[longitude_column] = pd.to_numeric(
        df[longitude_column],
        errors="coerce",
    )

    df[latitude_column] = pd.to_numeric(
        df[latitude_column],
        errors="coerce",
    )

    return df.dropna(
        subset=[
            longitude_column,
            latitude_column,
        ]
    ).copy()


def sample_dataframe(
    df: pd.DataFrame,
    maximum_rows: int,
) -> pd.DataFrame:
    """Prend un échantillon reproductible si le dataset est trop grand."""
    if len(df) <= maximum_rows:
        return df.copy()

    return df.sample(
        n=maximum_rows,
        random_state=RANDOM_STATE,
    ).copy()


def main() -> None:
    # ========================================================
    # Load datasets
    # ========================================================

    references = load_csv(
        REFERENCE_FILE,
        {
            "station",
            "reference_longitude",
            "reference_latitude",
        },
    )

    filtered = load_csv(
        FILTERED_FILE,
        {
            "station",
            "datetime",
            "longitude",
            "latitude",
            "distance_km",
        },
    )

    rejected = load_csv(
        REJECTED_FILE,
        {
            "datetime",
            "longitude",
            "latitude",
            "distance_km",
        },
    )

    # ========================================================
    # Prepare coordinates
    # ========================================================

    references = prepare_coordinates(
        references,
        "reference_longitude",
        "reference_latitude",
    )

    filtered = prepare_coordinates(
        filtered,
        "longitude",
        "latitude",
    )

    rejected = prepare_coordinates(
        rejected,
        "longitude",
        "latitude",
    )

    filtered["distance_km"] = pd.to_numeric(
        filtered["distance_km"],
        errors="coerce",
    )

    rejected["distance_km"] = pd.to_numeric(
        rejected["distance_km"],
        errors="coerce",
    )

    # Take samples for visualization.
    filtered_sample = sample_dataframe(
        filtered,
        MAX_FILTERED_POINTS,
    )

    rejected_sample = sample_dataframe(
        rejected,
        MAX_REJECTED_POINTS,
    )

    # ========================================================
    # Create map
    # ========================================================

    center_latitude = references[
        "reference_latitude"
    ].mean()

    center_longitude = references[
        "reference_longitude"
    ].mean()

    station_map = folium.Map(
        location=[
            center_latitude,
            center_longitude,
        ],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Separate layers.
    station_layer = folium.FeatureGroup(
        name="Reference stations",
        show=True,
    )

    filtered_layer = folium.FeatureGroup(
        name="Accepted points",
        show=True,
    )

    rejected_layer = folium.FeatureGroup(
        name="Rejected points",
        show=False,
    )

    filtered_cluster = MarkerCluster(
        name="Accepted-point clusters",
    )

    rejected_cluster = MarkerCluster(
        name="Rejected-point clusters",
    )

    # ========================================================
    # Add reference stations
    # ========================================================

    for row in references.itertuples(index=False):
        popup_lines = [
            f"<b>Station:</b> {row.station}",
            (
                "<b>Reference longitude:</b> "
                f"{row.reference_longitude:.5f}"
            ),
            (
                "<b>Reference latitude:</b> "
                f"{row.reference_latitude:.5f}"
            ),
        ]

        if hasattr(row, "n_occurrences"):
            popup_lines.append(
                f"<b>Occurrences:</b> {row.n_occurrences}"
            )

        if hasattr(row, "purity"):
            popup_lines.append(
                f"<b>Purity:</b> {row.purity}"
            )

        folium.Marker(
            location=[
                row.reference_latitude,
                row.reference_longitude,
            ],
            popup=folium.Popup(
                "<br>".join(popup_lines),
                max_width=350,
            ),
            tooltip=f"Reference station {row.station}",
            icon=folium.Icon(
                color="blue",
                icon="cloud",
                prefix="fa",
            ),
        ).add_to(station_layer)

        # Draw the 10 km acceptance area.
        folium.Circle(
            location=[
                row.reference_latitude,
                row.reference_longitude,
            ],
            radius=10_000,
            tooltip=f"10 km area — station {row.station}",
            color="blue",
            weight=1,
            fill=True,
            fill_opacity=0.05,
        ).add_to(station_layer)

    # ========================================================
    # Add accepted points
    # ========================================================

    for row in filtered_sample.itertuples(index=False):
        distance_text = (
            f"{row.distance_km:.3f} km"
            if pd.notna(row.distance_km)
            else "Unknown"
        )

        assignment_method = getattr(
            row,
            "assignment_method",
            "Not specified",
        )

        popup = (
            f"<b>Station:</b> {row.station}<br>"
            f"<b>Datetime:</b> {row.datetime}<br>"
            f"<b>Distance:</b> {distance_text}<br>"
            f"<b>Assignment:</b> {assignment_method}"
        )

        folium.CircleMarker(
            location=[
                row.latitude,
                row.longitude,
            ],
            radius=3,
            popup=folium.Popup(
                popup,
                max_width=350,
            ),
            tooltip=f"Accepted — station {row.station}",
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.65,
            weight=1,
        ).add_to(filtered_cluster)

    filtered_cluster.add_to(filtered_layer)

    # ========================================================
    # Add rejected points
    # ========================================================

    for row in rejected_sample.itertuples(index=False):
        distance_text = (
            f"{row.distance_km:.3f} km"
            if pd.notna(row.distance_km)
            else "Unknown"
        )

        station = getattr(
            row,
            "station",
            "Unknown",
        )

        assignment_method = getattr(
            row,
            "assignment_method",
            "Not specified",
        )

        popup = (
            f"<b>Station:</b> {station}<br>"
            f"<b>Datetime:</b> {row.datetime}<br>"
            f"<b>Distance:</b> {distance_text}<br>"
            f"<b>Reason/method:</b> {assignment_method}"
        )

        folium.CircleMarker(
            location=[
                row.latitude,
                row.longitude,
            ],
            radius=3,
            popup=folium.Popup(
                popup,
                max_width=350,
            ),
            tooltip="Rejected point",
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.65,
            weight=1,
        ).add_to(rejected_cluster)

    rejected_cluster.add_to(rejected_layer)

    # ========================================================
    # Add layers and legend
    # ========================================================

    station_layer.add_to(station_map)
    filtered_layer.add_to(station_map)
    rejected_layer.add_to(station_map)

    folium.LayerControl(
        collapsed=False,
    ).add_to(station_map)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 230px;
        padding: 12px;
        background-color: white;
        border: 2px solid grey;
        border-radius: 6px;
        z-index: 9999;
        font-size: 14px;
    ">
        <b>Map legend</b><br>
        <span style="color: blue;">●</span>
        Reference station<br>
        <span style="color: green;">●</span>
        Accepted point (≤ 10 km)<br>
        <span style="color: red;">●</span>
        Rejected point (&gt; 10 km or invalid)<br>
        <span style="color: blue;">○</span>
        10 km acceptance area
    </div>
    """

    station_map.get_root().html.add_child(
        folium.Element(legend_html)
    )

    # Fit the map to the reference stations.
    station_map.fit_bounds(
        [
            [
                references["reference_latitude"].min(),
                references["reference_longitude"].min(),
            ],
            [
                references["reference_latitude"].max(),
                references["reference_longitude"].max(),
            ],
        ]
    )

    station_map.save(OUTPUT_FILE)

    # ========================================================
    # Report
    # ========================================================

    print("=== Visualization report ===")
    print(f"Reference stations       : {len(references):,}")
    print(f"Accepted rows available  : {len(filtered):,}")
    print(f"Accepted points displayed: {len(filtered_sample):,}")
    print(f"Rejected rows available  : {len(rejected):,}")
    print(f"Rejected points displayed: {len(rejected_sample):,}")
    print(f"Map created              : {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()