from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "Arome_filtered.csv"
OUTPUT_FILE = PROJECT_ROOT / "Arome_clean_final.csv"


KEY_COLUMNS = [
    "station",
    "datetime",
]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"station": "string"},
    )

    required_columns = {
        "station",
        "datetime",
        "distance_km",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {sorted(missing)}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df["distance_km"] = pd.to_numeric(
        df["distance_km"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "station",
            "datetime",
            "distance_km",
        ]
    )

    total_rows = len(df)

    clean = (
    df.sort_values(
        by=[
            "datetime",
            "distance_km",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .drop_duplicates(
        subset=KEY_COLUMNS,
        keep="first",
    )
    .sort_values(
        by="datetime"
    )
    .reset_index(drop=True)
)

    removed_rows = total_rows - len(clean)

    # Supprimer les colonnes techniques du DataFrame final.
    clean = clean.drop(
        columns=[
            "distance_km",
            "assignment_method",
        ],
        errors="ignore",
    )

    clean.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    print("=== Final deduplication report ===")
    print(f"Rows read             : {total_rows:,}")
    print(f"Duplicates removed    : {removed_rows:,}")
    print(f"Rows kept             : {len(clean):,}")
    print(
        "Remaining duplicates  : "
        f"{clean.duplicated(KEY_COLUMNS).sum():,}"
    )
    print(f"Output file           : {OUTPUT_FILE.resolve()}")

    print("\nFinal columns:")
    print(clean.columns.tolist())


if __name__ == "__main__":
    main()