import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "../Arome/Datasetfinal"
OUTPUT_FILE = PROJECT_ROOT / "Datasetfinal.csv"
REJECTED_FILE = PROJECT_ROOT / "rejected_structure.tsv"


COLUMNS = [
    "station",
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
]


def convert_datetime(raw_datetime: str) -> str:
    """
    Convertit une date du format YYYYMMDDHH
    vers YYYY-MM-DD HH:MM:SS.
    """
    parsed_datetime = datetime.strptime(
        raw_datetime,
        "%Y%m%d%H",
    )

    return parsed_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def normalize_row(
    tokens: list[str],
) -> tuple[list[str] | None, str | None]:
    """
    Normalise une ligne AROME.

    - 18 champs : conserve l'identifiant original de la station.
    - 17 champs : ajoute une station vide.
    - Convertit la date au format datetime.
    - Les autres structures sont rejetées.
    """
    field_count = len(tokens)

    # Ligne complète avec station.
    if field_count == 18:
        try:
            formatted_datetime = convert_datetime(tokens[1])
        except ValueError:
            return None, "invalid_datetime"

        normalized = [
            tokens[0],
            formatted_datetime,
            *tokens[2:],
        ]

        return normalized, None

    # Ligne sans identifiant de station.
    if field_count == 17:
        try:
            formatted_datetime = convert_datetime(tokens[0])
        except ValueError:
            return None, "invalid_datetime"

        normalized = [
            "",
            formatted_datetime,
            *tokens[1:],
        ]

        return normalized, None

    if field_count == 16:
        return None, "16_fields_missing_information"

    return None, f"unexpected_field_count_{field_count}"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE.resolve()}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_lines = 0
    rows_from_18 = 0
    rows_from_17 = 0
    invalid_dates = 0
    rejected_lines = 0

    with (
        INPUT_FILE.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source,
        OUTPUT_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as destination,
        REJECTED_FILE.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as rejected_file,
    ):
        writer = csv.writer(destination)
        writer.writerow(COLUMNS)

        rejected_writer = csv.writer(
            rejected_file,
            delimiter="\t",
        )

        rejected_writer.writerow(
            [
                "line_number",
                "field_count",
                "reason",
                "raw_preview",
            ]
        )

        for line_number, raw_line in enumerate(
            source,
            start=1,
        ):
            total_lines += 1

            tokens = raw_line.strip().split()

            if not tokens:
                rejected_lines += 1

                rejected_writer.writerow(
                    [
                        line_number,
                        0,
                        "empty_line",
                        "",
                    ]
                )
                continue

            normalized, reason = normalize_row(tokens)

            if reason is not None:
                rejected_lines += 1

                if reason == "invalid_datetime":
                    invalid_dates += 1

                rejected_writer.writerow(
                    [
                        line_number,
                        len(tokens),
                        reason,
                        raw_line.strip()[:150],
                    ]
                )
                continue

            if len(tokens) == 18:
                rows_from_18 += 1

            elif len(tokens) == 17:
                rows_from_17 += 1

            writer.writerow(normalized)

    print("=== Structure validation report ===")
    print(f"Lines read                    : {total_lines:,}")
    print(f"Accepted lines with 18 fields : {rows_from_18:,}")
    print(f"Accepted lines with 17 fields : {rows_from_17:,}")
    print(f"Invalid dates                 : {invalid_dates:,}")
    print(f"Rejected lines                : {rejected_lines:,}")
    print(
        f"Rows written                  : "
        f"{rows_from_18 + rows_from_17:,}"
    )
    print(f"Output file                   : {OUTPUT_FILE.resolve()}")
    print(f"Rejected file                 : {REJECTED_FILE.resolve()}")


if __name__ == "__main__":
    main()