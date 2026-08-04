import polars as pl


# ==========================================================
# 1. Configuration
# ==========================================================

INPUT_FILE = "../feature-engineering/Dataset_AROME_METAR_features.parquet"
OUTPUT_PARQUET = "AROME_METAR_features.parquet"
OUTPUT_CSV = "AROME_METAR_features.csv"

KT_TO_MPS = 0.514444


# ==========================================================
# 2. Load dataset
# ==========================================================

df = pl.read_parquet(INPUT_FILE)

print("Initial shape:", df.shape)


# ==========================================================
# 3. Check required columns
# ==========================================================

required_columns = {
    "wind_mean",
    "wind_final",
    "speed_gust60",
    "has_gust"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


# ==========================================================
# 4. Replace METAR wind values from knots to m/s
# ==========================================================

df = df.with_columns([
    (
        pl.col("wind_mean")
        .cast(pl.Float64, strict=False)
        * KT_TO_MPS
    ).alias("wind_mean"),

    (
        pl.col("wind_final")
        .cast(pl.Float64, strict=False)
        * KT_TO_MPS
    ).alias("wind_final")
])


# ==========================================================
# 5. Create AROME error and correction target
# ==========================================================

df = df.with_columns([
    (
        pl.col("speed_gust60")
        - pl.col("wind_final")
    ).alias("arome_error"),

    (
        pl.col("wind_final")
        - pl.col("speed_gust60")
    ).alias("correction_target")
])


# ==========================================================
# 6. Evaluate AROME on true gust events
# ==========================================================

df_gust = df.filter(
    pl.col("has_gust") == 1
)

evaluation = df_gust.select([
    pl.len().alias("number_of_gusts"),

    pl.col("arome_error")
    .mean()
    .alias("bias_mps"),

    pl.col("arome_error")
    .abs()
    .mean()
    .alias("mae_mps"),

    (
        pl.col("arome_error") ** 2
    )
    .mean()
    .sqrt()
    .alias("rmse_mps")
])

print("\nAROME evaluation on true gust events:")
print(evaluation)


# ==========================================================
# 7. Verify statistics after conversion
# ==========================================================

print("\nWind statistics after conversion to m/s:")
print(
    df.select([
        "speed_gust60",
        "wind_mean",
        "wind_final"
    ]).describe()
)


# ==========================================================
# 8. Export
# ==========================================================

df.write_parquet(OUTPUT_PARQUET)

df.write_csv(
    OUTPUT_CSV,
    datetime_format="%Y-%m-%d %H:%M:%S"
)

print("\nFiles created:")
print(f"- {OUTPUT_PARQUET}")
print(f"- {OUTPUT_CSV}")