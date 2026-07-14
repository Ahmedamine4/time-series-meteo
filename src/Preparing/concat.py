import polars as pl
import datetime
import os

# Chemin du dossier où chercher
folder_path =os.getcwd()   # à remplacer par ton chemin

# Lister tous les fichiers .parquet
parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]

if not parquet_files:
    print("Aucun fichier .parquet trouvé.")
else:
    print("Fichiers trouvés :")
    for i, file in enumerate(parquet_files, start=1):
        print(f"{i}. {file}")

    # Demander à l'utilisateur de choisir deux fichiers
    try:
        choice1 = int(input("Entrez le numéro du premier fichier : ")) - 1
        choice2 = int(input("Entrez le numéro du deuxième fichier : ")) - 1

        if 0 <= choice1 < len(parquet_files) and 0 <= choice2 < len(parquet_files):
            file1 = os.path.join(folder_path, parquet_files[choice1])
            file2 = os.path.join(folder_path, parquet_files[choice2])
            print(f"Vous avez choisi :\n - {file1}\n - {file2}")
        else:
            print("Numéros invalides.")
    except ValueError:
        print("Veuillez entrer des nombres valides.")

df1 = pl.read_parquet(file1)
df2= pl.read_parquet(file2)

a=df2
if df2["time"].min() < df1["time"].min():
    df2=df1
df1=a
df1=df1.filter(pl.col("time") < df2["time"].min())
df_concat = pl.concat([df1, df2], how="vertical")
print(df_concat.describe())

df_concat.write_csv("Rafale_METAR.csv")
df_concat.write_parquet("Rafale_METAR.parquet")