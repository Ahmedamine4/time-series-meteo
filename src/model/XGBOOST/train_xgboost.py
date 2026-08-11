import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve
)


# ============================================================
# 1. PARAMÈTRES
# ============================================================

DATASET_FILE = "C:\\Intern\\time-series-meteo\\data\\processed\\Dataset_features.parquet"

TARGET = "has_gust"
DATE_COLUMN = "datetime"
RANDOM_STATE = 42

MODEL_FILE = "modele_xgboost.joblib"
METRICS_FILE = "C:\\Intern\\time-series-meteo\\outputs\\metrics_xgboost.json"
PREDICTIONS_FILE = "predictions_xgboost_2025.csv"
IMPORTANCE_FILE = "importance_features_xgboost.csv"
CONFUSION_FILE = "confusion_matrix_xgboost_2025.png"


# 2. VARIABLES UTILISÉES

FEATURES = [
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
    "lat",
    "lon",
    "elevation_m",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos"
]


# 3. LIRE LE DATASET

df = pd.read_parquet(DATASET_FILE)

if DATE_COLUMN not in df.columns:
    raise ValueError(
        f"La colonne {DATE_COLUMN} est absente du dataset."
    )

if TARGET not in df.columns:
    raise ValueError(
        f"La colonne {TARGET} est absente du dataset."
    )


# 4. CRÉER LES VARIABLES TEMPORELLES

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    errors="coerce"
)

df = df.dropna(
    subset=[DATE_COLUMN, TARGET]
).copy()

df["hour"] = df[DATE_COLUMN].dt.hour # Ajout de la colonne "hour" pour l'heure de la journée
df["day_of_year"] = df[DATE_COLUMN].dt.dayofyear # Ajout de la colonne "day_of_year" pour le jour de l'année


# 5. VÉRIFIER LES VARIABLES

features = [
    colonne
    for colonne in FEATURES
    if colonne in df.columns
]

colonnes_manquantes = [
    colonne
    for colonne in FEATURES
    if colonne not in df.columns
]

if colonnes_manquantes:
    print(
        "Attention, variables absentes :",
        colonnes_manquantes
    )

if not features:
    raise ValueError(
        "Aucune variable d'entraînement n'est disponible."
    )

for colonne in features:
    df[colonne] = pd.to_numeric(
        df[colonne],
        errors="coerce"
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
).copy()

df[TARGET] = df[TARGET].astype(int)


# 6. SPLIT CHRONOLOGIQUE

train = df[
    df["year"].between(2021, 2023)
].copy()

validation = df[
    df["year"] == 2024
].copy()

test = df[
    df["year"] == 2025
].copy()

print("Train 2021-2023 :", train.shape)
print("Validation 2024 :", validation.shape)
print("Test 2025 :", test.shape)

if train.empty:
    raise ValueError("Le dataset d'entraînement est vide.")

if validation.empty:
    raise ValueError("Le dataset de validation est vide.")

if test.empty:
    raise ValueError("Le dataset de test est vide.")


# 7. PRÉPARER X ET Y

X_train = train[features]
y_train = train[TARGET].to_numpy()

X_validation = validation[features]
y_validation = validation[TARGET].to_numpy()

X_test = test[features]
y_test = test[TARGET].to_numpy()


# 8. REMPLIR LES VALEURS MANQUANTES

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = imputer.fit_transform(
    X_train
)

X_validation_imputed = imputer.transform(
    X_validation
)

X_test_imputed = imputer.transform(
    X_test
)


# 9. CALCULER SCALE_POS_WEIGHT car le dataset est déséquilibré

nb_negatifs = int(
    (y_train == 0).sum()
)

nb_positifs = int(
    (y_train == 1).sum()
)

if nb_positifs == 0:
    raise ValueError(
        "Aucune observation has_gust = 1 dans le train."
    )

scale_pos_weight = (
    nb_negatifs / nb_positifs
)

print()
print("Nombre classe 0 :", nb_negatifs)
print("Nombre classe 1 :", nb_positifs)
print(
    "scale_pos_weight :",
    round(scale_pos_weight, 4)
)



# 10 . CRÉER ET ENTRAÎNER XGBOOST

model = XGBClassifier(
    n_estimators=500, # Nombre d'arbres
    learning_rate=0.05, # Taux d'apprentissage
    max_depth=6, # Profondeur maximale des arbres
    min_child_weight=2, # Poids minimum d'un enfant
    subsample=0.8, # Proportion d'échantillons utilisés pour chaque arbre ( ici 80% )
    colsample_bytree=0.8, # Proportion de caractéristiques utilisées pour chaque arbre ( ici 80% )
    gamma=0,
    reg_alpha=0, # Poids de la régularisation L1
    reg_lambda=1, # Poids de la régularisation L2
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train_imputed,
    y_train,
    eval_set=[
        (
            X_train_imputed,
            y_train
        ),
        (
            X_validation_imputed,
            y_validation
        )
    ],
    verbose=False
)

print()
print("Entraînement terminé.")


# 11. CHERCHER LE MEILLEUR SEUIL SUR VALIDATION

proba_validation = model.predict_proba(
    X_validation_imputed
)[:, 1]

pr_auc_validation = average_precision_score(
    y_validation,
    proba_validation
)

precision, recall, thresholds = precision_recall_curve(
    y_validation,
    proba_validation
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    /
    (
        precision[:-1]
        + recall[:-1]
        + 1e-10
    )
)

best_index = int(
    np.argmax(f1_scores)
)

best_threshold = float(
    thresholds[best_index]
)

prediction_validation = (
    proba_validation >= best_threshold
).astype(int)

print()
print("VALIDATION 2024")
print("PR-AUC :", round(pr_auc_validation, 4))
print("Meilleur seuil :", round(best_threshold, 4))
print("Precision :", round(float(precision[best_index]), 4))
print("Recall :", round(float(recall[best_index]), 4))
print("F1-score :", round(float(f1_scores[best_index]), 4))

print()
print(
    classification_report(
        y_validation,
        prediction_validation,
        digits=3,
        zero_division=0
    )
)

print(
    confusion_matrix(
        y_validation,
        prediction_validation
    )
)


# 12. ÉVALUER SUR LE TEST 2025

proba_test = model.predict_proba(
    X_test_imputed
)[:, 1]

prediction_test = (
    proba_test >= best_threshold
).astype(int)

pr_auc_test = average_precision_score(
    y_test,
    proba_test
)

print()
print("==============================")
print("TEST 2025")
print("==============================")
print("PR-AUC :", round(pr_auc_test, 4))

print()
print(
    classification_report(
        y_test,
        prediction_test,
        digits=3,
        zero_division=0
    )
)

test_confusion_matrix = confusion_matrix(
    y_test,
    prediction_test
)

print("Matrice de confusion :")
print(test_confusion_matrix)


# 13. ENREGISTRER LA MATRICE DE CONFUSION

display = ConfusionMatrixDisplay(
    confusion_matrix=test_confusion_matrix,
    display_labels=[
        "Pas de rafale",
        "Rafale"
    ]
)

display.plot(values_format="d")
plt.title("Matrice de confusion XGBoost - Test 2025")
plt.tight_layout()
plt.savefig(
    CONFUSION_FILE,
    dpi=160
)
plt.close()


# 14. IMPORTANCE DES VARIABLES

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

print()
print("Variables les plus importantes :")
print(
    importance_df.head(15).to_string(index=False)
)



# 15. ENREGISTRER LES PRÉDICTIONS DU TEST

result_columns = [
    colonne
    for colonne in [
        "id",
        "icao",
        "station",
        DATE_COLUMN,
        TARGET
    ]
    if colonne in test.columns
]

predictions = test[result_columns].copy()
predictions["probabilite_has_gust"] = proba_test
predictions["prediction"] = prediction_test

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# 16. ENREGISTRER LE MODÈLE ET LES MÉTRIQUES

joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": best_threshold
    },
    MODEL_FILE
)

metrics = {
    "model": "xgboost",
    "train_period": "2021-2023",
    "validation_period": "2024",
    "test_period": "2025",
    "scale_pos_weight": float(scale_pos_weight),
    "best_threshold": best_threshold,
    "pr_auc_validation": float(pr_auc_validation),
    "f1_validation": float(f1_scores[best_index]),
    "pr_auc_test": float(pr_auc_test),
    "features": features,
    "classification_report_test": classification_report(
        y_test,
        prediction_test,
        output_dict=True,
        zero_division=0
    )
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        metrics,
        file,
        indent=2,
        ensure_ascii=False
    )

print()
print("Fichiers créés :")
print("-", MODEL_FILE)
print("-", METRICS_FILE)
print("-", PREDICTIONS_FILE)
print("-", IMPORTANCE_FILE)
print("-", CONFUSION_FILE)
