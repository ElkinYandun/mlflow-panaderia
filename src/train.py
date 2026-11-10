"""Entrena un modelo y registra parámetros, métricas, artefactos y modelo en MLflow."""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # genera gráficos sin abrir ventanas (scripts y servidores)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import CATEGORICAS, EXPERIMENTO, NUMERICAS, OBJETIVO, RUTA_DATOS, SEMILLA
from data import cargar_datos


def construir_modelo(n_estimators, max_depth):
    """Pipeline: imputa nulos, codifica variables de texto y entrena el bosque."""
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("nulos", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAS,
            ),
            ("numericas", "passthrough", NUMERICAS),
        ]
    )
    clasificador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=SEMILLA,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocesador", preprocesador), ("clasificador", clasificador)]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de desperdicio alto en panadería con MLflow"
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--run-name", default=None, help="nombre legible del run")
    args = parser.parse_args()

    # `mlflow run` ya asigna el experimento; con `python src/train.py` lo asignamos aquí
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENTO)

    X_train, X_test, y_train, y_test = cargar_datos()

    with mlflow.start_run(run_name=args.run_name) as run:
        # 1) Qué se probó
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "seed": SEMILLA,
                "test_size": 0.2,
            }
        )
        mlflow.set_tags(
            {
                "modelo": "RandomForest",
                "dataset": "bakery_sales_dataset.csv",
                "objetivo": OBJETIVO,
                "umbral_desperdicio": "33.3",
                "etapa": "desarrollo",
            }
        )
        mlflow.set_tag(
            "mlflow.note.content",
            "Predice si una línea de venta terminará con desperdicio alto "
            "(Waste_Pct > 33.3 %) usando solo información disponible antes de "
            "la venta: producto, vida útil, almacenamiento, tienda, clima, "
            "temporada, promoción y precios.",
        )

        # 2) Entrenar
        modelo = construir_modelo(args.n_estimators, args.max_depth)
        modelo.fit(X_train, y_train)

        # 3) Evaluar con datos que el modelo no vio y guardar las métricas
        pred = modelo.predict(X_test)
        metricas = {
            "accuracy": accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro"),
            "precision_macro": precision_score(
                y_test, pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "n_filas_prueba": len(y_test),
        }
        mlflow.log_metrics(metricas)

        # 4) Un gráfico como artefacto
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title("Matriz de confusión · desperdicio alto")
        mlflow.log_figure(fig, "matriz_confusion.png")
        plt.close(fig)

        # 5) El dataset usado, con su huella (hash) y su esquema
        df_train = X_train.assign(**{OBJETIVO: y_train})
        dataset = mlflow.data.from_pandas(
            df_train,
            source=str(RUTA_DATOS),
            name="bakery-train",
            targets=OBJETIVO,
        )
        mlflow.log_input(dataset, context="training")

        # 6) Una muestra de las predicciones, para revisarlas después
        muestra = X_test.head(25).copy()
        muestra["real"] = y_test.head(25).values
        muestra["prediccion"] = pred[:25]
        ruta_muestra = "muestra_predicciones.csv"
        muestra.to_csv(ruta_muestra, index=False)
        mlflow.log_artifact(ruta_muestra, artifact_path="datos")

        # 7) El modelo, con su firma y un ejemplo de entrada
        info = mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X_train, modelo.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )
        mlflow.set_tag("model_uri", info.model_uri)  # para encontrar el modelo después

        print(f"run_id={run.info.run_id}")
        print(
            f"accuracy={metricas['accuracy']:.3f}  "
            f"f1_macro={metricas['f1_macro']:.3f}  "
            f"model_uri={info.model_uri}"
        )


if __name__ == "__main__":
    main()
