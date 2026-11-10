"""Evalúa el modelo campeón sobre el conjunto de prueba y guarda las métricas en MLflow."""
import mlflow

from config import EXPERIMENTO, MODELO_REGISTRADO, OBJETIVO
from data import UMBRAL_DESPERDICIO, cargar_datos

# `mlflow.models.evaluate` guarda el resultado en el experimento activo.
# Sin esta línea, el run de evaluación termina en "Default".
mlflow.set_experiment(EXPERIMENTO)

X_train, X_test, y_train, y_test = cargar_datos()
datos = X_test.assign(**{OBJETIVO: y_test.values})

with mlflow.start_run(run_name="evaluacion-champion") as run:
    resultado = mlflow.models.evaluate(
        model=f"models:/{MODELO_REGISTRADO}@champion",
        data=datos,
        targets=OBJETIVO,
        model_type="classifier",
        # shap no es necesario para este ejemplo y con pandas 3 puede fallar
        evaluator_config={"log_model_explainability": False},
    )
    print(f"run_id={run.info.run_id}")
    for nombre, valor in resultado.metrics.items():
        if isinstance(valor, (int, float)):
            print(f"{nombre}={valor:.4f}")
    print(f"umbral_desperdicio={UMBRAL_DESPERDICIO}")
