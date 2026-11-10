# Panadería · predicción de desperdicio con MLflow

Proyecto de ejemplo para el módulo de MLOps. Usa el dataset real
`data/bakery_sales_dataset.csv` (19.321 líneas de venta, 40 columnas) para
predecir si una línea terminará con **desperdicio alto** (`Waste_Pct > 33.3 %`).

La guía completa, con la explicación paso a paso, los comandos y las métricas
verificadas, está en [`guia_mlflow_panaderia.html`](guia_mlflow_panaderia.html).

## El problema

Cada fila es una línea de una orden: producto, cantidad, precios, tienda,
clima, temporada, promoción y resultados de la venta. La variable objetivo se
construye así:

```python
Alto_Desperdicio = 1 si Waste_Pct > 33.3, si no 0
```

Para que el modelo sea útil **antes** de conocer el resultado de la venta, se
excluyen `Waste_Pct`, `Unsold_Units`, `Stock_Available`, `Units_Produced`,
`Units_Sold`, `Profit`, `Customer_Rating`, los identificadores y la fecha/hora.
El modelo aprende de producto, categoría, vida útil, almacenamiento,
perecibilidad, tienda, clima, temporada, promoción, precios y cantidad.

## Resultados de la ejecución de referencia

| Run | Modelo | accuracy | f1_macro |
|---|---|---:|---:|
| `rf_n100_d8` | RandomForest 100 árboles, profundidad 8 | 0.8875 | 0.8809 |
| `rf_n200_d12` | RandomForest 200 árboles, profundidad 12 | 0.8880 | 0.8816 |
| `evaluacion-champion` | evaluate de MLflow sobre el conjunto de prueba | 0.8880 | 0.8541 (f1 binario) |

El modelo campeón queda registrado como `clasificador-desperdicio`, versión 1,
con los alias `champion` y `challenger`.

## Estructura

```
.
├── README.md
├── guia_mlflow_panaderia.html   # guía completa del proyecto
├── requirements.txt
├── MLproject
├── .env.example
├── .gitignore
├── notebooks/
│   └── 01_exploracion.ipynb
├── src/
│   ├── config.py      # nombres, rutas, semilla y listas de columnas
│   ├── data.py        # carga, target y partición
│   ├── train.py       # entrena y registra en MLflow
│   ├── register.py    # registra el mejor run como 'challenger'
│   ├── evaluate.py    # evalúa el campeón con mlflow.models.evaluate
│   └── predict.py     # carga por alias y predice un CSV
├── data/
│   ├── bakery_sales_dataset.csv
│   └── nuevos_pedidos.csv   # 50 filas de ejemplo para predecir
├── mlflow.db          # (generado) metadatos
└── mlruns/            # (generado) artefactos
```

## Puesta en marcha

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

### 1. Entrenar

```bash
python src/train.py --n-estimators 100 --max-depth 8 --run-name rf_n100_d8

mlflow run . --env-manager local --experiment-name desperdicio-panaderia \
  -P n_estimators=200 -P max_depth=12 -P run_name=rf_n200_d12
```

### 2. Registrar y promover

```bash
python src/register.py
# Registrada la versión 1 (f1_macro=0.882) como 'challenger'
```

Para mover el alias `champion` a la versión registrada:

```bash
PYTHONPATH=src python - <<'EOF'
from mlflow import MlflowClient
from config import MODELO_REGISTRADO
c = MlflowClient()
v = c.get_model_version_by_alias(MODELO_REGISTRADO, "challenger")
c.set_registered_model_alias(MODELO_REGISTRADO, "champion", v.version)
print("champion ->", v.version)
EOF
```

### 3. Evaluar y predecir

```bash
python src/evaluate.py
python src/predict.py data/nuevos_pedidos.csv --alias champion
# prediccion
# 0    22
# 1    28
# Guardado en predicciones.csv
```

### 4. Abrir la interfaz

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# http://127.0.0.1:5000
```

### 5. Servir como API (opcional)

```bash
mlflow models serve -m "models:/clasificador-desperdicio@champion" \
  -p 5001 --env-manager local

curl http://127.0.0.1:5001/ping
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: text/csv" --data-binary @data/nuevos_pedidos.csv
```

## Notas

- El CSV original tiene BOM. Se lee con `encoding="utf-8-sig"`; sin eso la
  primera columna se llama `\ufeffOrder_ID`.
- El modelo es un `Pipeline` de scikit-learn: imputa nulos, codifica variables
  categóricas con one-hot y entrena un `RandomForestClassifier`.
- `skops_trusted_types` incluye `sklearn.tree._tree.Tree` y `numpy.dtype`, que
  son los tipos que MLflow 3.18 necesita autorizar para serializar el pipeline.
- `mlflow.models.evaluate` usa el experimento activo. `src/evaluate.py` llama a
  `mlflow.set_experiment` para que el run no termine en `Default`.
- `mlflow.db` y `mlruns/` se regeneran al entrenar; no van a Git.
