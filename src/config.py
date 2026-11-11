"""Constantes del proyecto. Un solo lugar para nombres, rutas y semilla."""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "data" / "bakery_sales_dataset.csv"

EXPERIMENTO = "desperdicio-panaderia"          # nombre del experimento en MLflow
MODELO_REGISTRADO = "clasificador-desperdicio"  # nombre en el Model Registry
MODELO_ALIAS = "champion"                       # alias que consume la API por defecto
SEMILLA = 42                                    # misma partición de datos en cada ejecución

# Ruta absoluta al almacén de MLflow: funciona aunque se ejecute desde otra carpeta.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{RAIZ / 'mlflow.db'}"
)

OBJETIVO = "Alto_Desperdicio"                   # 1 = Waste_Pct > 33.3 %, 0 = resto

# link del database: https://www.kaggle.com/datasets/sdcloud/historical-bakery-sales-dataset
# Columnas que no entran al modelo
COLUMNAS_EXCLUIDAS = [
    "Order_ID",
    "Customer_ID",
    "Date",
    "Time",
    "Customer_Rating",
    "Waste_Pct",
    "Unsold_Units",
    "Stock_Available",
    "Units_Produced",
    "Units_Sold",
    "Profit",
    OBJETIVO,
]

CATEGORICAS = [
    "Customer_Gender",
    "Customer_Segment",
    "Loyalty_Member",
    "Day_of_Week",
    "Weekend",
    "Season",
    "Festival",
    "Weather",
    "Product",
    "Category",
    "Payment_Method",
    "Promotion_Applied",
    "Promotion_Type",
    "Storage_Type",
    "Perishability",
    "Store_ID",
]

NUMERICAS = [
    "Customer_Age",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Temperature",
    "Quantity",
    "Unit_Price",
    "Cost_Price",
    "Discount_Percentage",
    "Selling_Price",
    "Total_Bill",
    "Shelf_Life_Days",
]


def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)
