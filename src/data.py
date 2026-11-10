"""Carga, preparación y partición del dataset de ventas de panadería.

Si cambian los datos o las columnas, se cambia solo este archivo.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import COLUMNAS_EXCLUIDAS, OBJETIVO, RUTA_DATOS, SEMILLA

UMBRAL_DESPERDICIO = 33.3  # % de desperdicio a partir del cual se considera "alto"


def cargar_datos(test_size=0.2):
    """Devuelve X_train, X_test, y_train, y_test con la misma partición siempre.

    El CSV original viene con BOM (utf-8-sig); sin esa codificación la primera
    columna se llamaría "\\ufeffOrder_ID".

    La variable objetivo se construye a partir de Waste_Pct, pero Waste_Pct y
    las demás columnas posteriores a la venta se excluyen de X: el modelo debe
    decidir ANTES de conocer cuánto se vendió.
    """
    df = pd.read_csv(RUTA_DATOS, encoding="utf-8-sig")
    df[OBJETIVO] = (df["Waste_Pct"] > UMBRAL_DESPERDICIO).astype(int)

    X = df.drop(columns=COLUMNAS_EXCLUIDAS)
    y = df[OBJETIVO]

    return train_test_split(
        X, y, test_size=test_size, random_state=SEMILLA, stratify=y
    )
