import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comun import RUTA_CRUDO, RUTA_MODELO, RUTA_PROCESADO  # noqa: E402


@pytest.fixture(scope="session")
def crudo():
    from comun import cargar_crudo
    return cargar_crudo(RUTA_CRUDO)


@pytest.fixture(scope="session")
def limpio(crudo):
    from comun import limpiar
    return limpiar(crudo)


@pytest.fixture(scope="session")
def por_hora(limpio):
    from comun import agregar_por_hora
    return agregar_por_hora(limpio)


@pytest.fixture(scope="session")
def procesado():
    if not RUTA_PROCESADO.exists():
        pytest.skip("Falta data/processed/salidas_por_hora.csv: ejecute src/preparar_datos.py")
    return pd.read_csv(RUTA_PROCESADO, dtype={"codigo_estacion": str, "codigo_zona": str})


@pytest.fixture(scope="session")
def paquete():
    if not RUTA_MODELO.exists():
        pytest.skip("Falta el modelo: ejecute src/entrenar_modelo.py")
    import joblib
    return joblib.load(RUTA_MODELO)
