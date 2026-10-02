from cenapred import ingest_cenapred
from chirps import ingest_chirps
from nasa_power import ingest_nasa_power
from openaq import ingest_openaq
from inegi import ingest_inegi
from maiac import ingest_maiac
from config.settings import FECHA_INICIO, FECHA_FIN

def ingest_all(start, end) -> None:
    ingest_inegi(start, end) ## debe ejecutarse primero para generar geometria_municipios.parquet, que es requerida por los otros scripts
    ingest_cenapred(start, end)
    ingest_chirps(start, end)
    ingest_nasa_power(start, end)
    ingest_openaq(start, end) # obtener y definir OPENAQ_API_KEY 
    ingest_maiac(start, end) # sesion de Earth Engine debe estar autenticada y definida en setting.py

if __name__ == "__main__":
    ingest_all(FECHA_INICIO, FECHA_FIN)