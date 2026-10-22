#################################################################################################################################################

    ######  #     #   #######   ######   #     #   #######      #######   #######   ######   #     #   #######   ######  #######  #######
    #       #     #   #         #    #   #     #      #         #     #   #     #   #    #   #     #   #         #          #     #     #
    #       #######   ###       #    #   #     #      #         #######   #####     #    #   #######   ###       #          #     #     #    
    #       #     #   #         #   ##   #     #      #         #         #   #     #    #      #      #         #          #     #     #   
    ######  #     #   #######   ######   #######   #######      #         #    #    ######      #      ######    ######     #     #######  
                                      # 
                                      
##################################################################################################################################################

### BASES DE DATOS ###

HOLSAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

`inegi.py` (geometría + población + elevación + ríos)
`chirps.py` (precipitación, zonal sobre polígono)
`nasa_power.py` (meteorología, centroide real)
`openaq.py` (PM2.5, point-in-polygon, solo Aguascalientes)
`maiac_aod.py` (AOD 1km, zonal sobre polígono, 11 municipios)
`cenapred.py` (ciclones, estática)

Para la ingesta se necesita obtener la clave API de Open AQ y el id del proyecto de Google Earth Engine.
Cada script sigue produciendo un parquet en `data/processed/` con:
 `date` (o nada si es estática)
 `municipio` (nombre exacto igual a las llaves de `MUNICIPIOS_AGS`)
 `source`, `year`y sus columnas de valor propias
 `municipio` sigue siendo el campo de join universal entre fuentes.

`config/settings.py` con constantes: `BBOX_AGS`, `MUNICIPIOS_AGS` con centroides fijos, `INEGI_MUNICIPIOS_AGS` con claves AGEM, rangos de fechas `FECHA_INICIO`/`FECHA_FIN` para ingesta diaria y `FECHA_INICIO_MASTER`/`FECHA_FIN_MASTER` para la tabla final, rutas RAW/PROCESSED/MASTER_DIR. 


                                #################################################
                                ######## Limpieza y Preparación de datos ########
                                #################################################

`clean.py` lee cada parquet de data/processed/ y escribe una versión validada en data/master/, sin modificar el origen. El problema de datos incompletos de OpenAQ se resuelve con la arquitectura que usa el maiac aod. Cada fuente tiene su limpiador, registrado en `FUENTES_DINAMICAS` (llave date+municipio) o `FUENTES_ESTATICAS` (llave municipio)
Todos aplican los mismos pasos: 
Normalizar el nombre del municipio contra el catálogo `MUNICIPIOS_AGS` (llave universal de join)
Convertir sentinelas a NaN y eliminar duplicados (XGBoost maneja NaN de forma nativa)
MAIAC valida aod_055_count, aod_055_mean y aod_qa_best_pct, marca celdas <10 px y conserva qc_aod_imputed como indicador de calidad
Si cualquier fuente falla, el pipeline se detiene con error para que no se construya un master incompleto

`build_master_table.py` Construye un grid canónico municipio por día (11 × 2,404 = 26,444 filas) y hace left join de las fuentes dinámicas por date+municipio, con validate="many_to_one" para detectar duplicados.
La ausencia de cualquier fuente es un error. De las estáticas solo se conservan elevación, distancia al río y centroide; población, viviendas, área y ciclones quedan en dim_municipios.parquet, porque con 11 municipios cada estática funciona como un identificador y los conteos históricos de ciclones incluyen años posteriores a cada fila. Las features derivadas usan solo información hasta el día t: lags de 1, 2, 3 y 7 días, medias móviles de 7 y 30 días, lluvia acumulada de 3 a 90 días, días sin lluvia, rango térmico, déficit de presión de vapor, AOD móvil de 7 días y estacionalidad cíclica. Los targets no se crean aquí porque dependen del horizonte, y year se conserva únicamente para el walk-forward

## conflicto de OpenAQ y MAIAC ##
Para evaluar las dos fuentes `clean.py` les aplicó controles de nombres de municipio normalizados contra el catálogo, sentinelas y valores fuera de rango convertidos a NaN, y duplicados eliminados. En MAIAC se verificó la coherencia entre aod_055_count, aod_055_mean y aod_qa_best_pct, y se marcaron las celdas con menos de 10 píxeles. 

En `build_master_table.py` OpenAQ quedó fuera del master porque cubre el 3.4 % de las celdas, todas de la capital. Para medir la utilidad de MAIAC se calculó la correlación de Spearman entre aod_055_mean y pm25_mean en la capital, solo con días de AOD observado:
(n = 896): ρ = -0.12 # es decir, casi sin señal
Con eso se descartó calibrar AOD a PM2.5, y como la proporción imputada fue 0 %, se confirmó que no hay fuga por gap filling (qc_aod_imputed es constante y no debe entrar a X). 
El AOD se mantiene en el master (79.4 % de cobertura), pero sus columnas no se usan como features del modelo evaluado. La evaluación walk-forward descrita abajo ya se usó para decidir qué modelos tienen habilidad y cuáles conviene congelar; no se plantea como una invitación a seguir entrenando hasta encontrar una mejora.

## Decisiones antes del Modelo ##
### Auditoría temporal y resultados de `rain_leakage.py`
`rain_leakage.py` reúne tres evaluaciones reproducibles 
`leakage` se reconstruyen las features con `build_master_table.add_features` y se comparan con las almacenadas en el master. Además, se alteran los valores de las fuentes en fechas posteriores a 12 cortes aleatorios y se comprueba que las features de fechas anteriores no cambien. Las 53 features evaluadas reproducen las del master y pasan esta prueba de no uso de información futura.
`rain` la única habilidad útil es la probabilidad de ocurrencia a un día: BSS = 0.24 y AUC = 0.85. La predicción del monto de lluvia, tanto en el modelo de dos etapas como con Tweedie, no supera la climatología. Por tanto, estos resultados respaldan congelar el clasificador de ocurrencia a h=1, no seguir ajustando un modelo de monto.
`scope`para `T2M_MIN`, el modelo global iguala o supera al ajuste independiente por municipio; a h=1 obtiene skill = 0.19 frente al mejor baseline evaluado. `T2M_MAX` no muestra habilidad en ninguno de los horizontes. A h=3 y h=7, ningún modelo supera el baseline suavizado.

Los CSV de estas corridas se escriben en `outputs_fase2/` (`rain_eval.csv` y `scope_<variable>.csv`). Los resultados acotan el uso operacional: temperatura mínima global y ocurrencia de lluvia, ambos a un día; no respaldan poner en producción `T2M_MAX`, horizontes de 3 o 7 días ni una predicción del monto de lluvia.

# Reglas de entrenamiento
Targets con groupby("municipio")[var].shift(-h), h ≥ 1, fuera del master.
X excluye date, municipio, year, los targets y qc_aod_imputed.
Walk-forward por año con ventana expansiva (folds 2023–2026; 2026 es año parcial).
Baselines: persistencia y climatología por municipio y día del año.
Corrida A (sin AOD) y corrida B (con AOD).

# Resultados
Primer test de XGBoost: AOD no aporta de forma consistente y queda fuera (usar_aod=False). Los skills iniciales, medidos contra una climatología sin suavizar, estaban inflados.

Fase 2, leakage: 12 cortes aleatorios, y las 53 features no usan información futura y reproducen el master.
Fase 2, lluvia (≥1 mm): solo hay habilidad en probabilidad de ocurrencia a h=1 (BSS 0.24, AUC 0.85, positivo en los 4 años). A h=3 y h=7 el BSS es negativo. Tweedie es peor que la climatología. El modelo de dos etapas gana en deviance solo en 2 de 4 años a h=1, por lo que el monto de lluvia no es confiable.
Fase 2, T2M_MIN: skill global contra el mejor baseline suavizado de 0.19 a h=1 (positivo en los 4 años), 0.03 a h=3 y −0.03 a h=7. El modelo global supera al de por municipio en 9 de 12 combinaciones, así que se elige el global.
Fase 2, T2M_MAX: skill de 0.003, −0.04 y −0.045 a h=1, 3 y 7, con un patrón que cambia de signo entre años (pierde en 2023-24, gana en 2025-26). Se excluye del piloto.
Conclusión: el piloto defendible es T2M_MIN global a h=1 y probabilidad de lluvia a h=1, a escala municipal, no vecinal.


# Monitoreo de resultados

La fase 3 en `monitor_rain-py` congela esos dos modelos (train [FECHA_CORTE]) y monitorea con monitor [VENTANA].
El monitoreo calcula el skill móvil de 90 días contra el baseline congelado. Si es negativo hoy y hace 90 días, el estado es RETIRAR; si solo es negativo hoy, ALERTA.
También vigila el porcentaje de NaN en las columnas base. Para ensayarlo hace falta un corte anterior a mediados de 2025 y al menos 180 días posteriores.

# Pendientes y límites conocidos actualmente

Definir quién re-entrena, monitorea y retira el modelo.
Falta probar que la construcción de targets no tenga leakage; la prueba actual solo cubre features.
Tampoco se ha verificado la calibración de las probabilidades de lluvia.
Con 4 folds, los intervalos de confianza no son creíbles.
Falta una regla para detectar cambios en NASA POWER, CHIRPS o MAIAC (revisiones históricas, flags de QA).
Hay que revisar los términos de uso de INEGI, CENAPRED y OpenAQ.

# Ejecución
Comandos para instalar requirements.txt:
Windows: install.bat
Linux/MacOS: bash install.sh

Activar env con:
source .venv/bin/activate


# Datos

data/
├── processed/   # salida de la ingesta por fuente (no versionado)
└── master/      # datos validados y tabla final (no versionado)
    ├── <fuente>.parquet
    ├── master_table.parquet
    ├── dim_municipios.parquet
    └── reports/
```

`config/settings.py` crea estas carpetas automáticamente.

## Fuentes

| Fuente | Contenido | Credenciales |
| NASA POWER | T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, PS, radiación, precipitación (~50 km) | No |
| CHIRPS v2.0 | Precipitación diaria (~5 km) | No |
| MAIAC (AOD 0.55 µm) | Profundidad óptica de aerosoles | `GEE_PROJECT` (Google Earth Engine) |
| INEGI | Geometría, población, elevación, ríos | `INEGI_TOKEN` (opcional) |
| CENAPRED | Ciclones históricos | No |
| OpenAQ v3 | PM2.5 (solo capital; **excluido del master**) | `OPENAQ_API_KEY` |


##### Data Sources, Licensing and Attribution
This project uses publicly accessible environmental, meteorological, geospatial, and air-quality data to develop a reproducible environmental-risk analysis pipeline for the 11 municipalities of Aguascalientes, Mexico.

### Data sources
The project retrieves and processes data from the following providers:

| Source | Data used | Official documentation |
| NASA POWER | Near-surface temperature, relative humidity, wind speed, pressure, solar radiation, and precipitation variables | https://power.larc.nasa.gov/docs/ |
| CHIRPS v2.0 | Daily precipitation estimates | https://www.chc.ucsb.edu/data/chirps |
| NASA MAIAC / MODIS | Aerosol Optical Depth (AOD), approximately 0.55 µm | https://developers.google.com/earth-engine/datasets |
| INEGI | Municipal boundaries, population, elevation, and river-related geographic information | https://www.inegi.org.mx/ |
| CENAPRED | Historical cyclone-related information, subject to the terms of the specific source dataset | https://www.gob.mx/cenapred |
| OpenAQ API v3 | PM2.5 measurements for the municipality of Aguascalientes; excluded from the master dataset in the current pipeline | https://docs.openaq.org/ |

### Licensing and attribution policy

The project distinguishes between the software developed for this repository and the third party datasets accessed or processed by its scripts.

- **NASA POWER:** NASA POWER data are acknowledged using the provider's recommended reference and data citation guidance. The version and access date should be recorded for each dataset used.
- **CHIRPS:** CHIRPS data are publicly available for use and redistribution under the applicable terms of the product. Appropriate source attribution and scientific references should be retained.
- **MAIAC / MODIS:** The licensing and redistribution conditions of the exact satellite product must be checked independently of the Google Earth Engine service terms. This repository does not claim ownership of the original satellite data.
- **INEGI:** Information is attributed to INEGI in accordance with its official terms of free use. Source metadata are preserved where applicable, and modifications and derived analyses are identified as work performed by this project.
- **CENAPRED:** The terms of the specific dataset are reviewed before redistributing source files. Public accessibility alone is not treated as proof of unrestricted redistribution rights.
- **OpenAQ:** The applicable license and attribution requirements of both OpenAQ and the original data provider are checked before redistributing measurements or derived datasets.

### Data redistribution

This repository prioritizes reproducibility through documented extraction and processing scripts rather than unrestricted redistribution of third party data.

Unless a dataset's applicable terms have been verified, downloaded source files and derived data files are not included in the public repository. Users should obtain the data from the official providers and comply with the applicable terms of use.

Processed datasets, including the master table, may contain information originating from multiple providers. Their redistribution must be assessed against the conditions applicable to the underlying sources.

### API credentials

Credentials are used only to access the relevant services and must not be committed to version control.

The following environment variables may be required:

- `GEE_PROJECT`
- `INEGI_TOKEN` (optional, depending on the access method)
- `OPENAQ_API_KEY`

Users should configure their own credentials locally, following the instructions in the project setup documentation. Secret values, personal access tokens, and credential files must remain outside the public repository.

### Disclaimer

This project is an independent research and educational initiative. It is not affiliated with, endorsed by, or officially supported by NASA, Google, the Climate Hazards Center, INEGI, CENAPRED, or OpenAQ.

Environmental datasets may contain measurement uncertainties, estimation errors, missing observations, and changes in methodology or availability. Results generated by this project are independent analyses and should not be interpreted as official assessments or warnings issued by the data providers.

Users are responsible for verifying the applicable data licenses and terms before redistributing datasets, publishing derived products, or using the results for operational or commercial purposes.



