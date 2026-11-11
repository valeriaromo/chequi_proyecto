    ######  #     #   #######   ######   #     #   #######      #######   #######   ######   #     #   #######   ######  #######  #######
    #       #     #   #         #    #   #     #      #         #     #   #     #   #    #   #     #   #         #          #     #     #
    #       #######   ###       #    #   #     #      #         #######   #####     #    #   #######   ###       #          #     #     #    
    #       #     #   #         #   ##   #     #      #         #         #   #     #    #      #      #         #          #     #     #   
    ######  #     #   #######   ######   #######   #######      #         #    #    ######      #      ######    ######     #     #######  
                                      # 
                                      
# Pipeline de riesgo ambiental para los 11 municipios de Aguascalientes

Pipeline reproducible que ingiere datos ambientales públicos, construye una tabla maestra diaria por municipio, audita fugas temporales, evalúa modelos con walk-forward y monitorea los modelos congelados. Alcance piloto: municipal, diario, horizonte `h=1`, con dos objetivos: probabilidad de lluvia (`≥ 1 mm`) y temperatura mínima (`T2M_MIN`).

---

## Contenido

1. [Estructura del repositorio](#estructura-del-repositorio)
2. [Ejecución e instalación](#ejecución-e-instalación)
3. [Bases de datos e ingesta](#bases-de-datos-e-ingesta)
4. [Limpieza y tabla maestra](#limpieza-y-tabla-maestra)
5. [Tabla con PCA](#tabla-con-pca)
6. [Conflicto de OpenAQ y MAIAC](#conflicto-de-openaq-y-maiac)
7. [Decisiones antes del modelo](#decisiones-antes-del-modelo)
8. [Reglas de entrenamiento](#reglas-de-entrenamiento)
9. [Resultados de la línea XGBoost](#resultados-de-la-línea-xgboost)
10. [Comparación de modelos (`src/pipeline/models`)](#comparación-de-modelos-srcpipelinemodels)
11. [Cómo analizar los resultados](#cómo-analizar-los-resultados)
12. [Monitoreo](#monitoreo)
13. [Pendientes y límites conocidos](#pendientes-y-límites-conocidos)
14. [Datos, licencias y atribución](#datos-licencias-y-atribución)

---

## Estructura del repositorio

config/                      settings.py: constantes (BBOX_AGS, MUNICIPIOS_AGS, fechas, rutas)
src/
├── ingestion/               un script por fuente (inegi, chirps, nasa_power, openaq, maiac_aod, cenapred)
├── pipeline/
│   ├── clean.py             validación por fuente: data/processed -> data/master
│   ├── build_master_table.py  grid municipio x día + features derivadas
│   ├── build_master_pca.py  tabla maestra + componentes principales por bloque
│   ├── rain_leakage.py      auditoría temporal, lluvia y alcance (Fase 2)
│   ├── monitor_rain.py      monitoreo de modelos congelados (Fase 3)
│   ├── outputs_fase2/       CSV de la Fase 2 (rain_eval.csv, scope_<variable>.csv)
│   └── models/              comparación de 6 modelos sklearn (ver sección propia)
│       └── outputs/         resultados por tabla y feature set
├── data_analysis/           análisis exploratorio y PCA (pca_walkforward, pca_variantes, pca_bloques,
│                            ref_produccion, regularizado, base8, sin_aod, variable_analysis)
├── utils/                   utilidades compartidas (p. ej. io_utils)
app/                         aplicación web (routes, services, templates, static, data/mock)


Las carpetas `app/` y `src/data_analysis/` solo se describen por su nombre; documenta aquí su propósito y cómo se ejecutan.

---

## Ejecución e instalación

1. Instalar dependencias: Windows `install.bat`; Linux/macOS `bash install.sh`.
2. Activar el entorno: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
3. Credenciales (variables de entorno, nunca en git): `GEE_PROJECT`, `OPENAQ_API_KEY`, `INEGI_TOKEN` (opcional).

### `src/data_analysis/`

Contiene la suite analítica exploratoria y experimental previa al pipeline de producción:

- `variable_analysis.py`: Genera mapas de calor de correlación (Spearman/Pearson) y la matriz de hipótesis.
- `pca_bloques.py` & `pca_walkforward.py`: Experimentos de reducción dimensional por bloques y validación de la estabilidad de los eigenvectors en ventanas *walk-forward* (sin usar datos del futuro).
- `output/`: Almacena los reportes tabulares (`analysis_report.md`), visualizaciones de distribuciones y resúmenes por fold.

### `app/`

Consola de visualización Web construida en **Flask** para auditoría e interpretabilidad operacional del modelo:

- `/dashboard`: Gráficos interactivos (`charts.js`) con indicadores de riesgo pluviométrico y térmico por municipio.
- `/alerts`: Matriz y semaforización de alertas meteorológicas (`alert.js`).
- `/model/shap` y `/model/walk_forward`: Vistas técnicas de diagnóstico de *SHAP values* (`shap.js`) y descomposición del rendimiento por fold de entrenamiento.
- `data/mock/`: Archivos JSON de prueba que permiten testear la interfaz sin procesar datasets pesados.

### Orden del pipeline

```bash
python -m src.ingestion.<fuente>              # una por fuente
python -m src.pipeline.clean
python -m src.pipeline.build_master_table
python -m src.pipeline.build_master_pca       # opcional, ver "Tabla con PCA"
python -m src.pipeline.rain_leakage
cd src/pipeline/models && python run_all.py --features base

```

## Bases de datos e ingesta

`inegi.py` (geometría + población + elevación + ríos), chirps.py (precipitación, zonal sobre polígono), nasa_power.py (meteorología, centroide real), openaq.py (PM2.5, point-in-polygon, solo Aguascalientes), maiac_aod.py (AOD 1 km, zonal sobre polígono, 11 municipios), cenapred.py (ciclones, estática).

Cada script produce un parquet en data/processed/ con: date (o nada si es estática), municipio (nombre exacto igual a las llaves de MUNICIPIOS_AGS), source, year y sus columnas de valor. municipio es el campo de join universal entre fuentes.

`config/settings.py` define: BBOX_AGS, MUNICIPIOS_AGS (centroides fijos), INEGI_MUNICIPIOS_AGS (claves AGEM), FECHA_INICIO/FECHA_FIN (ingesta diaria), FECHA_INICIO_MASTER/FECHA_FIN_MASTER (tabla final) y rutas RAW/PROCESSED/MASTER_DIR.

Fuente	Contenido	Credenciales
NASA POWER	T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, PS, radiación, precipitación (~50 km)	No
CHIRPS v2.0	Precipitación diaria (~5 km)	No
MAIAC (AOD 0.55 µm)	Profundidad óptica de aerosoles	GEE_PROJECT (Google Earth Engine)
INEGI	Geometría, población, elevación, ríos	INEGI_TOKEN (opcional)
CENAPRED	Ciclones históricos	No
OpenAQ v3	PM2.5 (solo capital; excluido del master)	OPENAQ_API_KEY

# Limpieza y tabla maestra

`clean.py` lee cada parquet de data/processed/ y escribe una versión validada en data/master/, sin modificar el origen. Cada fuente tiene su limpiador, registrado en FUENTES_DINAMICAS (llave date+municipio) o FUENTES_ESTATICAS (llave municipio). Todos aplican los mismos pasos:

Normalizar el nombre del municipio contra el catálogo MUNICIPIOS_AGS.
Convertir sentinelas a NaN y eliminar duplicados (XGBoost maneja NaN de forma nativa).
MAIAC valida aod_055_count, aod_055_mean y aod_qa_best_pct, marca celdas < 10 px y conserva qc_aod_imputed como indicador de calidad.
Si cualquier fuente falla, el pipeline se detiene con error para no construir un master incompleto.

`build_master_table.py` construye un grid canónico municipio × día (11 × 2,404 = 26,444 filas) y hace left join de las fuentes dinámicas por date+municipio, con validate="many_to_one". La ausencia de cualquier fuente es un error.

Estáticas conservadas en el master: elevación, distancia al río y centroide. Población, viviendas, área y ciclones quedan en dim_municipios.parquet, porque con 11 municipios cada estática funciona como identificador del municipio, y los conteos históricos de ciclones incluyen años posteriores a cada fila (fuga).
Features derivadas (solo información hasta el día t): lags de 1, 2, 3 y 7 días, medias móviles de 7 y 30 días, lluvia acumulada de 3 a 90 días, días sin lluvia, rango térmico, déficit de presión de vapor, AOD móvil de 7 días y estacionalidad cíclica.
Los targets no se crean aquí porque dependen del horizonte; year se conserva solo para el walk-forward.
Tabla con PCA

`build_master_pca.py` agrega componentes principales al master y produce master_table_pca.parquet, un manifest y los objetos congelados (pca_bloques.joblib) en data/master/pca/<version>/.

Bloques (sin AOD; el script verifica que cada columna esté en exactamente un bloque): temp, lluvia, atmos. La lluvia y sus derivadas pasan por log1p.
Transformación: se resta una climatología suavizada (ventana circular de 15 días) por municipio y día del año, se estandariza con z-score y se aplica SVD por bloque.
Columnas nuevas: temp_PC1…, lluvia_PC1…, atmos_PC1…, con el número de componentes que alcanza 80 %, 90 % o 95 % de varianza acumulada (--max-level, default 0.95).
Representaciones del manifest: orig65, solo_pcs_<nivel> y orig+pcs_<nivel>.
Verificación: al terminar, recarga los objetos congelados y comprueba que los PCs guardados coinciden con la transformación.

Fuga en backtests. Por defecto el PCA se ajusta con todos los años, así que master_table_pca.parquet filtra información del futuro a cualquier backtest. Para evaluar con walk-forward genera un corte único anterior al primer fold:

python -m src.pipeline.build_master_pca --fit-until 2022-12-31

Esto produce master_table_pca_fit20221231.parquet. Para usarlo en models/, agrega esa ruta a TABLES en model_config.py. Para PCs reajustados por fold, usa src/data_analysis/pca_walkforward.py.

## Hallazgos clave del Análisis Exploratorio (EDA)

Basado en el análisis de $26,444$ filas y $75$ columnas guardadas en la tabla maestra:

Zero-Inflation en Lluvia: El 70% de las observaciones de lluvia son exactamente $0\text{ mm}$ (mediana $0\text{ mm}$, media $1.39\text{ mm}$, asimetría/skewness $4.82$). Esto justifica separar el modelado en probabilidad de ocurrencia (clasificación) antes que intentar predecir montos continuos.
Faltantes en AOD (MAIAC): Presenta un 20.6% de datos faltantes a nivel anual, concentrados estacionalmente debido a la nubosidad (41% de faltantes en septiembre vs. 5% en marzo).
Incertidumbre CHIRPS vs. NASA POWER: Coinciden en clasificar días secos/mojados en el 84.2% de los casos, con una correlación de Spearman $\rho = 0.49$.
Estabilidad Térmica: `T2M_MIN` muestra una distribución simétrica con media de $10.37^\circ\text{C}$ y mediana de $10.80^\circ\text{C}$, confirmándose como un objetivo de regresión estable.

## Conflicto de OpenAQ y MAIAC

clean.py aplicó a ambas fuentes nombres normalizados, sentinelas y fuera de rango a NaN, y duplicados eliminados. En MAIAC se verificó la coherencia entre aod_055_count, aod_055_mean y aod_qa_best_pct, y se marcaron celdas con menos de 10 píxeles.

OpenAQ quedó fuera del master porque cubre el 3.4 % de las celdas, todas de la capital.
Correlación de Spearman entre aod_055_mean y pm25_mean en la capital, solo con AOD observado (n = 896): ρ = −0.12, casi sin señal. Se descartó calibrar AOD a PM2.5.
La proporción imputada fue 0 %: no hay fuga por gap filling (qc_aod_imputed es constante y no debe entrar a X).
El AOD se mantiene en el master (79.4 % de cobertura), pero sus columnas no se usan como features del modelo evaluado.
Decisiones antes del modelo

## Auditoría temporal y resultados de rain_leakage.py

Reúne tres evaluaciones reproducibles:

leakage: reconstruye las features con build_master_table.add_features y las compara con las del master. Además altera los valores de las fuentes en fechas posteriores a 12 cortes aleatorios y comprueba que las features anteriores no cambien. Las 53 features evaluadas reproducen las del master y no usan información futura.
rain: la única habilidad útil es la probabilidad de ocurrencia a un día: BSS = 0.24 y AUC = 0.85. El monto de lluvia (dos etapas y Tweedie) no supera la climatología. Se congela el clasificador de ocurrencia a h=1 y no se sigue ajustando un modelo de monto.
scope: para T2M_MIN, el modelo global iguala o supera al ajuste por municipio; a h=1 obtiene skill = 0.19 frente al mejor baseline evaluado. T2M_MAX no muestra habilidad en ningún horizonte. A h=3 y h=7 ningún modelo supera el baseline suavizado.

Los CSV se escriben en src/pipeline/outputs_fase2/ (rain_eval.csv, scope_<variable>.csv). Los resultados acotan el uso operacional a temperatura mínima global y ocurrencia de lluvia, ambos a un día; no respaldan T2M_MAX, horizontes de 3 o 7 días ni predicción del monto de lluvia.

Verificar: el README original habla de 53 features auditadas, pero build_master_pca.py y los resúmenes de models/ cuentan 65 columnas base (sin AOD). Aclara qué 12 columnas separan ambas cifras (por ejemplo, features agregadas después de la auditoría) o actualiza el número.

53 features:Subconjunto de variables meteorológicas dinámicas estrictamente auditadas contra *data leakage* temporal hasta el día $t$.
65 features (`base`): Conjunto numérico completo de la tabla maestra (agrega métricas de radiación y estacionalidad amplia). Es la corrida recomendada.
72 features (`all`): Suma a la base las 7 variables estáticas (geometría, población, viviendas y declaratorias históricas de ciclones).

## Reglas de entrenamiento
Targets con groupby("municipio")[var].shift(-h), h ≥ 1, fuera del master.
X excluye date, municipio, year, los targets y qc_aod_imputed.
Walk-forward por año con ventana expansiva (folds 2023–2026; 2026 es año parcial).
Baselines: persistencia y climatología por municipio y día del año.
Corrida A (sin AOD) y corrida B (con AOD).
Resultados de la línea XGBoost
Primer test de XGBoost: el AOD no aporta de forma consistente y queda fuera (usar_aod=False). Los skills iniciales, medidos contra una climatología sin suavizar, estaban inflados.
Fase 2, leakage: 12 cortes aleatorios; las 53 features no usan información futura y reproducen el master.
Fase 2, lluvia (≥ 1 mm): solo hay habilidad en probabilidad de ocurrencia a h=1 (BSS 0.24, AUC 0.85, positivo en los 4 años). A h=3 y h=7 el BSS es negativo. Tweedie es peor que la climatología. El modelo de dos etapas gana en deviance solo en 2 de 4 años a h=1.
Fase 2, T2M_MIN: skill global contra el mejor baseline suavizado de 0.19 a h=1 (positivo en los 4 años), 0.03 a h=3 y −0.03 a h=7. El global supera al de por municipio en 9 de 12 combinaciones.
Fase 2, T2M_MAX: skill de 0.003, −0.04 y −0.045 a h=1, 3 y 7, con signo cambiante entre años. Se excluye del piloto.
Conclusión: el piloto defendible es T2M_MIN global a h=1 y probabilidad de lluvia a h=1, a escala municipal, no vecinal.
Comparación de modelos (src/pipeline/models)

Compara seis modelos de scikit-learn con el mismo protocolo walk-forward. Es una línea independiente de XGBoost: usa imputación por mediana y estandarización dentro de cada fold, y no depende de NaN nativos.

Tarea	Target	Modelos
Clasificación	Lluvia al día siguiente (precipitation_mm ≥ 1 mm, RAIN_THRESHOLD_MM)	Naive Bayes, Neural Network, SVM
Regresión	T2M_MIN al día siguiente	Linear Regression, Neural Network Regressor, Support Vector Regression
Archivos
Archivo	Función
model_config.py	Rutas, tablas (master, pca), feature sets, horizonte (HORIZON=1), folds (FOLD_YEARS), RANDOM_STATE=42, columnas de lluvia y temperatura
dataset.py	build_dataset: carga la tabla, une estáticas según el feature set, crea el target con shift(-h) y devuelve X, y, año y persistencia; feature_group clasifica cada columna
runner.py	ModelSpec (nombre, tarea, parámetros, complejidad, interpretabilidad, despliegue) y run_model: walk-forward, métricas, importancia por permutación, gráficos y *_summary.json
evaluation.py	Métricas de clasificación (accuracy, precision, recall, F1, ROC AUC) y regresión (MAE, MSE, RMSE, R²)
plots.py	Matriz de confusión, ROC, overlay de ROC, predicho vs real, importancia y heatmap de importancia
report.py	Lee los *_summary.json y genera machine_learning_challenge.md y las matrices de importancia
run_all.py	Corre los seis modelos y el reporte
naive_bayes.py, svm_classifier.py, mlp_classifier.py, linear_regression.py, svr.py, mlp_regressor.py	Un SPEC por modelo
Feature sets (--features)
Set	Contenido	Estado
base	Todas las columnas numéricas de la tabla, sin llaves, sin targets, sin AOD y sin columnas constantes	Recomendado
aod	base + columnas AOD	Solo para comparar
static	base + geom_area_km2, pob, viv (desde dim_municipios.parquet)	Exploratorio: identifican municipios
all	static + 4 columnas de declaratorias por ciclones; sin AOD	Exploratorio: los conteos de ciclones incluyen años posteriores a cada fila (fuga)

Una columna con un solo valor en la tabla se descarta, y un assert falla si se cuela una columna AOD fuera del set aod. Con --table pca, las features son las 65 originales más todos los PCs guardados en el parquet, lo que añade redundancia fuerte y hereda la fuga del PCA ajustado con todos los años (ver Tabla con PCA).

# Protocolo
Validación: walk-forward por año con ventana expansiva (folds 2023–2026); cada año de prueba se predice con un modelo entrenado en los años anteriores.
Pipeline por fold: SimpleImputer(mediana) → StandardScaler → modelo, ajustado solo con filas de entrenamiento.
Sin ajuste de hiperparámetros: los parámetros están fijos en cada script para no usar los folds de prueba en la selección.
Importancia: permutación sobre el último fold (2026, año parcial), muestra de 1,000 filas y 3 repeticiones. Mide el aumento de MAE o la caída de ROC AUC.
Baseline de regresión: persistencia (T2M_MIN de hoy). Skill = 1 − MAE_modelo / MAE_persistencia.
Clasificación: umbral fijo de 0.5. Las métricas de ranking (ROC AUC) usan predict_proba o, en SVM, decision_function; SVM no entrega probabilidades.
Parámetros
Modelo	Parámetros	Complejidad	Interpretabilidad	Despliegue
Naive Bayes	var_smoothing=1e-9	Low	High	Low
Neural Network	capas (64, 32), relu, alpha=1e-4, early_stopping=True, max_iter=300, random_state=42	High	Low	Medium
SVM	rbf, C=1.0, gamma=scale, class_weight=balanced	Medium	Low	Medium
Linear Regression	fit_intercept=True	Low	High	Low
Neural Network Regressor	igual que la red de clasificación; target estandarizado	High	Low	Medium
Support Vector Regression	rbf, C=1.0, epsilon=0.1, gamma=scale; target estandarizado	Medium	Low	Medium
Regla de selección

El mejor modelo es el de mayor ROC AUC (clasificación) o menor MAE (regresión). Si varios quedan a menos de 2 % del mejor, gana el de menor complejidad. Umbrales del veredicto:

Criterio	Clasificación	Regresión
Útil	ROC AUC ≥ 0.70	MAE por debajo de persistencia
Fuerte	ROC AUC ≥ 0.80	≥ 10 % de mejora en MAE
Confiable	Todos los folds cumplen "útil"	Todos los folds cumplen "útil"

Veredicto: READY FOR MODEL VALIDATION (todo cumple), CONDITIONALLY READY (alguna tarea es útil) o ADDITIONAL DATA REQUIRED.

Ejecución
cd src/pipeline/models
python run_all.py --table master --features base     # corrida recomendada
python run_all.py --table master --features all      # exploratoria
python naive_bayes.py --features base                # un solo modelo
python report.py --features base                     # regenera solo el reporte
Salidas

Se guardan en src/pipeline/models/outputs/[<tabla>/][<features>/]: la tabla master con base va en outputs/, master con all en outputs/all/ y pca con all en outputs/pca/all/.

# Archivo	Contenido
<slug>_summary.json	Parámetros, lista de features, métricas agregadas, baseline y tiempos
<slug>_folds.csv	Métricas por fold
<slug>_predictions.parquet	y_true, y_pred, y_score (y persistencia en regresión)
<slug>_importance.csv / .png	Importancia por permutación
clf_*_confusion_matrix.png, clf_*_roc.png, clf_roc_overlay.png	Gráficos de clasificación
reg_*_pred_vs_actual.png	Predicho vs real
classification_feature_importance_matrix.csv, regression_feature_importance_matrix.csv	Importancia de todas las features por modelo
*_importance_heatmap.png	Heatmap de las 20 features más importantes
machine_learning_challenge.md	Reporte completo (inventario de features, evaluación, comparación, selección, insights, resumen)

El reporte solo incluye los modelos que tengan *_summary.json en la carpeta de esa corrida; si falta un modelo, revisa que su script terminó sin error.

# Resultados de la corrida exploratoria (master, all, 72 features, 26,433 filas)

Clasificación (lluvia ≥ 1 mm a h=1; lluvia en 18.0 % de los días de prueba):

Modelo	Accuracy	Precision	Recall	F1	ROC AUC	CV de AUC por fold
Naive Bayes	0.768	0.417	0.721	0.528	0.808	5.3 %
Neural Network	0.835	0.545	0.522	0.533	0.792	5.5 %
SVM (seleccionado)	0.820	0.500	0.701	0.584	0.845	3.5 %

Regresión (T2M_MIN a h=1; MAE de persistencia 1.322):

Modelo	MAE	RMSE	R²	vs persistencia	Tiempo de ajuste
Linear Regression (seleccionado)	0.986	1.269	0.884	+25 %	1.2 s
Support Vector Regression	1.133	1.454	0.847	+14 %	235.8 s
Neural Network Regressor	1.336	1.712	0.789	−1 % (no supera)	89.4 s

Lectura: el modelo lineal gana en regresión, así que la señal de T2M_MIN a un día es en su mayoría lineal (persistencia y estacionalidad) y el modelo complejo no aporta. En lluvia, el SVM tiene el mejor ranking, pero no es comparable con el BSS de 0.24 de rain_leakage.py: aquí no hay baseline de climatología ni probabilidades calibradas.

# Cómo analizar los resultados

Orden recomendado de lectura del reporte: Feature Inventory (qué columnas entraron) → Model Evaluation Matrix → Stability (CV y peor fold) → Feature Evaluation → Limitations. Qué revisar en cada parte:

Confirma el feature set antes de leer métricas. El encabezado del reporte y el resumen ejecutivo lo declaran. Una corrida all o static no es comparable con base.
No compares skills de baselines distintos. models/ mide contra persistencia (+25 % MAE); la Fase 2 mide contra el mejor baseline suavizado (skill 0.19). El README documenta que medir contra climatología sin suavizar infla el skill. Para decidir, usa la Fase 2.
Desconfía de importancias gigantes. En la regresión lineal, T2M_MIN, T2M_MAX y t2m_rango suman importancias de ~10⁵. Son dependientes exactas (t2m_rango = T2M_MAX − T2M_MIN), así que los coeficientes se cancelan y la permutación rompe esa cancelación. No significa que sean las variables más útiles; compara solo las importancias de las redes y del SVR, o quita t2m_rango.
Una estática en el top es mala señal. viv y pob aparecen entre las 10 principales de la regresión lineal. Identifican municipios, no explican el clima del día siguiente.
Lee el peor fold, no solo el promedio. El peor fold de los tres regresores es 2024; 2026 es año parcial y su prueba es más corta.
Con 4 folds no hay intervalos creíbles. Diferencias menores a ~2 % entre modelos no son concluyentes.
Importancia ≤ 0. Las features con importancia media ≤ 0 (21 de 72 en clasificación) son ruido o redundantes; sirven para podar.
Recall y precision en lluvia. Con 18 % de días lluviosos, accuracy engaña. El umbral de 0.5 no está optimizado para el costo de falsas alarmas o lluvias perdidas.
Cruza resultados entre carpetas. Compara outputs/ (base) contra outputs/all/ y outputs/pca/... para medir si más columnas realmente mejoran.
Monitoreo

La Fase 3 (monitor_rain.py) congela los dos modelos del piloto (entrenados hasta [FECHA_CORTE]) y los monitorea con una ventana de [VENTANA] días. Calcula el skill móvil de 90 días contra el baseline congelado: si es negativo hoy y hace 90 días, el estado es RETIRAR; si solo es negativo hoy, ALERTA. También vigila el porcentaje de NaN en las columnas base. Para ensayarlo hace falta un corte anterior a mediados de 2025 y al menos 180 días posteriores.

Completa [FECHA_CORTE] y [VENTANA] con los valores reales del script.

## Pendientes y límites conocidos
Definir quién re-entrena, monitorea y retira el modelo.
Falta probar que la construcción de targets no tenga fuga; la prueba actual solo cubre features.
La calibración de las probabilidades de lluvia no está verificada (y el SVM no entrega probabilidades).
Con 4 folds, los intervalos de confianza no son creíbles; 2026 es año parcial.
Los modelos de models/ no tienen ajuste de hiperparámetros; el early stopping de las redes usa una validación aleatoria dentro de los años de entrenamiento, no temporal.
Los 11 municipios comparten sistemas meteorológicos: las filas del mismo día no son independientes.
Reconciliar el conteo de features (53 auditadas vs 65 en models/).
Falta una regla para detectar cambios en NASA POWER, CHIRPS o MAIAC (revisiones históricas, flags de QA).
Revisar los términos de uso de INEGI, CENAPRED y OpenAQ.
Datos, licencias y atribución

## Data sources

Este proyecto usa datos ambientales, meteorológicos, geoespaciales y de calidad del aire de acceso público para desarrollar un pipeline reproducible de análisis de riesgo ambiental para los 11 municipios de Aguascalientes, México.

Source	Data used	Official documentation
NASA POWER	Near-surface temperature, relative humidity, wind speed, pressure, solar radiation, and precipitation variables	https://power.larc.nasa.gov/docs/
CHIRPS v2.0	Daily precipitation estimates	https://www.chc.ucsb.edu/data/chirps
NASA MAIAC / MODIS	Aerosol Optical Depth (AOD), approximately 0.55 µm	https://developers.google.com/earth-engine/datasets
INEGI	Municipal boundaries, population, elevation, and river-related geographic information	https://www.inegi.org.mx/
CENAPRED	Historical cyclone-related information, subject to the terms of the specific source dataset	https://www.gob.mx/cenapred
OpenAQ API v3	PM2.5 measurements for the municipality of Aguascalientes; excluded from the master dataset in the current pipeline	https://docs.openaq.org/
Licensing and attribution policy

The project distinguishes between the software developed for this repository and the third-party datasets accessed or processed by its scripts.

NASA POWER: data are acknowledged using the provider's recommended reference and citation guidance. The version and access date should be recorded for each dataset used.
CHIRPS: publicly available for use and redistribution under the applicable terms of the product. Appropriate source attribution and scientific references should be retained.
MAIAC / MODIS: the licensing and redistribution conditions of the exact satellite product must be checked independently of the Google Earth Engine service terms. This repository does not claim ownership of the original satellite data.
INEGI: information is attributed to INEGI in accordance with its official terms of free use. Source metadata are preserved where applicable, and modifications and derived analyses are identified as work performed by this project.
CENAPRED: the terms of the specific dataset are reviewed before redistributing source files. Public accessibility alone is not treated as proof of unrestricted redistribution rights.
OpenAQ: the applicable license and attribution requirements of both OpenAQ and the original data provider are checked before redistributing measurements or derived datasets.
Data redistribution

This repository prioritizes reproducibility through documented extraction and processing scripts rather than unrestricted redistribution of third-party data. Unless a dataset's applicable terms have been verified, downloaded source files and derived data files (including the master table and the contents of models/outputs/) are not included in the public repository. Users should obtain the data from the official providers and comply with the applicable terms of use.

# API credentials

Credentials are used only to access the relevant services and must not be committed to version control. Environment variables that may be required: GEE_PROJECT, INEGI_TOKEN (optional, depending on the access method) and OPENAQ_API_KEY. Secret values, personal access tokens, and credential files must remain outside the public repository.

# Disclaimer

This project is an independent research and educational initiative. It is not affiliated with, endorsed by, or officially supported by NASA, Google, the Climate Hazards Center, INEGI, CENAPRED, or OpenAQ.

Environmental datasets may contain measurement uncertainties, estimation errors, missing observations, and changes in methodology or availability. Results generated by this project are independent analyses and should not be interpreted as official assessments or warnings issued by the data providers. Users are responsible for verifying the applicable data licenses and terms before redistributing datasets, publishing derived products, or using the results for operational or commercial purposes.