#################################################################################################################################################

    ######  #     #   #######   ######   #     #   #######      #######   #######   ######   #     #   #######   ######  #######  #######
    #       #     #   #         #    #   #     #      #         #     #   #     #   #    #   #     #   #         #          #     #     #
    #       #######   ###       #    #   #     #      #         #######   #####     #    #   #######   ###       #          #     #     #    
    #       #     #   #         #   ##   #     #      #         #         #   #     #    #      #      #         #          #     #     #   
    ######  #     #   #######   ######   #######   #######      #         #    #    ######      #      ######    ######     #     #######  
                                      # 
                                      
##################################################################################################################################################

                                #################################################
                                ############ Arquitectura de ingesta ############
                                #################################################

### BASES DE DATOS ###

`inegi.py` (geometría + población + elevación + ríos)
`chirps.py` (precipitación, zonal sobre polígono)
`nasa_power.py` (meteorología, centroide real)
`openaq.py` (PM2.5, point-in-polygon, solo Aguascalientes)
`maiac_aod.py` (AOD 1km, zonal sobre polígono, 11 municipios)
`cenapred.py` (ciclones, estática)

Para la igesta se necesita obtener la clave API de Open AQ y el id del proyecto de Google Earth Engine.
Cada script sigue produciendo un parquet en `data/processed/` con al menos `date` (o nada si es estática), `municipio` (nombre exacto igual a las llaves de `MUNICIPIOS_AGS`), `source`, `year`, más sus columnas de valor propias. `municipio` sigue siendo el campo de join universal entre fuentes.

`config/settings.py` con constantes: `BBOX_AGS`, `MUNICIPIOS_AGS` con centroides fijos, `INEGI_MUNICIPIOS_AGS` con claves AGEM, rangos de fechas `FECHA_INICIO`/`FECHA_FIN` para ingesta diaria y `FECHA_INICIO_MASTER`/`FECHA_FIN_MASTER` para la tabla final, rutas RAW/PROCESSED/MASTER_DIR. 


Se necesita obtener la clave API de Open AQ y el id del proyecto de Google Earth Engine

A esto se añadieron las constantes específicas de la nueva fuente satelital: `MAIAC_COLLECTION` (MCD19A2 V6.1), `MAIAC_QA_BITS_VALIDOS`, `MAIAC_RESOLUCION_M` (1000), `MAIAC_BANDAS` (`Optical_Depth_055`, `Optical_Depth_047`, `Column_WV`, `AOD_QA`) y `MAIAC_MAX_AOD` (6.0).

`src/utils/geo_utils.py` usa una capa de acceso geoespacial con la misma API: `municipios_iter()` itera los centroides fijos de settings, `nearest_municipio()` hace distancia euclidiana como fallback, y `load_municipios_geom()` carga el polígono real de INEGI desde `data/processed/inegi.parquet/geometria_municipios.parquet`. Esa pieza de geometría real es la que ahora habilita dos consumidores con zonal stats reales: `chirps.py` (rasterio.mask sobre el polígono), `nasa_power.py` obtiene centroide real vía `municipios_iter_geom()`y  `maiac.py`, que también hace `rasterio.mask` sobre el mismo polígono pero en proyección sinusoidal reproyectada. `openaq.py` sigue con point-in-polygon vía `municipio_from_point()` y fallback a `nearest_municipio`. `src/utils/io_utils.py` usa `save_parquet`/`append_parquet` con downcast de dtypes y partición `source`/`year` es la firma obligatoria para cualquier script de ingesta.


El problema estructural era que OpenAQ solo tiene 4 estaciones, todas en la zona urbana de la ciudad de Aguascalientes, por lo que 10 de los 11 municipios quedaban con `pm25_* = NaN` en el master. La solución implementada es una arquitectura de dos capas calibradas. La capa 1 es OpenAQ (medición directa, 1 municipio), que se conserva tal cual pero con dos features nuevas inyectadas en `clean.py` que no existían antes: `qc_pm25_observado` (booleano que indica si el dato viene de estación física) y `dist_monitor_km` (distancia del centroide municipal al monitor más cercano, unida como estática). OpenAQ pasa de ser "el dato" a ser el ground truth de calibración. La capa 2 es MODIS MAIAC AOD (proxy satelital, 11 municipios): el nuevo `maiac_aod.py` descarga MCD19A2 V6.1 vía NASA LAADS DAAC o Google Earth Engine, decodifica el bitmask de `AOD_QA` (bits 8-11 = calidad AOD, bits 0-2 = nube, bits 3-4 = nieve), reproyecta de sinusoidal a EPSG:6372, y hace zonal stats sobre `geometria_municipios.parquet`. El resultado es `aod_055_mean`, `aod_055_count` y `aod_qa_best_pct` por municipio y día. La cobertura observada real es de ~48%-67% por nubosidad (MNAR), y el script aplica gap-filling con dos niveles: interpolación temporal (rápida, `qc_aod_imputed=True`) o GAIN (red generativa, R=0.90 validado con AERONET, para cuando el ciclo lo permita).

El puente entre capas es la calibración AOD→PM2.5. En `build_master_table.py` se entrena un modelo de conversión `PM2.5 = f(AOD, T2M, RH2M, WS2M, PRECTOTCORR)` usando solo los datos de Aguascalientes capital (donde OpenAQ y MAIAC se solapan). Ese modelo calibrado se aplica a los 10 municipios restantes, generando `pm25_from_maiac`. La investigación en Kinshasa demostró que MLR mejora R² de 0.51→0.57 y XGBoost de 0.65→0.73 frente a AOD crudo, por lo que la calibración no es opcional. El master final lleva una columna `qc_pm25_fuente` con valores `"station"` (Aguascalientes) o `"satellite"` (resto), más `pm25_from_maiac`, `aod_055_mean` y `aod_qa_best_pct`. Esto le permite a XGBoost aprender a ponderar confianza y a SHAP distinguir entre importancia genuina de PM2.5 e importancia espuria de identidad municipal.

`clean.py` lee cada parquet de `data/processed/` vía `MAPEO_RUTAS` (verificar siempre contra el filesystem). A las fuentes existentes se añade `maiac_aod` en `FUENTES_DINAMICAS` con su limpiador `clean_maiac`, que aplica `flag_out_of_range` con `MAIAC_MAX_AOD=6.0` (C6.1 elevó el tope desde 4.0), preserva `qc_aod_imputed` como auditoría (no como feature), y reporta cobertura observada vs. post-imputación por separado. Los limpiadores existentes no cambian de firma. `build_master_table.py` sigue armando el grid canónico municipio×día con `FECHA_INICIO_MASTER`/`FECHA_FIN_MASTER`, hace left-join de dinámicas por `date+municipio` y estáticas por `municipio`, con los mismos asserts anti-duplicados. Se agrega un paso de calibración y transformación AOD→PM2.5 entre el join y el guardado final, y se registran las nuevas features `qc_pm25_fuente`, `dist_monitor_km` y `pm25_from_maiac`.

Cualquier fuente nueva sigue el contrato de tres pasos: (1) guardarse desde ingesta con `save_parquet` y partición `source,year`, (2) tener su limpiador en `clean.py` registrado en `FUENTES_DINAMICAS` o `FUENTES_ESTATICAS` con su ruta real en `MAPEO_RUTAS`, y (3) respetar `municipio` como llave de join contra el catálogo de `MUNICIPIOS_AGS`. MAIAC cumple los tres: parquet particionado en `data/processed/maiac_aod.parquet`, limpiador `clean_maiac` registrado, y nombres normalizados contra el catálogo. La cobertura PM2.5 del master no será 100% diaria: OpenAQ aporta el 9% de las celdas (solo Aguascalientes), MAIAC aporta ~48%-67% observado y ~80%-100% tras gap-filling, con incertidumbre mayor en los días imputados. Los 10 municipios sin monitor tendrán PM2.5 estimado con incertidumbre que debe reflejarse en `qc_pm25_fuente` y `aod_qa_best_pct`. Esto es una limitación de las fuentes físicas, no del pipeline, y debe quedar explícito en el README del master.


####### MASTER TABLE ######
aod_055_mean/count/qa_best_pct van crudos para los 11 municipios, y qc_pm25_observado marca las celdas con estación; excluye pm25_* de los features de XGBoost (~91% NaN y colineal con la identidad municipal).