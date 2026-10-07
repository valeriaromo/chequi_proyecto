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


### Siguiente fase: congelamiento y monitoreo

La fase siguiente es `fase3_produccion.py`: congelar el modelo global de `T2M_MIN` a h=1 y el clasificador de ocurrencia de lluvia a h=1, y monitorear el skill móvil de 90 días frente al baseline congelado. No es una nueva fase de entrenamiento o búsqueda de modelos: el artefacto y el baseline de referencia deben permanecer fijos durante el monitoreo para que las alertas sean comparables.

La regla de estado prevista es:

- **RETIRAR:** el skill es negativo hoy y también era negativo hace 90 días.
- **ALERTA:** el skill es negativo hoy, pero no se cumple la condición de retiro.
- Si el skill no es negativo hoy, no se activa ninguna de esas dos condiciones.

Para ensayar la regla con un corte histórico, la interfaz prevista es:

El ensayo histórico permite probar el flujo, pero no sustituye el monitoreo con datos posteriores ni valida los aspectos aún no evaluados. La regla de retiro solo es operativa si una persona responsable revisa el estado y puede actuar. Todavía falta nombrar quién tiene esa responsabilidad y autoridad; hasta entonces, el pipeline no debe considerarse un proceso de retiro autónomo ni una alerta atendida.
