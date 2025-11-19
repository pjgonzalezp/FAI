☀️ Solar Data Downloader

Descarga automática de datos GOES, flares (NOAA/SWPC) y regiones activas (HEK) por años completos, con estructura organizada y soporte para ejecución vía Bash y screen en servidores Linux.

📁 Estructura del repositorio
├── functions_download.py      # Funciones para descargar GOES, flares y AR
├── main_years_download.py     # Pipeline anual: organiza carpetas y llama a las funciones
├── run_download.py            # Ejecuta la descarga de un año específico
└── general_download.sh        # Script Bash para ejecución automática y robusta

🗂 Organización de los datos por año

Cada año queda organizado así:

GOES_data/
└── YYYY/
    ├── logs/
    │   └── errores_YYYY.log
    ├── 01/
    │   ├── nc_files/                   # Archivos GOES en formato .nc
    │   ├── plots/                      # Figuras de diferencia con background
    │   ├── df_full_YYYY_01.csv         # Datos GOES corregidos + T/EM
    │   ├── df_flare_data_YYYY_01.csv  # Flares GOES/NOAA
    │   └── df_AR_YYYY_01.csv           # Regiones activas HEK
    ├── 02/ ...
    └── resumen_YYYY.txt                # Resumen anual

📄 Descripción de los CSVs
Archivo	Contenido
df_full	Datos GOES corregidos + temperatura + emisión medida (T/EM)
df_flare_data	Flares GOES/NOAA del mes
df_AR	Regiones activas HEK del mes
resumen_YYYY.txt	Días procesados, días sin datos, total de flares y AR


⚙️ Archivos principales

functions_download.py
 Contiene toda la lógica de descarga y procesamiento:

       Función	              Descripción
    -Download_Data():	    Descarga GOES XRS
    -running_difference():	Resta background y genera gráficos
    -Calculate_Tem_EM():	Calcula temperatura y emisión medida
    -build_full_dataframe():Construye dataframe completo día por día
    -get_flares():	        Descarga flares GOES/NOAA
    -get_active_regions():	Descarga regiones activas desde HEK

main_years_download.py
    Pipeline anual completo:
    - Crea un dataframe con todos los días del año
    - Itera por meses y días
    - Procesa GOES + flares + AR
    - Guarda CSVs mensuales
    - Reconstruye archivos anuales (df_full_YYYY.csv, df_flare_data_YYYY.csv, df_AR_YYYY.csv)
    - Genera resumen anual (resumen_YYYY.txt)
    - Guarda logs mensuales y anuales
Ejecutar directamente:
    python3 main_years_download.py

-run_download.py
 Permite descargar un año específico:
    python3 run_download.py 2022

-general_download.sh
 Script Bash robusto para ejecución automática:
    Timeout por año (TIMEOUT_PER_YEAR)
    Watchdog de actividad (15 min sin cambios en output_YYYY.txt)
    Logs generales

Ejecución continua por años

Configurar años:

    START_YEAR=2020
    END_YEAR=2025

Ejecutar: bash general_download.sh

Con screen (recomendado en servidores):

screen -S solar 
    bash general_download.sh
# Salir del screen: Ctrl+A D
# Reingresar: screen -r solar

🔄 ¿Qué hace exactamente el pipeline?

Por cada día:
    - Descarga GOES XRS (resolución 1 min)
    - Resta background y genera gráfico de diferencia
    - Calcula temperatura y EM (abundancia coronal y fotosférica)
    - Combina todo en df_full
    - Descarga flares GOES/NOAA
    - Descarga regiones activas HEK
    - Guarda resultados en CSV por mes

Al final del año:
    - Reconstruye archivos anuales desde los CSVs mensuales
    - Genera un resumen con días procesados, días sin datos, total de flares y AR

🛠 Requisitos: Python >= 3.8

Bibliotecas: sunpy, pandas, numpy, matplotlib

Instalación rápida: pip install sunpy pandas numpy matplotlib
