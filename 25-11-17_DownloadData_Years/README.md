Solar Data Downloader

Descarga automática de datos GOES, flares (NOAA/SWPC) y regiones activas (HEK) por años completos, 
con estructura organizada y soporte para ejecución vía Bash y screen en servidores Linux.

Estructura del repositorio
├── functions_download.py      # Funciones para descargar GOES, flares y AR
├── main_years_download.py     # Lógica por año: organiza carpetas y llama a las funciones
├── run_download.py            # Ejecuta la descarga desde Python
└── general_download.sh        # Script bash para ejecución automática

Cada año queda organizado así:

GOES_data/
└── YYYY/
    ├── logs/
    │   └── errores_YYYY.log
    ├── 01/
    │   ├── nc_files/     # Archivos GOES en .nc
    │   ├── plots/        # Figuras de diferencia con background
    │   ├── df_full_YYYY_01.csv      # Datos GOES corregidos + T/EM
    │   ├── df_flare_data_YYYY_01.csv
    │   └── df_AR_YYYY_01.csv
    ├── 02/
    │   └── ...
    ├── ...
    └── resumen_YYYY.txt
    
df_full → contiene datos GOES corregidos + temperatura + EM.

df_flare_data → flares GOES de ese mes.

df_AR → regiones activas HEK de ese mes (formato CSV).

Archivos del repositorio
1. functions_download.py
  Contiene toda la lógica:
    - Download_Data() → descarga GOES XRS
    - running_difference() → resta background y genera gráficos
    - Calculate_Tem_EM() → temperatura y emisión medida
    - build_full_dataframe() → dataframe completo día por día
    - get_flares() → descarga flares
    - get_active_regions() → descarga AR desde HEK

2. main_years_download.py
  Pipeline completo:
      Crea los días del año
      Itera por meses
      Procesa GOES + flares + AR
      Guarda CSVs mensuales
      Produce resumen anual
      Guarda logs mensuales y anuales
  Puedes correrlo directo: python3 main_years_download.py

3. run_download.py
  Descarga un solo año: python3 run_download.py 2022

4. general_download.sh
  Script bash robusto con:
    timeout por año
    watchdog de actividad
    logs generales
    ejecución continua
  Edita los años en:
    START_YEAR=2020
    END_YEAR=2025
  Ejecutar: bash general_download.sh
  O dentro de un screen: screen -S solar
                        bash general_download.sh
                        Ctrl+A D   # para salir del screen
          Reingresar:   screen -r solar


¿Qué hace exactamente el pipeline?

Por cada día:
    Descarga GOES XRS (1 min)
    Resta background y genera gráfico
    Calcula temperatura y EM (abundancia coronal y fotosférica)
    Une todo en df_full
    Descarga flares GOES/NOAA
    Descarga regiones activas HEK
    Guarda resultados en CSV por mes

Requisitos
    python >= 3.8
    sunpy
    pandas
    numpy
    matplotlib


Instalar: pip install sunpy pandas numpy matplotlib
                        
