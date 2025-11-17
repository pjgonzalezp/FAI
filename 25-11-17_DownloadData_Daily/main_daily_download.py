# main_daily_download.py

from functions_download import (
    Download_Data, running_difference, Calculate_Tem_EM,
    build_full_dataframe, get_flares, get_active_regions
)
import os
import pandas as pd

# -------------------------
# Configuración de fechas
# -------------------------
start_time = "1980-01-03 00:00:00"
end_time   = "1980-01-03 23:59:59"

# -------------------------
# Estructura de carpetas
# -------------------------
data_dir = "GOES_data"

raw_goes_dir       = os.path.join(data_dir, "raw", "GOES")
raw_flares_dir     = os.path.join(data_dir, "raw", "Flares")
raw_ar_dir         = os.path.join(data_dir, "raw", "AR")

processed_goes_dir = os.path.join(data_dir, "processed", "GOES_full")
processed_flares_dir = os.path.join(data_dir, "processed", "Flares_clean")
processed_ar_dir     = os.path.join(data_dir, "processed", "AR_clean")

plots_goes_dir     = os.path.join(data_dir, "plots", "GOES")
plots_flares_dir   = os.path.join(data_dir, "plots", "Flares")
plots_ar_dir       = os.path.join(data_dir, "plots", "AR")

# Crear todas las carpetas si no existen
for d in [raw_goes_dir, raw_flares_dir, raw_ar_dir,
          processed_goes_dir, processed_flares_dir, processed_ar_dir,
          plots_goes_dir, plots_flares_dir, plots_ar_dir]:
    os.makedirs(d, exist_ok=True)

# -------------------------
# 1️⃣ Descargar datos GOES
# -------------------------
print("1. Descargar datos GOES")
goes_ts, observatory = Download_Data(start_time, end_time, output_dir=raw_goes_dir)

if goes_ts is None:
    print(f"No hay datos GOES para {start_time} - {end_time}. Día saltado.\n")
    df_goes = pd.DataFrame()
else:
    print(f"Se encontraron datos GOES para {start_time} - {end_time}.")
    df_goes = goes_ts.to_dataframe()
    print(f"Número de registros: {len(df_goes)}")
    print(f"Columnas disponibles: {list(df_goes.columns)}")

    # -------------------------
    # 2️⃣ Restar Background
    # -------------------------
    print("2. Restar Background")
    goes_ts_corrected_diff = running_difference(goes_ts, Dif_time=5, plot=True,
                                                block_dir=plots_goes_dir, start_time=start_time)

    # -------------------------
    # 3️⃣ Calcular T y EM
    # -------------------------
    print("3. Calcular T y EM")
    temp_em_cor  = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='coronal')
    temp_em_phot = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='photospheric')

    # -------------------------
    # 4️⃣ Construir df_full
    # -------------------------
    print("4. Construir df_full")
    df_full = build_full_dataframe(goes_ts, goes_ts_corrected_diff,
                                   temp_em_cor, temp_em_phot,
                                   clip_negative=True, normalize_em=True)

    # -------------------------
    # 5️⃣ Añadir columna observatorio
    # -------------------------
    df_full["observatory"] = observatory if observatory else "Unknown"
    cols = ["observatory"] + [col for col in df_full.columns if col != "observatory"]
    df_full = df_full[cols]

# -------------------------
# 6️⃣ Descargar flares
# -------------------------
print(f"6. Descargando flares: {start_time} - {end_time}")
flare_data = get_flares(start_time, end_time, output_dir=raw_flares_dir)

df_flare_data = flare_data if flare_data is not None else pd.DataFrame()
if df_flare_data.empty:
    print(f"No se encontraron flares para {start_time} - {end_time}.")
else:
    print(f"Se encontraron {len(df_flare_data)} flares para {start_time} - {end_time}.")

# -------------------------
# 7️⃣ Descargar AR
# -------------------------
print(f"7. Descargando AR: {start_time} - {end_time}")
ar_data = get_active_regions(start_time, end_time, output_dir=raw_ar_dir)

df_ar_data = ar_data if ar_data is not None else pd.DataFrame()
if df_ar_data.empty:
    print(f"No se encontraron AR para {start_time} - {end_time}.")
else:
    print(f"Se encontraron {len(df_ar_data)} AR para {start_time} - {end_time}.")

# -------------------------
# Guardar resultados en CSV
# -------------------------
if not df_goes.empty:
    out_csv_goes = os.path.join(processed_goes_dir, f"GOES_full_dataframe_{start_time[:10]}.csv")
    df_full.to_csv(out_csv_goes, index=True)
    print(f"GOES full dataframe guardado en {out_csv_goes}")

if not df_flare_data.empty:
    out_csv_flares = os.path.join(processed_flares_dir, f"Flares_{start_time[:10]}.csv")
    df_flare_data.to_csv(out_csv_flares, index=False)
    print(f"Flares guardados en {out_csv_flares}")

if not df_ar_data.empty:
    out_csv_ar = os.path.join(processed_ar_dir, f"ActiveRegions_{start_time[:10]}.csv")
    df_ar_data.to_csv(out_csv_ar, index=False)
    print(f"Active Regions guardadas en {out_csv_ar}")

print("✅ Descarga y procesamiento completado para el día.")
