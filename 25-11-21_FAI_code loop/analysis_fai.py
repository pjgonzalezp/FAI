# analysis_fai.py
import os
import pandas as pd
from data_cleaning_utils import (ensure_dir)
from calculate_fai_utils import (fai_from_df,
                                 anticipation_fai_analysis,
                                 associate_fai_to_flare_dataframes)

# ============================================================
# 1. PARÁMETROS DEL ANÁLISIS
# ============================================================
n=100
new_n = 59 # nuevo número de días
fecha_actual = "2025-11-19"
# Parámetros de FAI
window_minutes=30
fai_temp_range = (9, 25)
fai_em_threshold = 0.006
date_column = "date"
duration = True
FAI_duration = 3
filter_flare_coincidence = True
method = "filtered" # "all", "true" o "filtered"


# Formato de threshold para nombres de carpetas
threshold_str = f"{fai_em_threshold}"   # "0.005" # convertir a string sin notación científica
fai_em_threshold_d = threshold_str.replace(".", "d") # reemplazar el punto por 'd'

# ---------------------------
# Carpetas de análisis
# ---------------------------
analysis_dir = f"Analysis_FAI"
ensure_dir(path = analysis_dir) # Verificar folder

# Crear subcarpeta para guardar resultados de análisis
analysis_esp = os.path.join(analysis_dir, f"Analysis_FAI_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min")
# Verificar folder
ensure_dir(path = analysis_esp)

# ---------------------------
# Lectura de datos # Valid data:
# ---------------------------
input_dir = f"Data_for_{n}_days"

# Path of cleaned data in csv
csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
# Path of valid flares: 
csv_path_flares = f"{input_dir}/df_flares_{new_n}_valid.csv"

# Verificar existencia de archivos y avisar
if not os.path.exists(csv_path_full):
    print(f"⚠️ No se encontró el archivo GOES: {csv_path_full}")
else:
    print(f"✅ Archivo GOES encontrado: {csv_path_full}")

if not os.path.exists(csv_path_flares):
    print(f"⚠️ No se encontró el archivo de flares: {csv_path_flares}")
else:
    print(f"✅ Archivo de flares encontrado: {csv_path_flares}")

# Cargar CSV
df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)

# Asegurar datetime
for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# ---------------------------
# Cálculo de FAI
# ---------------------------
result = fai_from_df(df_full=df_full_valid,
                    fai_temp_range = fai_temp_range,
                    fai_em_threshold = fai_em_threshold,
                    date_column = "date",
                    duration = duration,
                    FAI_duration = FAI_duration,
                    filter_flare_coincidence = filter_flare_coincidence,
                    df_flares = df_flares_valid,
                    flare_peak_col = "PeakTime",
                    flare_end_col = "EndTime",
                    verbose = True,
                    output_dir=analysis_esp)

# Acceder a los resultados
df_fai_full = result["df_fai_full"]      # Todos los datos GOES con columnas de evaluación
df_fai_all = result["df_fai_all"]        # Todos los candidatos
df_fai_true = result["df_fai_true"]      # FAI con duración mínima
df_fai_filtered = result["df_fai_filtered"]  # FAI sin coincidencia con el final de flares

# all = todos los FAI segun críterios de EM y T
# true = todos los FAI segun críterios de EM, T y duración del FAI activado
# filtered = todos los FAI segun críterios de EM, T, duración del FAI activado
#            y que no están entre el peak y end de una fulguración
#method = "all"  # "all", "true" o "filtered"

method_mapping = {
    "all": ("df_fai_all", df_fai_all),
    "true": ("df_fai_true", df_fai_true),
    "filtered": ("df_fai_filtered", df_fai_filtered)
}

if method in method_mapping:
    df_name, df_fai_selected_calculate = method_mapping[method]
    print(f"Método elegido: {method} → {df_name}")
else:
    raise ValueError(f"Método '{method}' no reconocido. Use 'all', 'true' o 'filtered'")

# ---------------------------
# Anticipación de flares (CALCULA LOS TIEMPOS RELATIVOS y cuáles flares están siendo anticipados)
# ---------------------------
df_anticipation_time, false_negative_percentage = anticipation_fai_analysis(
    df_fai_selected=df_fai_selected_calculate,
    df_flare_data=df_flares_valid,
    start_col="StartTime",
    peak_col="PeakTime",
    end_col="EndTime",
    window_minutes=window_minutes,
    max_prev_flare_minutes=180,
    save_csv=True,
    save_summary=True,
    output_dir=analysis_esp,
    method=method,
    fai_temp_range=fai_temp_range,
    fai_em_threshold_d=fai_em_threshold_d,
    FAI_duration=FAI_duration )

df_fai_assoc, false_positive_percentage = associate_fai_to_flare_dataframes(
    df_fai_selected=df_fai_selected_calculate,
    df_flares=df_flares_valid,
    window_minutes=window_minutes,
    include_inside=True,
    save_csv=True,
    save_summary=True,
    output_dir=analysis_esp,
    method=method,
    fai_temp_range=fai_temp_range,
    fai_em_threshold_d=fai_em_threshold_d,
    FAI_duration=FAI_duration
)


print("\n🎯 ANÁLISIS COMPLETO")
print(f"Falsos negativos (flares sin FAI): {false_negative_percentage:.2f}%")
print(f"Falsos positivos (FAIs no asociados a flares): {false_positive_percentage:.2f}%")