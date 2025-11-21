# grid_search_fai_params_paralelizado.py
import os
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor
from data_cleaning_utils import ensure_dir
from calculate_fai_utils import (
    fai_from_df,
    anticipation_fai_analysis,
    associate_fai_to_flare_dataframes
)

# ===============================
# 1. PARÁMETROS GENERALES
# ===============================
n = 100
new_n = 59
fecha_actual = "2025-11-19"

window_minutes = 30
FAI_duration = 3
method = "filtered"

# ---------------------------
# Carpetas de análisis
# ---------------------------
analysis_dir = "Analysis_FAI"
ensure_dir(analysis_dir)

# ---------------------------
# Lectura de datos
# ---------------------------
input_dir = f"Data_for_{n}_days"
csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
csv_path_flares = f"{input_dir}/df_flares_{new_n}_valid.csv"

df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)
for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# ===============================
# 2. GRID DE PARÁMETROS
# ===============================
temp_min_values = range(5, 7)   # [5, 6] ejemplo pequeño
temp_max_values = range(11, 13) # [11, 12]
em_threshold_values = [0.003, 0.004]

param_combinations = list(itertools.product(temp_min_values, temp_max_values, em_threshold_values))

# ===============================
# 3. FUNCION DE EJECUCIÓN PARA UNA COMBINACIÓN
# ===============================
def run_single_combination(params):
    Tmin, Tmax, EM = params
    threshold_str = str(EM).replace(".", "d")
    analysis_esp = os.path.join(
        analysis_dir,
        f"Analysis_FAI_T{Tmin}-{Tmax}_EM{threshold_str}_dur{FAI_duration}min"
    )
    ensure_dir(analysis_esp)

    # --- Cálculo FAI ---
    result = fai_from_df(
        df_full=df_full_valid,
        fai_temp_range=(Tmin, Tmax),
        fai_em_threshold=EM,
        date_column="date",
        duration=True,
        FAI_duration=FAI_duration,
        filter_flare_coincidence=True,
        df_flares=df_flares_valid,
        flare_peak_col="PeakTime",
        flare_end_col="EndTime",
        verbose=False,
        output_dir=analysis_esp
    )
    
    df_fai_selected = result["df_fai_filtered"]

    # --- Anticipación de flares ---
    _, FN = anticipation_fai_analysis(
        df_fai_selected=df_fai_selected,
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
        fai_temp_range=(Tmin, Tmax),
        fai_em_threshold_d=threshold_str,
        FAI_duration=FAI_duration
    )

    # --- Asociación FAIs a flares ---
    _, FP = associate_fai_to_flare_dataframes(
        df_fai_selected=df_fai_selected,
        df_flares=df_flares_valid,
        window_minutes=window_minutes,
        include_inside=True,
        save_csv=True,
        save_summary=True,
        output_dir=analysis_esp,
        method=method,
        fai_temp_range=(Tmin, Tmax),
        fai_em_threshold_d=threshold_str,
        FAI_duration=FAI_duration
    )

    return {"Tmin": Tmin, "Tmax": Tmax, "EM": EM, "FN": FN, "FP": FP, "score": FN+FP}

# ===============================
# 4. EJECUCIÓN EN PARALELO
# ===============================
results = []
with ProcessPoolExecutor() as executor:
    for r in executor.map(run_single_combination, param_combinations):
        results.append(r)

# ===============================
# 5. GUARDAR TABLA FINAL
# ===============================
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(["score", "FN", "FP"])  # Orden por mejor combinación
df_results_path = os.path.join(analysis_dir, "grid_search_fai_results.csv")
df_results.to_csv(df_results_path, index=False)

print(f"\n✅ Grid search completado. Resultados guardados en: {df_results_path}")
print("Top combinaciones:")
print(df_results.head(10))
