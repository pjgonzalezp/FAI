# grid_search_fai_em.py
import os
import pandas as pd
import numpy as np
from data_cleaning_utils import ensure_dir
from calculate_fai_utils import (
    fai_from_df,
    anticipation_fai_analysis,
    associate_fai_to_flare_dataframes
)

# ===============================
# 1. PARÁMETROS GENERALES
# ===============================
n = 16752
new_n = 8164
fecha_actual = "2026-02-17"

window_minutes = 30
FAI_duration = 4
method = "filteredend_nooverlap"

# Rango de temperatura fijo 
fai_temp_range = (8, 35)

# Carpetas de análisis
analysis_dir = f"Data_for_{n}_days/Analysis_FAI"
ensure_dir(analysis_dir)

# Lectura de datos
input_dir = f"Data_for_{n}_days"
csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
csv_path_flares = f"{input_dir}/df_flares_{new_n}_ABCMX.csv"

df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)
for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# ===============================
# 2. RANGO DE EM JUMPS
# ===============================
fai_em_values = np.arange(0.014, 0.017, 0.001)  # 0.014, 0.015, 0.016

# ===============================
# 3. FUNCION PARA UNA CORRIDA
# ===============================
def run_single_em(fai_em_jump):
    threshold_str = str(fai_em_jump).replace(".", "d")
    
    # Carpeta de resultados
    analysis_esp = os.path.join(
        analysis_dir,
        f"Analysis_FAI_T{fai_temp_range[0]}-{fai_temp_range[1]}_dEM{threshold_str}_dur{FAI_duration}min"
    )
    ensure_dir(analysis_esp)

    # --- Cálculo FAI ---
    result = fai_from_df(
        df_full=df_full_valid,
        fai_temp_range=fai_temp_range,
        fai_em_jump=fai_em_jump,
        date_column="date",
        duration=True,
        FAI_duration=FAI_duration,
        exclude_last_minutes=20,
        filter_flare_coincidence=True,
        df_flares=df_flares_valid,
        flare_peak_col="PeakTime",
        flare_end_col="EndTime",
        nooverlap=True,
        min_gap_minutes_overlap=20,
        verbose=False,
        output_dir=analysis_esp
    )

    df_fai_selected = result["df_fai_filteredend_nooverlap"]

    # --- Anticipación de flares ---
    df_anticipation_time, FN = anticipation_fai_analysis(
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
        fai_temp_range=fai_temp_range,
        fai_em_threshold_d=threshold_str,
        FAI_duration=FAI_duration
    )

    # --- Asociación FAIs a flares ---
    df_fai_assoc, FP = associate_fai_to_flare_dataframes(
        df_fai_selected=df_fai_selected,
        df_flares=df_flares_valid,
        window_minutes=window_minutes,
        include_inside=True,
        save_csv=True,
        save_summary=True,
        output_dir=analysis_esp,
        method=method,
        fai_temp_range=fai_temp_range,
        fai_em_threshold_d=threshold_str,
        FAI_duration=FAI_duration
    )

    # --- Calcular detección por clase ---
    flare_classes = ['A', 'B', 'C', 'M', 'X']
    detection_dict = {}
    for cl in flare_classes:
        df_class = df_anticipation_time[df_anticipation_time['ClassLetter'] == cl]
        if len(df_class) > 0:
            pct_detected = df_class['FAIalerts_WPeak'].gt(0).sum() / len(df_class) * 100
        else:
            pct_detected = 0.0
        detection_dict[cl] = round(pct_detected, 1)

    # Guardar CSV resumen de esta combinación
    summary_dict = {
        "Configuration": f"T in [{fai_temp_range[0]},{fai_temp_range[1]}], duration={FAI_duration}min",
        "ΔEM": fai_em_jump,
        "False Alarm (%)": round(FP,1),
        **detection_dict
    }
    df_summary = pd.DataFrame([summary_dict])

    csv_summary_path = os.path.join(analysis_esp, f"FAI_summary_{method}.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    
    txt_summary_path = os.path.join(analysis_esp, f"FAI_summary_{method}.txt")
    df_summary.to_csv(txt_summary_path, index=False, sep='\t')
    
    print(f"✅ Guardado resumen EM={fai_em_jump} en: {csv_summary_path}")

    return summary_dict

# ===============================
# 4. EJECUTAR GRID SEARCH
# ===============================
all_results = []

for EM in fai_em_values:
    res = run_single_em(EM)
    all_results.append(res)

# ===============================
# 5. GUARDAR CSV GLOBAL
# ===============================
df_global = pd.DataFrame(all_results)
df_global_path = os.path.join(analysis_dir, f"FAI_global_summary_{method}.csv")
df_global.to_csv(df_global_path, index=False)
print(f"\n✅ CSV global con todas las combinaciones guardado en: {df_global_path}")
