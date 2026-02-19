# grid_search_fai_parallel_EM.py
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_cleaning_utils import ensure_dir
from calculate_fai_utils import (
    fai_from_df,
    anticipation_fai_analysis,
    associate_fai_to_flare_dataframes
)
import multiprocessing

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

# Rango de ΔEM a probar
fai_em_values =  np.round(np.arange(0.001, 0.5+0.001, 0.001), 3)  # 0.014, 0.015, 0.016

# Rutas absolutas
input_dir = "/scratch/leuven/386/vsc38664/Thesis/26-02-17_Data/Data_for_16752_days"
csv_path_full = os.path.join(input_dir, f"df_full_{new_n}_valid.csv")
csv_path_flares = os.path.join(input_dir, f"df_flares_{new_n}_ABCMX.csv")
analysis_dir = os.path.join(input_dir, "Analysis_FAI")
ensure_dir(analysis_dir)

# Verificar archivos
if not os.path.isfile(csv_path_full):
    raise FileNotFoundError(f"No se encontró el archivo: {csv_path_full}")
if not os.path.isfile(csv_path_flares):
    raise FileNotFoundError(f"No se encontró el archivo: {csv_path_flares}")

# ===============================
# 2. LEER DATOS
# ===============================
df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)
for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# ===============================
# 3. FUNCION PARA UNA CORRIDA
# ===============================
def run_single_em(fai_em_jump):
    threshold_str = str(fai_em_jump).replace(".", "d")
    
    analysis_esp = os.path.join(
        analysis_dir,
        f"Analysis_FAI_T{fai_temp_range[0]}-{fai_temp_range[1]}_dEM{threshold_str}_dur{FAI_duration}min"
    )
    ensure_dir(analysis_esp)
    print(f"Test with Delta EM: {fai_em_jump:.3f}")
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
    detection_dict = {}       # % detectado en peak
    detection_start_dict = {} # % detectado antes del peak

    for cl in flare_classes:
        df_class = df_anticipation_time[df_anticipation_time['ClassLetter'] == cl]

        # Porcentaje detectado en peak
        pct_peak = df_class['FAIalerts_WPeak'].gt(0).sum() / len(df_class) * 100 if len(df_class) > 0 else 0.0
        detection_dict[cl] = round(pct_peak, 1)

        # Porcentaje detectado antes del peak
        pct_start = df_class['FAIalerts_WStart'].gt(0).sum() / len(df_class) * 100 if len(df_class) > 0 else 0.0
        detection_start_dict[cl] = round(pct_start, 1)

    # --- Guardar CSV resumen de esta combinación ---
    summary_dict = {
        "Configuration": f"T in [{fai_temp_range[0]},{fai_temp_range[1]}], duration={FAI_duration}min",
        "ΔEM": fai_em_jump,
        "False Alarm (%)": round(FP, 1),
        **{f"{cl}_Peak": detection_dict[cl] for cl in flare_classes},
        **{f"{cl}_Start": detection_start_dict[cl] for cl in flare_classes}
    }

    df_summary = pd.DataFrame([summary_dict])
    csv_summary_path = os.path.join(analysis_esp, f"FAI_summary_{method}.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    txt_summary_path = os.path.join(analysis_esp, f"FAI_summary_{method}.txt")
    df_summary.to_csv(txt_summary_path, index=False, sep='\t')

    print(f"✅ Guardado resumen EM={fai_em_jump} en: {csv_summary_path}")
    return summary_dict

# ===============================
# 4. EJECUTAR GRID SEARCH PARALELO
# ===============================
if __name__ == "__main__":
    multiprocessing.freeze_support()  # compatibilidad con SLURM / Windows

    all_results = []
    max_workers = 72  # Debe coincidir con --ntasks-per-node en SLURM
    
    print("start")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_em = {executor.submit(run_single_em, EM): EM for EM in fai_em_values}
        for future in as_completed(future_to_em):
            em_val = future_to_em[future]
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                print(f"❌ Error en EM={em_val}: {e}")

    # ===============================
    # 5. GUARDAR CSV GLOBAL
    # ===============================
    df_global = pd.DataFrame(all_results)
    df_global_path = os.path.join(analysis_dir, f"FAI_global_summary_{method}.csv")
    df_global.to_csv(df_global_path, index=False)
    print(f"\n✅ CSV global con todas las combinaciones guardado en: {df_global_path}")
