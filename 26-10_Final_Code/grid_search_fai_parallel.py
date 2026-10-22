# grid_search_fai_parallel.py

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from calculate_fai_utils import (
    fai_from_df,
    anticipation_fai_analysis,
    associate_fai_to_flare_dataframes
)
from data_cleaning_utils import ensure_dir

# --------------------------
# 1. Parámetros generales
# --------------------------
n = 16752
new_n = 8164
window_minutes = 30
FAI_duration = 4
method = "filteredend_nooverlap"

# Carpeta de datos en el servidor (SCRATCH)
VSC_DATA = "/scratch/leuven/386/vsc38664/Thesis/26-02-17_Data"
os.chdir(VSC_DATA)

input_dir = f"Data_for_{n}_days"

csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
csv_path_flares = f"{input_dir}/df_flares_{new_n}_ABCMX.csv"

print("Cargando datos...")

df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)

for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# Carpeta global de resultados
analysis_dir = os.path.join(input_dir, "Analysis_FAI_GridSearch")
ensure_dir(analysis_dir)

# --------------------------
# 2. Valores a iterar
# --------------------------
fai_em_values = [0.014, 0.015, 0.016]

# --------------------------
# 3. Función que corre cada ΔEM
# --------------------------
def run_single_em(EM):

    Tmin, Tmax = 8, 35
    threshold_str = str(EM).replace(".", "d")

    print(f"Iniciando EM={EM}")

    # Carpeta específica de esta configuración
    analysis_esp = os.path.join(
        analysis_dir,
        f"Analysis_FAI_T{Tmin}-{Tmax}_EM{threshold_str}_dur{FAI_duration}min"
    )
    ensure_dir(analysis_esp)

    # --- Cálculo FAI ---
    result = fai_from_df(
        df_full=df_full_valid,
        fai_temp_range=(Tmin, Tmax),
        fai_em_jump=EM,
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

    df_fai_selected = result["df_fai_filteredend_nooverlap"]

    # --- Anticipación ---
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

    # --- Falsos positivos ---
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

    # --- Tabla resumen ---
    flare_classes = ['A', 'B', 'C', 'M', 'X']

    table_dict = {
        "Configuration": [f"T in [{Tmin},{Tmax}], duration={FAI_duration}min"],
        "ΔEM": [EM],
        "False Alarm": [FP]
    }

    for cl in flare_classes:
        df_class = df_flares_valid[df_flares_valid['ClassLetter'] == cl]
        if len(df_class) > 0:
            pct_detected = (
                df_class['FAIalerts_WPeak'].gt(0).sum()
                / len(df_class) * 100
            )
        else:
            pct_detected = 0.0

        table_dict[cl] = [round(pct_detected, 1)]

    df_summary = pd.DataFrame(table_dict)

    summary_path = os.path.join(
        analysis_dir,
        f"FAI_summary_{threshold_str}.csv"
    )

    df_summary.to_csv(summary_path, index=False)

    print(f"✔ EM={EM} terminado")

    return df_summary


# --------------------------
# 4. Paralelización
# --------------------------
if __name__ == "__main__":

    # Detectar CPUs asignadas por SLURM
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    print(f"\nUsando {n_workers} workers en paralelo\n")

    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for r in executor.map(run_single_em, fai_em_values):
            results.append(r)

    df_all = pd.concat(results, ignore_index=True)

    global_summary_path = os.path.join(
        analysis_dir,
        "FAI_gridsearch_all_results.csv"
    )

    df_all.to_csv(global_summary_path, index=False)

    print("\n✅ GRID SEARCH COMPLETADO")
    print(f"Resultados guardados en: {global_summary_path}")
