# grid_search_fai_params.py
import os
import pandas as pd
from data_cleaning_utils import ensure_dir
from calculate_fai_utils import (
    fai_from_df,
    anticipation_fai_analysis,
    associate_fai_to_flare_dataframes
)

# ---------------------------
# Parámetros generales
# ---------------------------
n = 100
new_n = 59
fecha_actual = "2025-11-19"
window_minutes = 30
date_column = "date"
duration = True
FAI_duration = 3
filter_flare_coincidence = True
method = "filtered"

# Carpetas
analysis_dir = "Analysis_FAI"
ensure_dir(analysis_dir)

# ---------------------------
# Cargar datos
# ---------------------------
input_dir = f"Data_for_{n}_days"
csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
csv_path_flares = f"{input_dir}/df_flares_{new_n}_valid.csv"

df_full_valid = pd.read_csv(csv_path_full)
df_flares_valid = pd.read_csv(csv_path_flares)

# Asegurar datetime
for col in ["StartTime", "PeakTime", "EndTime"]:
    df_flares_valid[col] = pd.to_datetime(df_flares_valid[col])

# ---------------------------
# Valores a testear
# ---------------------------
temp_min_values = range(5, 7)    # Tmin: 5,6
temp_max_values = range(11, 13)  # Tmax: 11,12
em_threshold_values = [0.003, 0.004]

# DataFrame para guardar resultados
columns = ["Tmin", "Tmax", "EM", "False_Negative_%", "False_Positive_%", "Score"]
results = []

# ---------------------------
# Grid search
# ---------------------------
for Tmin in temp_min_values:
    for Tmax in temp_max_values:
        if Tmax <= Tmin:
            continue  # evitar Tmax < Tmin
        for EM in em_threshold_values:

            print(f"\n▶ Probando combinación: Tmin={Tmin}, Tmax={Tmax}, EM={EM}")

            # Formato de threshold para nombres
            EM_d = str(EM).replace(".", "d")
            fai_temp_range = (Tmin, Tmax)
            fai_em_threshold = EM

            # Crear subcarpeta para guardar resultados de este test
            analysis_esp = os.path.join(
                analysis_dir, f"Analysis_FAI_T{Tmin}-{Tmax}_EM{EM_d}_dur{FAI_duration}min"
            )
            ensure_dir(analysis_esp)

            # 1) Calcular FAI
            result = fai_from_df(
                df_full=df_full_valid,
                fai_temp_range=fai_temp_range,
                fai_em_threshold=fai_em_threshold,
                date_column=date_column,
                duration=duration,
                FAI_duration=FAI_duration,
                filter_flare_coincidence=filter_flare_coincidence,
                df_flares=df_flares_valid,
                flare_peak_col="PeakTime",
                flare_end_col="EndTime",
                verbose=False,
                output_dir=analysis_esp
            )

            df_fai_selected_calculate = result["df_fai_filtered"] if method=="filtered" else result[f"df_fai_{method}"]

            # 2) Calcular anticipación (falsos negativos)
            _, fn_percentage = anticipation_fai_analysis(
                df_fai_selected=df_fai_selected_calculate,
                df_flare_data=df_flares_valid,
                start_col="StartTime",
                peak_col="PeakTime",
                end_col="EndTime",
                window_minutes=window_minutes,
                max_prev_flare_minutes=180,
                save_csv=True,          # ✅ Guardar CSV en carpeta del test
                save_summary=True,      # ✅ Guardar TXT en carpeta del test
                output_dir=analysis_esp,
                method=method,
                fai_temp_range=fai_temp_range,
                fai_em_threshold_d=EM_d,
                FAI_duration=FAI_duration
            )

            # 3) Asociar FAIs a flares (falsos positivos)
            _, fp_percentage = associate_fai_to_flare_dataframes(
                df_fai_selected=df_fai_selected_calculate,
                df_flares=df_flares_valid,
                window_minutes=window_minutes,
                include_inside=True,
                save_csv=True,           # ✅ Guardar CSV en carpeta del test
                save_summary=True,       # ✅ Guardar TXT en carpeta del test
                output_dir=analysis_esp,
                method=method,
                fai_temp_range=fai_temp_range,
                fai_em_threshold_d=EM_d,
                FAI_duration=FAI_duration
            )

            # Score = FN + FP
            score = fn_percentage + fp_percentage

            results.append({
                "Tmin": Tmin,
                "Tmax": Tmax,
                "EM": EM,
                "False_Negative_%": fn_percentage,
                "False_Positive_%": fp_percentage,
                "Score": score
            })

# ---------------------------
# Guardar tabla resumen general
# ---------------------------
df_results = pd.DataFrame(results)
df_results.sort_values("Score", inplace=True)
df_results.reset_index(drop=True, inplace=True)

summary_file = os.path.join(analysis_dir, "grid_search_fai_results.csv")
df_results.to_csv(summary_file, index=False)
print(f"\n✅ Grid search completo. Resultados guardados en:\n{summary_file}")

# Mostrar las mejores combinaciones
print("\n🏆 Mejores combinaciones:")
print(df_results.head(10))
