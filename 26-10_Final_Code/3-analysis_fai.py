# analysis_fai.py
import os
import pandas as pd
from data_cleaning_utils import (ensure_dir)
from calculate_fai_utils import (fai_from_df,
                                 anticipation_fai_analysis,
                                 associate_fai_to_flare_dataframes)
import time
start_time = time.time() 
# ============================================================
# 1. PARÁMETROS DEL ANÁLISIS
# ============================================================
n=16752
new_n = 8164 # nuevo número de días
fecha_actual = "2025-12-01"
# Parámetros de FAI
window_minutes=30
fai_temp_range = (8, 35)
fai_em_jump = 0.014
fai_em_threshold_d=fai_em_jump
date_column = "date"
duration = True
FAI_duration = 4
exclude_last_minutes = 20 # minutos al final del día para excluir FAIs
filter_flare_coincidence = True
method = "filteredend_nooverlap" # "all", "true", "filtered", "filteredend" o "filteredend_nooverlap"
flares="ABCMX" # "valid", "no_A", "no_A_B", 
                                  # "no_A_B_nonoverlap" o "no_A_B_nonoverlap_wstart"
                                  # "M_X_nonoverlap_wstart"
nooverlap = True
min_gap_minutes_overlap = 20 #FILTRO EXTRA: remover FAI que ocurren tras un flare en [end, end + min_gap_minutes_overlap] 

# Formato de threshold para nombres de carpetas
threshold_str = f"{fai_em_jump}"   # "0.005" # convertir a string sin notación científica
fai_em_threshold_d = threshold_str.replace(".", "d") # reemplazar el punto por 'd'

# ---------------------------
# Carpetas de análisis
# ---------------------------
# Folder Name
output_dir = f"Data_for_{n}_days"
ensure_dir(path = output_dir)

analysis_dir = f"{output_dir}/Analysis_FAI"
ensure_dir(path = analysis_dir) # Verificar folder

# Crear subcarpeta para guardar resultados de análisis
analysis_esp = os.path.join(analysis_dir, f"Analysis_FAI_{flares}_T{fai_temp_range[0]}-{fai_temp_range[1]}_dEM{fai_em_threshold_d}_dur{FAI_duration}min")
# Verificar folder
ensure_dir(path = analysis_esp)

# ---------------------------
# Lectura de datos # Valid data:
# ---------------------------
input_dir = f"Data_for_{n}_days"

# Path of cleaned data in csv
csv_path_full = f"{input_dir}/df_full_{new_n}_valid.csv"
# Path of valid flares: 
csv_path_flares = f"{input_dir}/df_flares_{new_n}_{flares}.csv"

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
                    fai_em_jump = fai_em_jump,
                    date_column = "date",
                    duration = duration,
                    FAI_duration = FAI_duration,
                    exclude_last_minutes = exclude_last_minutes,
                    filter_flare_coincidence = filter_flare_coincidence,
                    df_flares = df_flares_valid,
                    flare_peak_col = "PeakTime",
                    flare_end_col = "EndTime",
                    nooverlap = nooverlap,          
                    min_gap_minutes_overlap = min_gap_minutes_overlap,
                    verbose = True,
                    output_dir=analysis_esp)

# Acceder a los resultados
df_fai_full = result["df_fai_full"]      # Todos los datos GOES con columnas de evaluación
df_fai_all = result["df_fai_all"]        # Todos los candidatos
df_fai_true = result["df_fai_true"]      # FAI con duración mínima
df_fai_filtered = result["df_fai_filtered"]  # FAI sin coincidencia con el final de flares
df_fai_filteredend = result["df_fai_filteredend"]  # FAI sin los últimos {exclude_last_minutes} min del día
df_fai_filteredend_nooverlap = result["df_fai_filteredend_nooverlap"] # FAI sin alertas unos minutos después de un flare

# all = todos los FAI segun críterios de EM y T
# true = todos los FAI segun críterios de EM, T y duración del FAI activado
# filtered = todos los FAI segun críterios de EM, T, duración del FAI activado
#            y que no están entre el peak y end de una fulguración
#method = "all"  # "all", "true" o "filtered"

method_mapping = {
    "all": ("df_fai_all", df_fai_all),
    "true": ("df_fai_true", df_fai_true),
    "filtered": ("df_fai_filtered", df_fai_filtered),
    "filteredend": ("df_fai_filteredend", df_fai_filteredend),
    "filteredend_nooverlap": ("df_fai_filteredend_nooverlap", df_fai_filteredend_nooverlap)
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


# gráfica % detección por clases
def plot_fai_flare_dualbar_by_class(
    df_fai_assoc, 
    df_anticipation_time,
    n_days,
    window_minutes,
    graphics_dir,
    figsize_X=11, figsize_y=6,
    column_assoc="Associated_Flare", 
    fai_column="FAIalerts_WPeak",
    class_column="ClassLetter"
):
    """
    Gráfico 100% stacked para cualquier tipo de clase (A/B/C/M/X o grupos).
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    # --- Preparar datos de Flares por clase ---
    if df_anticipation_time[fai_column].dtype in ['int64', 'float64']:
        df_anticipation_time['FAI_bool'] = df_anticipation_time[fai_column] > 0
    else:
        df_anticipation_time['FAI_bool'] = df_anticipation_time[fai_column].astype(bool)

    # Obtener lista de clases automáticamente
    flare_classes = sorted(df_anticipation_time[class_column]
                           .dropna()
                           .unique()
                           .tolist())

    class_stats = []

    for class_value in flare_classes:
        class_flares = df_anticipation_time[df_anticipation_time[class_column] == class_value]
        if len(class_flares) > 0:
            flare_with_fai = class_flares['FAI_bool'].sum()
            flare_without_fai = len(class_flares) - flare_with_fai
            total_flare_class = len(class_flares)

            pct_flare_with = flare_with_fai / total_flare_class * 100
            pct_flare_without = flare_without_fai / total_flare_class * 100
            
            class_stats.append({
                'Class': class_value,
                'Total_Flares': total_flare_class,
                'With_FAI': flare_with_fai,
                'Without_FAI': flare_without_fai,
                'Pct_With_FAI': pct_flare_with,
                'Pct_Without_FAI': pct_flare_without
            })

    # Crear DataFrame con stats
    stats_df = pd.DataFrame(class_stats)

    # --- Gráfico ---
    
    fig, ax = plt.subplots(figsize=(figsize_X, figsize_y))

    bars = stats_df['Class'].tolist()
    pct_with_fai = stats_df['Pct_With_FAI'].tolist()
    pct_without_fai = stats_df['Pct_Without_FAI'].tolist()
    totals = stats_df['Total_Flares'].tolist()

    # Parte inferior (con FAI)
    ax.bar(bars, pct_with_fai,
           color='#6BCB77', label='With FAI', alpha=0.9, edgecolor='black')

    # Parte superior (sin FAI)
    ax.bar(bars, pct_without_fai,
           bottom=pct_with_fai,
           color='#FF6B6B', label='Without FAI', alpha=0.9, edgecolor='black')

    # Etiquetas internas
    for i, (with_pct, without_pct, total) in enumerate(zip(pct_with_fai, pct_without_fai, totals)):
        if with_pct > 5:
            ax.text(i, with_pct / 2, f"{with_pct:.1f}%",
                    ha='center', va='center', fontsize=9, color='black', fontweight='bold')

        if without_pct > 5:
            ax.text(i, with_pct + without_pct/2, f"{without_pct:.1f}%",
                    ha='center', va='center', fontsize=9, color='black', fontweight='bold')

        ax.text(i, -6, f"n={total}",
                ha='center', va='top', fontsize=9, color='black', fontweight='bold')

    # Ajustes de gráfico
    ax.set_ylim(0, 101)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Flare Class", fontsize=12,  labelpad=15)
    ax.set_title(
        f"FAI Detection by Flare Class\n({n_days} days, Window = {window_minutes} min)",
        fontsize=14, fontweight='bold')

    ax.legend(title="FAI Detection", loc='center left',
              bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Totales generales
    total_flares_all = df_anticipation_time[class_column].notna().sum()
    total_with_fai_all = df_anticipation_time['FAI_bool'].sum()
    total_without_fai_all = total_flares_all - total_with_fai_all
    pct_with_fai_all = total_with_fai_all / total_flares_all * 100

    total_text = (
        f"Overall Totals:\n"
        f"Total Flares: {total_flares_all}\n"
        f"With FAI: {total_with_fai_all} ({pct_with_fai_all:.1f}%)\n"
        f"Without FAI: {total_without_fai_all} ({100 - pct_with_fai_all:.1f}%)"
    )
 
    # Crear el recuadro de totales
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8, edgecolor='black')
    ax.text(1.05, 0.85, total_text, transform=ax.transAxes, fontsize=11,
            fontweight='bold', verticalalignment='top', bbox=props)

    # Recuadro 2: Parámetros FAI (debajo de la leyenda)
    params_text = f"Window:{fai_column}\nMethod:{method}\nFAI Parameters:\nT: {fai_temp_range[0]}-{fai_temp_range[1]} MK\ndEM > {fai_em_jump}\nDuration: {FAI_duration} {duration}"
    
    ax.text(1.05, 0.25, params_text, transform=ax.transAxes, fontsize=9,
            fontweight='normal', verticalalignment='top', bbox=props,
            ha='left', linespacing=1.3)

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # Guardar imagen
    output_name = f"{n_days}d_FAI_Detection_By_Class_W{window_minutes}min_{class_column}.png"
    output_path = os.path.join(graphics_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Gráfico guardado en: {output_path}")

    return stats_df

#graficas
print("Creando carpetas")
analysis_dir = os.path.join(f"Data_for_{n}_days/Analysis_FAI")
# Crear carpeta de análisis (si no existe) y mostrar mensaje
if os.path.exists(analysis_dir ):
    print(f"⚠️ La carpeta de análisis ya existía: {analysis_dir }")
else:
    os.makedirs(analysis_dir , exist_ok=True)
    print(f"📁 Carpeta de análisis creada: {analysis_dir}")

# Crear subcarpeta para guardar resultados de análisis
analysis_esp = os.path.join(analysis_dir, f"Analysis_FAI_{flares}_T{fai_temp_range[0]}-{fai_temp_range[1]}_dEM{fai_em_threshold_d}_dur{FAI_duration}min")
# Crear carpeta de análisis (si no existe) y mostrar mensaje
if os.path.exists(analysis_esp ):
    print(f"⚠️ La carpeta de análisis ya existía: {analysis_esp }")
else:
    os.makedirs(analysis_esp , exist_ok=True)
    print(f"📁 Carpeta de análisis creada: {analysis_esp}")

folder = f"Graphics_{method}_W{window_minutes}min"

# Crear subcarpeta para guardar resultados de análisis
graphics_dir = os.path.join(analysis_esp, folder)
# Crear subcarpeta para guardar resultados de análisis (si no existe) y mostrar mensaje
if os.path.exists(graphics_dir):
    print(f"⚠️ La carpeta de análisis ya existía: {graphics_dir}")
else:
    os.makedirs(graphics_dir , exist_ok=True)
    print(f"📁 Carpeta de análisis creada: {graphics_dir}") 


print("Creando grafica:")
# Llamar la función
stats = plot_fai_flare_dualbar_by_class(
    df_fai_assoc, 
    df_anticipation_time,
    n_days=new_n,
    window_minutes=window_minutes,
    graphics_dir=graphics_dir,
    fai_column="FAIalerts_WPeak",
    class_column="ClassLetter"
)

# Llamar la función
stats = plot_fai_flare_dualbar_by_class(
    df_fai_assoc, 
    df_anticipation_time,
    n_days=new_n,
    window_minutes=window_minutes,
    figsize_X=15, figsize_y=6,
    graphics_dir=graphics_dir,
    fai_column="FAIalerts_WPeak",
    class_column="ClassGroup"
)


print("\n🎯 ANÁLISIS COMPLETO")
print(f"Falsos negativos (flares sin FAI): {false_negative_percentage:.2f}%")
print(f"Falsos positivos (FAIs no asociados a flares): {false_positive_percentage:.2f}%")

end_time = time.time()
elapsed = end_time - start_time

print("\n⏱️ Tiempo total de ejecución:")
mins = int(elapsed // 60)
secs = int(elapsed % 60)

print(f"\n⏱️ Tiempo total: {mins} min {secs} sec")

# ---------------------------
# 7️⃣ Crear tabla resumen FAI por clase
# ---------------------------

import pandas as pd
import os

# Configuración de parámetros para esta corrida
configuration = f"T in [{fai_temp_range[0]},{fai_temp_range[1]}], duration={FAI_duration}min"
delta_em = fai_em_jump
false_alarm_rate = false_positive_percentage  # ya calculado

# Lista de clases GOES
flare_classes = ['A', 'B', 'C', 'M', 'X']

# Inicializar diccionario de resultados
table_dict = {
    "Configuration": [configuration],
    "ΔEM": [delta_em],
    "False Alarm": [false_alarm_rate]
}

# Calcular porcentaje de detección por clase
for cl in flare_classes:
    df_class = df_anticipation_time[df_anticipation_time['ClassLetter'] == cl]
    if len(df_class) > 0:
        pct_detected = df_class['FAIalerts_WPeak'].gt(0).sum() / len(df_class) * 100
    else:
        pct_detected = 0.0
    table_dict[cl] = [round(pct_detected, 1)]

# Crear DataFrame final
df_summary = pd.DataFrame(table_dict)

# Carpeta de resultados
summary_dir = os.path.join(analysis_esp, "Summary")
os.makedirs(summary_dir, exist_ok=True)

# Guardar como CSV
csv_summary_path = os.path.join(summary_dir, f"FAI_summary_{method}.csv")
df_summary.to_csv(csv_summary_path, index=False)
print(f"✅ Tabla resumen guardada en CSV: {csv_summary_path}")

# Guardar también como TXT (formato tabulado)
txt_summary_path = os.path.join(summary_dir, f"FAI_summary_{method}.txt")
df_summary.to_csv(txt_summary_path, index=False, sep='\t')
print(f"✅ Tabla resumen guardada en TXT: {txt_summary_path}")
