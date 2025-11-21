# clean_flares.py               
# Limpieza específica de flares (GOES, HEK, SWPC)
import os
import pandas as pd
from data_cleaning_utils import ( ensure_dir,
                                 check_and_fix_csv,
                                 filter_flares_by_goes_dates,
                                 calculate_flare_durations,
                                 clean_flare_by_class,
                                 clean_flares_by_duration,
                                 plot_flare_times_histograms,
                                 plot_flare_distribution,
                                 filter_flares_by_class)
    
# número de días a analizar
n = 100
new_n = 59
fecha_actual = "2025-11-19"

# Folder Name
output_dir = f"Data_for_{n}_days"

ensure_dir(path = output_dir)

# path of GOES and flare data full:
csv_path_flares = f"{output_dir}/all_df_flare_data_{n}.csv"
# Verificar existencia de archivos y avisar
if not os.path.exists(csv_path_flares):
    print(f"⚠️ No se encontró el archivo de flares: {csv_path_flares}")
else:
    print(f"✅ Archivo de flares encontrado: {csv_path_flares}")

df_flares = pd.read_csv(csv_path_flares)

# Path of cleaned data in csv
csv_path_full = f"{output_dir}/df_full_{new_n}_valid.csv"
# Verificar existencia de archivos y avisar
if not os.path.exists(csv_path_full):
    print(f"⚠️ No se encontró el archivo GOES: {csv_path_full}")
else:
    print(f"✅ Archivo GOES encontrado: {csv_path_full}")
df_full_valid = pd.read_csv(csv_path_full)

Flare_graphics_dir = os.path.join(f"Flare_analysis_graphs_{new_n}")
ensure_dir(path = Flare_graphics_dir)

# fix: column-datetime, resolution-1min, No-duplicate
df_flares_clean = check_and_fix_csv(csv_path_flares, output_dir, n,
                             output_filename = f"all_df_flares_{n}_cleaned.csv", 
                             time_col="PeakTime")

# --------------------------------------------------
# 1) Filtrar por días de datos GOES válidos
df_flares_filtered = filter_flares_by_goes_dates(
    df_flares_clean,
    df_full_valid,
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}.csv"
)
# 2) Limpiar y arreglar columnas de clase
df_flares_clean_class = clean_flare_by_class(
    df_flares_filtered,
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}_cleaned_class.csv",
    group_subclasses=True,
    funnel_dir=Flare_graphics_dir
)
# Calcular duraciones necesarias para filtrado
df_flares_clean_class = calculate_flare_durations(df_flares_clean_class)

# 3) Aplicar filtros por duración
df_flares_valid = clean_flares_by_duration(
    df_flares_clean_class,
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}_valid.csv",
    funnel_dir=Flare_graphics_dir
)

print("✔ Limpieza de flares completada.")
print("\nGraficando histogramas")
# Histogramas para todas las clases combinadas
plot_flare_times_histograms(df_flares_valid, class_column="ClassLetter", class_filter=None, graphics_dir=Flare_graphics_dir)
# Histogramas para flares clase C
plot_flare_times_histograms(df_flares_valid, class_column="ClassLetter", class_filter="C", graphics_dir=Flare_graphics_dir)
# Histogramas para flares clase M
plot_flare_times_histograms(df_flares_valid, class_column="ClassLetter", class_filter="M", graphics_dir=Flare_graphics_dir)
# Histogramas para flares clase X
plot_flare_times_histograms(df_flares_valid, class_column="ClassLetter", class_filter="X", graphics_dir=Flare_graphics_dir)
#
plot_flare_times_histograms(df_flares_valid, class_column="ClassGroup", class_filter="C1-4.9", graphics_dir=Flare_graphics_dir)
#
print("\nGraficando distribución de flares por clases")
# Histogramas por letra de clase
plot_flare_distribution(df_flares_valid, column="ClassLetter", graphics_dir=Flare_graphics_dir)
# Histogramas por grupo de clase
plot_flare_distribution(df_flares_valid, column="ClassGroup", graphics_dir=Flare_graphics_dir)

print("\nFiltrando flares por clases")
# Excluir la clase A
df_no_A = filter_flares_by_class(
    df_flares_valid,
    exclude_classes=['A'],
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}_no_A.csv"
)
# Excluir las clases A y B
df_no_A_B = filter_flares_by_class(
    df_flares_valid,
    exclude_classes=['A','B'],
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}_no_A_B.csv"
)
#Incluir solo clases  M, X
df_M_X = filter_flares_by_class(
    df_flares_valid,
    include_classes=['M','X'],
    output_dir=output_dir,
    output_name=f"df_flares_{new_n}_no_A_B_C.csv"
)
print("\nFiltros terminados")