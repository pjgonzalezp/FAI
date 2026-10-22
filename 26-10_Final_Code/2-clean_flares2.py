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
                                 filter_flares_by_class,
                                 filter_overlapping_flares,
                                 filter_flares_start_of_day)
import time
start_time = time.time() 
# número de días a analizar
n = 16752
new_n = 8164 #número de df_full_{#}_valid.csv
fecha_actual = "2025-12-01"
# filtros de flares
min_gap_minutes=20    # Quitar solapamientos. tiempo entre el end y start del siguiente flare
min_start_minutes=15  # Quitar los flares que están en los primeros X minutos

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

Flare_graphics_dir = os.path.join(f"{output_dir}/Flare_analysis_graphs_{new_n}")
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
    output_name=f"df_flares_{new_n}_filtered.csv" 
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

print("\nInicio Filtrando archivo general que usaremos para FAI completo")



print(f"\n✅ Flares validos luego de limpieza: {len(df_flares_valid)}")
print("\nAhora empezamos filtros:")
# ==========================
# CONFIGURACIÓN DE CLASES
# ==========================
class_filters = {
    "ABCMX": ['A','B','C','M','X'],
    "BCMX": ['B','C','M','X'],
    "CMX": ['C','M','X'],
    "MX":  ['M','X'],
    "A":   ['A'],
    "B":   ['B'],
    "C":   ['C'],
    "M":   ['M'],
    "X":   ['X'],
}

# ==========================
# PIPELINE AUTOMÁTICO
# ==========================
def generate_class_filtered_flares(df_flares_valid,
                                    class_filters,
                                    min_gap_minutes=min_gap_minutes,
                                    min_start_minutes=min_start_minutes,
                                    output_dir="."):
    """
    Pipeline que:
      1) filtra por clase,
      2) elimina solapamientos entre flares,
      3) elimina flares iniciados en primeros minutos del día.

    Genera un CSV por cada categoría del diccionario class_filters.
    """

    results = {}

    for label, classes in class_filters.items():

        print(f"\n### Procesando clases {label}: {classes}")

        # 1. Filtrar por clase
        df_class = filter_flares_by_class(df_flares_valid,
                                          include_classes=classes)
        print(f" - Después de class filter: {len(df_class)} flares")

        # 2. Quitar solapamientos
        df_no_overlap = filter_overlapping_flares(
            df_class,
            start_col="StartTime",
            end_col="EndTime",
            min_gap_minutes=min_gap_minutes
        )
        print(f" - No solapados: {len(df_no_overlap)} flares")

        # 3. Quitar los que empiezan en los primeros X minutos
        df_filtered = filter_flares_start_of_day(
            df_no_overlap,
            start_col="PeakTime",
            min_start_minutes=min_start_minutes
        )
        print(f" - Después de early-start filter: {len(df_filtered)} flares")

        # Guardar CSV
        output_path = os.path.join(output_dir, f"df_flares_{new_n}_{label}.csv")
        df_filtered.to_csv(output_path, index=False)
        print(f" - Archivo guardado en: {output_path}")

        # Guardar resultado COMPLETO
        results[label] = {
            "df_class": df_class,
            "df_no_overlap": df_no_overlap,
            "df_filtered": df_filtered
        }
    # ==============
    # GUARDAR RESUMEN
    # ==============
    summary_path = os.path.join(output_dir, "flare_class_summary.txt")
    with open(summary_path, "w") as f:
        f.write("RESUMEN DE FILTROS DE FLARES POR CLASE\n")
        f.write("=====================================\n\n")
        f.write(f"Total flares válidos después de limpieza general: {len(df_flares_valid)}\n\n")

        for label, classes in class_filters.items():
            df_class   = results[label]["df_class"]
            df_overlap = results[label]["df_no_overlap"]
            df_final   = results[label]["df_filtered"]

            f.write(f"--- Clase {label} ({classes}) ---\n")
            f.write(f"Flares después del class filter: {len(df_class)}\n")
            f.write(f"Flares sin solapamiento: {len(df_overlap)}\n")
            f.write(f"Flares sin solapamiento y sin early-start: {len(df_final)}\n\n")

    print(f"\n📄 Resumen guardado en: {summary_path}")

    return results

results = generate_class_filtered_flares(
    df_flares_valid=df_flares_valid,
    class_filters=class_filters,
    min_gap_minutes=min_gap_minutes,
    min_start_minutes=min_start_minutes,
    output_dir=output_dir
)

df_ABCMX = results["ABCMX"]["df_filtered"]  # tomar la versión final filtrada

plot_flare_distribution(
    df_ABCMX,
    column="ClassLetter",
    graphics_dir=Flare_graphics_dir,
    df_label="ABCMX_filtered"
)
plot_flare_distribution(
    df_ABCMX,
    column="ClassGroup",
    graphics_dir=Flare_graphics_dir,
    df_label="ABCMX_filtered"
)
c
df_BCMX = results["BCMX"]["df_filtered"]  # tomar la versión final filtrada

plot_flare_distribution(
    df_BCMX,
    column="ClassLetter",
    graphics_dir=Flare_graphics_dir,
    df_label="BCMX_filtered"
)
plot_flare_distribution(
    df_BCMX,
    column="ClassGroup",
    graphics_dir=Flare_graphics_dir,
    df_label="BCMX_filtered"
)

df_CMX = results["CMX"]["df_filtered"]  # tomar la versión final filtrada

plot_flare_distribution(
    df_CMX,
    column="ClassLetter",
    graphics_dir=Flare_graphics_dir,
    df_label="CMX_filtered"
)
plot_flare_distribution(
    df_CMX,
    column="ClassGroup",
    graphics_dir=Flare_graphics_dir,
    df_label="CMX_filtered"
)

df_MX = results["MX"]["df_filtered"]  # tomar la versión final filtrada

plot_flare_distribution(
    df_MX,
    column="ClassLetter",
    graphics_dir=Flare_graphics_dir,
    df_label="MX_filtered"
)
plot_flare_distribution(
    df_MX,
    column="ClassGroup",
    graphics_dir=Flare_graphics_dir,
    df_label="MX_filtered"
)

end_time = time.time()
elapsed = end_time - start_time

print("\n⏱️ Tiempo total de ejecución:")
mins = int(elapsed // 60)
secs = int(elapsed % 60)

print(f"\n⏱️ Tiempo total: {mins} min {secs} sec")