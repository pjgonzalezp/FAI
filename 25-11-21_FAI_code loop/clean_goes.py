#clean_goes.py          
## Limpieza específica de datos GOES
import os
import pandas as pd
from data_cleaning_utils import ( ensure_dir,
                                 check_and_fix_csv,
                                 filter_days_with_long_gaps,
                                 clean_goes_data_full,
                                 plot_with_mask_comparison,
                                 plot_comparison_scatter)
    
# número de días a analizar
n=100
fecha_actual = "2025-11-19"

# Folder Name
output_dir = f"Data_for_{n}_days"
# Verificar folder
ensure_dir(path = output_dir)

# path of GOES and flare data full:
csv_path_full = f"{output_dir}/all_df_full_{n}.csv"

# Verificar existencia de archivos y avisar
if not os.path.exists(csv_path_full):
    print(f"⚠️ No se encontró el archivo GOES: {csv_path_full}")
else:
    print(f"✅ Archivo GOES encontrado: {csv_path_full}")

df_full = pd.read_csv(csv_path_full)

# fix: column-datetime, resolution-1min, No-duplicate
df_full_clean = check_and_fix_csv(csv_path_full, output_dir, n,
                             output_filename = f"all_df_full_{n}_cleaned.csv", 
                             time_col="date")
# -------------------------------------------------------------------
# Búsqueda de días con datos faltantes en más de 20 min consecutivos
# -------------------------------------------------------------------
df_full_clean_without_gaps, days_with_gaps, new_n = filter_days_with_long_gaps(df_full_clean, output_dir, gap_minutes=20)


# Carpeta principal donde se guardarán las gráficas
GOES_graphics_dir = f"GOES_analysis_graphs_{new_n}"
# Crear carpeta si no existe y avisar
ensure_dir(path = GOES_graphics_dir)

# Llamar a la función
result = clean_goes_data_full(
    df=df_full_clean_without_gaps,  # tu DataFrame limpio sin gaps
    output_dir=output_dir,          
    GOES_graphics_dir=GOES_graphics_dir,
    repeat_filter=1,                # filtra T_cor que se repite más de 1 vez
    T_lower=1,                      # límite inferior T_cor (MK)
    T_upper=100,                    # límite superior T_cor (MK)
    save_funnel=True                # guarda la gráfica tipo funnel
)

# Obtener DataFrames filtrados
df_cleaned = result["df_cleaned"]   # antes de filtrar repetidos y outliers, para gráficas con mask
df_final = result["df_final"]       # filtrado completo, listo para análisis final

# info para la gráfica tipo funnel
steps_counts = result["steps_counts"]
steps_labels = result["steps_labels"]


# Crear máscara para los T_cor que se repiten (más de 1 vez)
T_counts = df_cleaned['T_cor'].value_counts()
mask1 = df_cleaned['T_cor'].isin(T_counts[T_counts > 1].index)

# Dividir en “top” y “other” según la máscara
df_cleaned = df_cleaned.copy()
df_cleaned['xrsa_corr/xrsb_corr_ratio'] = df_cleaned['xrsa_corr'] / df_cleaned['xrsb_corr']

df_topT = df_cleaned[mask1]
df_other = df_cleaned[~mask1]

# Combinar para usar en la función de plot
df_combined = pd.concat([df_other, df_topT], ignore_index=True)
mask_combined = pd.Series([False]*len(df_other) + [True]*len(df_topT))

# Llamar a la función de plot
fig, axes = plot_with_mask_comparison(
    df=df_combined,
    mask=mask_combined,
    x_cols=['xrsa_corr/xrsb_corr_ratio', 'xrsa_corr/xrsb_corr_ratio'],  # calculado antes
    y_cols=['T_cor', 'EM_cor'],
    titles=['T_cor vs ratio', 'EM_cor vs ratio'],
    mask_labels=['Frequent T_cor', 'Non-frequent T_cor'],
    colors=['red', 'blue'],
    x_scales=['log', 'log'],
    y_scales=['linear', 'log'],
    x_limits=[None, None], y_limits=[(2.5, 3.5), None], 
    figsize=(12, 6),
    alpha=0.5, s=10,
    save=True, save_dir=GOES_graphics_dir,
    filename="mask_comparison.png")

# Definir parámetros
dataframes = [df_full_clean_without_gaps, df_cleaned, df_final]  # Tus DataFrames
nombres = ['Full DataFrame', 'Cleaned DataFrame', 'Valid DataFrame']
    
columnas_x = ['EM_cor_norm', 'T_cor']  # Columnas para eje X
columnas_y = ['xrsb_corr', 'xrsb_corr']  # Columnas para eje Y

escalas_x = ['log', 'linear']  # Escalas para eje X
escalas_y = ['log', 'log']     # Escalas para eje Y

limites_x = [None, (0, 30)]    # Límites para eje X
limites_y = [None, None]       # Límites para eje Y

colores = ['blue', 'red']      # Colores para cada subplot

# Llamar a la función
plot_comparison_scatter(
    df_list=dataframes,
    x_cols=columnas_x,
    y_cols=columnas_y,
    titles=nombres,
    colors=colores,
    x_scales=escalas_x,
    y_scales=escalas_y,
    x_limits=limites_x,
    y_limits=limites_y,
    figsize=(10, 9),
    alpha=0.5, s=10,
    save=True, save_dir=GOES_graphics_dir,
    filename="scatter_comparison.png")