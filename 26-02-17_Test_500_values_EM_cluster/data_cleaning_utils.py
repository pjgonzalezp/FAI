#data_cleaning_utils.py      
## Funciones genéricas de limpieza, filtros, helpers
# - Limpieza de CSV y verificación (check_and_fix_csv)
# - Filtrado de días con huecos largos (filter_days_with_long_gaps)
# - Limpieza completa de GOES (clean_goes_data_full)
# - Gráficas: 
#       - Funnel plot (plot_cleaning_funnel)
#       - Comparación con máscara (plot_with_mask_comparison)
#       - Comparación entre DataFrames (plot_comparison_scatter)

from datetime import datetime, timedelta
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# Función: Verificar folder
def ensure_dir(path):
    """
    Verifica si una carpeta existe; si no, la crea.
    
    Parameters
    ----------
    path : str
        Ruta de la carpeta a verificar o crear.

    Returns
    -------
    str
        La ruta final de la carpeta.
    """
    if os.path.exists(path):
        print(f"⚠️ La carpeta ya existía: {path}")
    else:
        os.makedirs(path, exist_ok=True)
        print(f"📁 Carpeta creada: {path}")
    
    return path

# Función: fix: column-datetime, resolution-1min, No-duplicate
def check_and_fix_csv(csv_path, output_dir, n, output_filename, time_col="Unnamed: 0"):
    """
    Checks and fixes a CSV file to ensure:
    - The time column is in datetime format
    - The time resolution is 1 minute
    - There are no duplicate timestamps
    Saves the cleaned file in the specified output directory.
    """
    
    print(f"\n=== Checking file: {csv_path} ===")
    
    # --- Load CSV ---
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading the file: {e}")
        return
    
    changes = []  # keep track of any corrections made
    
    # --- Verify time column ---
    if time_col not in df.columns:
        print(f"❌ The time column '{time_col}' does not exist in the CSV.")
        return
    
    # --- Convert to datetime ---
    # Remove decimals if present
    original_time_values = df[time_col].astype(str)
    if original_time_values.str.contains(r"\.").any():
        print("🧹 Decimal points detected in timestamps. Removing fractional seconds...")
        df[time_col] = original_time_values.str.split(".").str[0]
    else:
        print("✅ No decimal points found in timestamps.")

    # Try to convert to datetime
    try:
        df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if df[time_col].isna().any():
            print("⚠️ Some timestamps could not be parsed using the strict format. They were set to NaT.")
    except Exception as e:
        print(f"⚠️ Error while converting to datetime: {e}")
        print("🔄 Retrying with automatic format detection...")
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")



    # Drop invalid (NaT) rows
    # --- Drop invalid (NaT) rows ---
    n_null = df[time_col].isna().sum()
    if n_null > 0:
        print(f"⚠️ Found {n_null} invalid or missing timestamps. Removing them...")
        changes.append(f"Removed {n_null} rows with invalid or missing timestamps.")
        df = df.dropna(subset=[time_col])
    else:
        print("✅ No invalid or missing timestamps found.")

    # --- Check for duplicates ---
    duplicate_count = df.duplicated(subset=time_col).sum()
    if duplicate_count > 0:
        print(f"⚠️ Found {duplicate_count} duplicated timestamps. Keeping the first occurrence...")
        changes.append(f"Removed {duplicate_count} duplicate rows.")
        n_before = len(df)
        df = df.drop_duplicates(subset=time_col, keep="first")
        n_after = len(df)
        print(f"   → Rows before: {n_before}, after: {n_after}")
    else:
        print("✅ No duplicated timestamps found.")
    
    # --- Check time resolution ---
    df = df.sort_values(by=time_col)
    diffs = df[time_col].diff().dropna()
    freq_counts = diffs.value_counts()

    if not freq_counts.empty:
        most_common_freq = freq_counts.index[0]
        if most_common_freq.total_seconds() == 60:
            print("✅ Main resolution: 1 minute")
        else:
            changes.append(f"Detected time step: {most_common_freq}.")
            print(f"⚠️ Main resolution is not 1 minute, detected: {most_common_freq}")
    else:
        print("⚠️ Could not determine time resolution (too few rows).")
        
    # --- Save cleaned CSV ---
    
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    
    # --- Final report ---
    print(f"\nFile saved to: {output_path}")
    if changes:
        print("🔧 Changes made:")
        for c in changes:
            print(" - " + c)
    else:
        print("✅ Data were already clean. No changes applied.")
    
    return df

# Función: detectar huecos largos
def filter_days_with_long_gaps(df, output_dir, gap_minutes=20):
    """
    Detecta y elimina días que tengan >= gap_minutes consecutivos de datos faltantes en GOES (xrsa/xrsb).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'date', 'xrsa', 'xrsb'.
    gap_minutes : int
        Número de registros consecutivos faltantes permitido antes de eliminar el día.
    
    Returns
    -------
    df_valid : DataFrame
        DataFrame filtrado sin los días con huecos largos.
    days_with_gaps : list
        Lista de días que fueron eliminados.
    new_n : int
        Número de días restantes después del filtro.
    """

    # --- Preparación ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.replace("", pd.NA, inplace=True)
    df["day"] = df["date"].dt.date

    goes_cols = ["xrsa", "xrsb"]
    df["GOES_missing"] = df[goes_cols].isna().any(axis=1)

    # --- Función interna ---
    def has_long_gap(group):
        missing = group.sort_values("date")["GOES_missing"].values
        count = 0
        for m in missing:
            if m:
                count += 1
                if count >= gap_minutes:
                    return True
            else:
                count = 0
        return False

    # --- Detectar días con huecos largos ---
    days_with_gaps = (
        df.groupby("day")
        .apply(lambda g: has_long_gap(g), include_groups=False)
    )

    days_with_gaps = days_with_gaps[days_with_gaps].index.tolist()

    # --- Filtrar días válidos ---
    df_valid = df[~df["day"].isin(days_with_gaps)].copy()
    new_n = df_valid['day'].nunique()

    # --- Resumen ---
    print(f"🕳️ Días con huecos ≥{gap_minutes} min consecutivos en GOES:")
    print(f"\n📅 Total de días originales: {df['day'].nunique()}")
    print(f"❌ Días eliminados: {len(days_with_gaps)}")
    print(f"✅ Días restantes: {new_n}")

    # --- Guardar archivos ---
    output_filename = f'df_full_{new_n}_cleaned.csv'
    output_path = os.path.join(output_dir, output_filename)
    df_valid.to_csv(output_path, index=False)

    # Guarda también los días con huecos
    gaps_filename = f"days_with_{gap_minutes}min_gaps.csv"
    pd.Series(days_with_gaps, name=f"days_with_{gap_minutes}min_gaps").to_csv(
        os.path.join(output_dir, gaps_filename),
        index=False
    )

    print(f"\n📁 Archivo guardado GOES clean sin gaps: {output_path}\n")
    
    return df_valid, days_with_gaps, new_n

# Función: filtrar negativos, ceros, NaN y T_Cor repetidos
def clean_goes_data_full(df, output_dir, GOES_graphics_dir, repeat_filter=1,
                         T_lower=1, T_upper=100, save_funnel=True):
    """
    Limpieza completa de datos GOES:
    1️⃣ Elimina negativos, ceros y NaN
    2️⃣ Filtra valores repetidos de T_cor
    3️⃣ Filtra outliers de T_cor según límites físicos
    4️⃣ Genera gráfica tipo funnel y guarda CSV final (opcional)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original con columnas 'xrsa', 'xrsb', 'xrsa_corr', 'xrsb_corr', 'T_cor', 'EM_cor', etc.
    output_dir : str
        Carpeta donde se guardará el CSV filtrado
    GOES_graphics_dir : str
        Carpeta donde se guardará la gráfica tipo funnel
    repeat_filter : int
        Número máximo de repeticiones de T_cor permitidas (mayor será eliminado)
    T_lower : float
        Límite inferior físico para T_cor (MK)
    T_upper : float
        Límite superior físico para T_cor (MK)
    save_funnel : bool
        Si True, guarda la gráfica tipo funnel
    
    Returns
    -------
    dict
        {
            "df_cleaned": df_no_nan,      # filtrado por negativos, ceros y NaN
            "df_final": df_Tcor_valid,    # filtrado completo incluyendo repetidos y outliers
            "steps_counts": counts_list,  # para gráfica funnel
            "steps_labels": steps_list    # labels de los pasos
        }
    """
    df = df.copy()
    
    # --- Convertir fechas y preparar ---
    df["date"] = pd.to_datetime(df["date"])
    df.replace("", pd.NA, inplace=True)
    print(f"\n1. Eliminando negativos:")
    # --- Eliminar negativos ---
    num_cols = df.select_dtypes(include='number').columns
    df = df[(df[num_cols] >= 0).all(axis=1)]

    print(f"2. Eliminando ceros:")
    # --- Eliminar ceros ---
    zero_cols = ['xrsa', 'xrsb', 'xrsa_corr', 'xrsb_corr']
    df = df[(df[zero_cols] != 0).all(axis=1)]
    
    print(f"3. Eliminando NaN:")
    # --- Eliminar NaN ---
    nan_cols = [col for col in df.columns if col not in ['date', 'observatory']]
    df_no_nan = df.dropna(subset=nan_cols)
    
    # --- Guardar df limpio antes de T_cor (para gráficas con mask) ---
    df_cleaned = df_no_nan.copy()
    
    print(f"4. Eliminando repetidos de T_cor:")
    # --- Filtrar valores repetidos de T_cor ---
    T_counts = df_no_nan['T_cor'].value_counts()
    repeated_T_values = T_counts[T_counts > repeat_filter].index
    df_no_repeated = df_no_nan[~df_no_nan['T_cor'].isin(repeated_T_values)]
    
    print(f"5. Eliminando outliers T_cor según límites físicos:")
    # --- Filtrar outliers T_cor según límites físicos ---
    df_Tcor_valid = df_no_repeated[
        (df_no_repeated["T_cor"] >= T_lower) &
        (df_no_repeated["T_cor"] <= T_upper)].copy()
    # por confirmar el número de días
    df_Tcor_valid["day"] = df_Tcor_valid["date"].dt.date

    new_n = df_Tcor_valid['day'].nunique()
    
    # --- Guardar CSV final ---
    final_filename = f"df_full_{new_n}_valid.csv"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, final_filename)
    df_Tcor_valid.to_csv(output_path, index=False)
    print(f"\n📁 Archivo guardado GOES sin gaps,-,0,NaN,Trep,outliers: {output_path}\n")

    # --- Preparar info para funnel plot ---
    steps_list = [
        'Total rows',
        'After removing negatives',
        'After removing zeros',
        'After removing NaN',
        'After removing repeated T_cor',
        'After removing T_cor outliers'
    ]
    counts_list = [
        len(df),                       # Total rows
        len(df),                       # Después de negativos
        len(df[df[zero_cols].gt(0).all(axis=1)]),  # Después de ceros
        len(df_no_nan),                 # Después de NaN
        len(df_no_repeated),            # Después de repetidos
        len(df_Tcor_valid)              # Después de outliers
    ]
    
    # --- Llamar a plot_cleaning_funnel si save_funnel=True ---
    if save_funnel:
        plot_cleaning_funnel(steps=steps_list,
                            counts=counts_list,
                            df_full_clean_len=len(df),
                            GOES_graphics_dir=GOES_graphics_dir,
                            filename="funnel_cleaning.png")
        

    return {
        "df_cleaned": df_cleaned,
        "df_final": df_Tcor_valid,
        "steps_counts": counts_list,
        "steps_labels": steps_list
    }


# Gráfica: Cleaning funnel plot
def plot_cleaning_funnel(steps, counts,
                         df_full_clean_len,
                         GOES_graphics_dir, 
                         filename="data_cleaning_funnel.png"):
    
    colors = ['lightblue', 'lightcoral', 'khaki', 'lightgreen', 'orchid', 'salmon']

    plt.figure(figsize=(10, 5))
    bars = plt.barh(range(len(steps)), counts, color=colors[:len(steps)], edgecolor='black')

    plt.yticks(range(len(steps)), steps)
    plt.xlabel('Number of rows')
    plt.title('Data Cleaning Funnel: GOES')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')

    # Añadir números y porcentajes
    percentages = [count / df_full_clean_len * 100 for count in counts]
    for i, (bar, count, percent) in enumerate(zip(bars, counts, percentages)):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        width_ratio = width / max(counts)
        if width_ratio > 0.25:
            plt.text(width / 2, y_pos, f"{count:,}\n({percent:.1f}%)", 
                     va='center', ha='center', fontsize=10)
        else:
            label_x = width + (max(counts) * 0.02)
            plt.text(label_x, y_pos, f"{count:,}\n({percent:.1f}%)", 
                     va='center', ha='left', fontsize=10)

    plt.xlim(0, max(counts)*1.1)
    plt.tight_layout()

    plot_path = os.path.join(GOES_graphics_dir, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"📁 Cleaning funnel plot saved to: {plot_path}")

# Gráfica: Mascara de revisión
def plot_with_mask_comparison(
        df, mask, x_cols, y_cols, titles=None,
        mask_labels=None, colors=None, x_scales=None, y_scales=None,
        x_limits=None, y_limits=None, figsize=(12, 6), alpha=0.5, s=10,
        save=False, save_dir=None, filename="mask_comparison.png"):
    """
    Scatter plots highlighting masked vs non-masked data.
    """

    n_plots = len(x_cols)

    # Defaults
    if mask_labels is None:
        mask_labels = ['Masked data', 'Non-masked data']
    if colors is None:
        colors = ['red', 'blue']
    if titles is None:
        titles = [f'{y} vs {x}' for x, y in zip(x_cols, y_cols)]
    if x_scales is None:
        x_scales = ['linear'] * n_plots
    if y_scales is None:
        y_scales = ['linear'] * n_plots

    df_masked = df[mask]
    df_non_masked = df[~mask]

    print(f"Masked data: {len(df_masked)} rows")
    print(f"Non-masked data: {len(df_non_masked)} rows")

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
        ax = axes[i]

        ax.scatter(df_non_masked[x_col], df_non_masked[y_col], 
                   s=s, alpha=alpha, color=colors[1], 
                   label=mask_labels[1])

        ax.scatter(df_masked[x_col], df_masked[y_col], 
                   s=s, alpha=alpha, color=colors[0], 
                   label=mask_labels[0])

        ax.set_xscale(x_scales[i])
        ax.set_yscale(y_scales[i])

        if x_limits and x_limits[i]:
            ax.set_xlim(x_limits[i])
        if y_limits and y_limits[i]:
            ax.set_ylim(y_limits[i])

        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()

    # Save or show
    if save:
        if save_dir is None:
            raise ValueError("save_dir must be provided when save=True.")
        plt.savefig(f"{save_dir}/{filename}", dpi=150)
        print(f"📁 Saved: {save_dir}/{filename}")
    else:
        plt.show()

    plt.close()

    return fig, axes

# Gráfica: comparación datos con y sin filtro
def plot_comparison_scatter(
        df_list, x_cols, y_cols, titles=None,
        colors=None, x_scales=None, y_scales=None,
        x_limits=None, y_limits=None,
        figsize=(10, 9), alpha=0.5, s=10,
        save=False, save_dir=None, filename="scatter_comparison.png"):
    """
    Grid of scatter plots comparing multiple DataFrames.
    """

    n_dfs = len(df_list)
    n_plots = len(x_cols)

    if titles is None:
        titles = [f'DataFrame {i+1}' for i in range(n_dfs)]
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    if x_scales is None:
        x_scales = ['linear'] * n_plots
    if y_scales is None:
        y_scales = ['linear'] * n_plots

    fig, axes = plt.subplots(n_dfs, n_plots, figsize=figsize)

    # Ensure 2D axes
    if n_dfs == 1:
        axes = axes.reshape(1, -1)
    if n_plots == 1:
        axes = axes.reshape(-1, 1)

    for i, df in enumerate(df_list):
        for j in range(n_plots):
            ax = axes[i, j]

            ax.scatter(df[x_cols[j]], df[y_cols[j]],
                       s=s, alpha=alpha, color=colors[j],
                       label=f'{x_cols[j]} vs {y_cols[j]}')

            ax.set_xscale(x_scales[j])
            ax.set_yscale(y_scales[j])

            if x_limits and x_limits[j]:
                ax.set_xlim(x_limits[j])
            if y_limits and y_limits[j]:
                ax.set_ylim(y_limits[j])

            ax.set_xlabel(x_cols[j], fontsize=12)
            ax.set_ylabel(y_cols[j], fontsize=12)
            ax.set_title(f'{titles[i]}: {x_cols[j]} vs {y_cols[j]}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

    plt.tight_layout()

    # Save or show
    if save:
        if save_dir is None:
            raise ValueError("save_dir must be provided when save=True.")
        plt.savefig(f"{save_dir}/{filename}", dpi=150)
        print(f"📁 Saved: {save_dir}/{filename}")
    else:
        plt.show()

    plt.close()

    return fig, axes

# Gráfica: resumen filtros flares
def plot_funnel(steps, counts, output_dir=None, filename="funnel_cleaning.png"):

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, steps, marker='o', color='blue')
    ax.set_xlabel("Número de flares válidos")
    ax.set_ylabel("Paso de limpieza")
    ax.set_title("Funnel plot limpieza de flares")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()  # Paso inicial arriba
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"📊 Funnel plot guardado: {path}")
    
    plt.close()

def save_cleaning_log(steps, counts, output_dir, filename="cleaning_log.txt"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    with open(path, "w") as f:
        f.write("Limpieza de flares - resumen por pasos\n")
        f.write("="*40 + "\n")
        for step, count in zip(steps, counts):
            f.write(f"{step:<35} : {count}\n")
        f.write("="*40 + "\n")
        f.write(f"Total flares finales: {counts[-1]}\n")
    
    print(f"📝 Log de limpieza guardado: {path}")

# crear columnas de tiempos
def calculate_flare_durations(df):
    """
    Convierte las columnas de tiempo a datetime, calcula duraciones en minutos
    y reordena columnas, manteniendo todas las columnas originales.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas 'StartTime', 'PeakTime', 'EndTime' y otras.

    Retorna
    -------
    pd.DataFrame
        DataFrame con nuevas columnas: 'StartPeak', 'StartEnd', 'PeakEnd',
        y con las columnas de tiempo/duración al principio.
    """
    df = df.copy()

    # --- Convertir a datetime ---
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors='coerce')
    df['PeakTime']  = pd.to_datetime(df['PeakTime'], errors='coerce')
    df['EndTime']   = pd.to_datetime(df['EndTime'], errors='coerce')

    # --- Calcular duraciones en minutos ---
    df['StartPeak'] = (df['PeakTime'] - df['StartTime']).dt.total_seconds() / 60
    df['StartEnd']  = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 60
    df['PeakEnd']   = (df['EndTime'] - df['PeakTime']).dt.total_seconds() / 60

    # --- Reordenar columnas ---
    time_cols = ['StartTime', 'PeakTime', 'EndTime', 'StartPeak', 'StartEnd', 'PeakEnd', 'Class', 'ClassLetter', 'ClassNumber', 'ClassGroup', 'NOAA_Class']
    other_cols = [c for c in df.columns if c not in time_cols]
    df = df[time_cols + other_cols]

    print(f"✔ Columnas de duración calculadas y DataFrame reordenado ({len(df)} flares).")

    return df

# Función: solo limpiar y arreglar clase
def clean_flare_by_class(df_flares, output_dir=None, output_name=None, group_subclasses=True, funnel_dir=None):
    df = df_flares.copy()
    counts = [len(df)]
    steps = ["Original"]
    
    # Validar clase
    df["Class"] = df["Class"].astype(str)
    pattern = r"^[A-Z]\d+(\.\d+)?$"
    df = df[df["Class"].str.match(pattern, na=False)]
    counts.append(len(df))
    steps.append("Clase válida (regex)")
    
    # Extraer letra y número
    df["ClassLetter"] = df["Class"].str[0]
    df["ClassNumber"] = df["Class"].str[1:].astype(float)

    # Eliminar letras no válidas (seguridad extra)
    valid_letters = ["A","B","C","M","X"]
    df = df[df["ClassLetter"].isin(valid_letters)]
    counts.append(len(df))
    steps.append("Eliminar letras inválidas")

    # Eliminar flares como M0.9, C0.3, etc.
    df = df[df["ClassNumber"] >= 1.0]
    counts.append(len(df))
    steps.append("Eliminar clases con número < 1.0")

    # Agrupar subclases
    group_ranges = {
        "A": [(1, 4.9), (5, 9.9)],
        "B": [(1, 4.9), (5, 9.9)],
        "C": [(1, 4.9), (5, 9.9)],
        "M": [(1, 4.9), (5, 9.9)],
        "X": [(1, 4.9), (5, 1000)],
    }
    def assign_group(cls):
        letter = cls[0]
        number = float(cls[1:])
        for low, high in group_ranges.get(letter, []):
            if low <= number <= high:
                return f"{letter}{low}-{high if high < 1000 else '+'}"
        return cls
    if group_subclasses:
        df["ClassGroup"] = df["Class"].apply(assign_group)
    else:
        df["ClassGroup"] = df["ClassLetter"]
    
    print(f"✔ Limpieza por clase completa. {len(df)} flares válidos.")
    
    # Guardar CSV
    if output_dir and output_name:
        path = os.path.join(output_dir, output_name)
        df.to_csv(path, index=False)
        print(f"📁 Archivo guardado: {path}")
    
    # Funnel plot
    if funnel_dir:
        plot_funnel(steps, counts, output_dir=funnel_dir, filename="funnel_class.png")
        save_cleaning_log(steps, counts, output_dir=funnel_dir, filename="log_class.txt")
    
    return df

# Función: limpieza por duración
def clean_flares_by_duration(df, output_dir=None, output_name=None, funnel_dir=None):
    """
    Limpieza basada en duración del flare:
    - Validar Start ≤ Peak ≤ End
    - Validar StartPeak y PeakEnd ≤ 720
    - Filtro especial de duración según clase
    - Crear Flare_ID
    """
    df = df.copy()
    counts = [len(df)]
    steps = ["Original"]
    
    letters = df['ClassLetter']
    
    # Duración según clase
    duration_by_class = ((letters < 'C') & (df['StartEnd'] <= 180)) | ((letters >= 'C') & (df['StartEnd'] <= 720))
    df = df[duration_by_class]
    counts.append(len(df))
    steps.append("Duración según clase")
    
    # Orden correcto
    valid_order = (df['StartTime'] <= df['PeakTime']) & (df['PeakTime'] <= df['EndTime'])
    df = df[valid_order]
    counts.append(len(df))
    steps.append("Orden Start ≤ Peak ≤ End")
    
    # Duraciones razonables
    valid_duration = (df['StartPeak'] <= 720) & (df['PeakEnd'] <= 720)
    df = df[valid_duration]
    counts.append(len(df))
    steps.append("StartPeak & PeakEnd ≤ 720")
    
    # Crear Flare_ID
    df["Flare_ID"] = df["PeakTime"].dt.strftime("%Y%m%d%H%M") + "_" + df["Class"]

    # --- Reordenar columnas ---
    time_cols = ['Flare_ID', 'StartTime']
    other_cols = [c for c in df.columns if c not in time_cols]
    df = df[time_cols + other_cols]

    counts.append(len(df))
    steps.append("Creación Flare_ID")
    
    print(f"✔ Limpieza por duración completa. {len(df)} flares válidos de {counts[0]}.")
    
    # Guardar CSV
    if output_dir and output_name:
        path = os.path.join(output_dir, output_name)
        df.to_csv(path, index=False)
        print(f"📁 Archivo guardado: {path}")
    
    # Funnel plot y log
    if funnel_dir:
        plot_funnel(steps, counts, output_dir=funnel_dir, filename="funnel_duration.png")
        save_cleaning_log(steps, counts, output_dir=funnel_dir, filename="log_duration.txt")
    
    return df

# Función: filtro de flares por días de datos GOES válidos
def filter_flares_by_goes_dates(df_flares, df_goes, output_dir=None, output_name=None):
    """
    Filtra flares para quedarse solo con aquellos cuyos DÍAS existen en los datos GOES.

    Parameters
    ----------
    df_flares : pd.DataFrame
        DataFrame de flares limpios, debe contener columna 'PeakTime'.
    df_goes : pd.DataFrame
        DataFrame GOES con columna 'date'.
    output_dir : str, optional
        Carpeta donde guardar el CSV final.
    output_name : str, optional
        Nombre del archivo CSV a guardar.

    Returns
    -------
    pd.DataFrame
        DataFrame de flares filtrados por días con datos GOES válidos.
    """

    df = df_flares.copy()

    # Convertir columnas a datetime
    df["PeakTime"] = pd.to_datetime(df["PeakTime"], errors="coerce")
    df_goes["date"] = pd.to_datetime(df_goes["date"], errors="coerce")

    # EXTRAER SOLO LA FECHA (sin hora)
    df["PeakDate"] = df["PeakTime"].dt.date
    df_goes["DateOnly"] = df_goes["date"].dt.date

    # Guardamos tamaño inicial
    n_before = len(df)

    # FILTRAR POR DÍA, NO POR DATETIME EXACTO
    df = df[df["PeakDate"].isin(df_goes["DateOnly"])]

    # Tamaño después
    n_after = len(df)

    # Log del filtrado
    print("=== Filtro por días GOES válidos ===")
    print(f"Flares antes del filtro:     {n_before}")
    print(f"Flares después del filtro:   {n_after}")
    print(f"Flares eliminados:           {n_before - n_after}")

    # Guardar si se pide
    if output_dir and output_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, output_name)
        df.to_csv(path, index=False)
        print(f"📁 Archivo filtrado guardado en: {path}")

    # Quitar columnas temporales si quieres
    df = df.drop(columns=["PeakDate"], errors="ignore")

    return df


# Gráfica: Histograma de tiempos por clase de flare
def plot_flare_times_histograms(df, class_column="ClassLetter", class_filter=None, graphics_dir="."):
    """
    Genera histogramas de tiempos (StartPeak, PeakEnd, StartEnd) para los flares filtrados por clase.
    
    Parámetros:
        df: DataFrame con los flares.
        class_column: columna a usar para filtrar la clase ('ClassLetter' o 'ClassGroup').
        class_filter: valor de la clase a filtrar, e.g. 'C', 'M', 'C1-4.9'. Si es None, usa todos.
        graphics_dir: carpeta donde guardar la figura.
    """

    # Filtrar DataFrame por clase si se indica
    if class_filter is not None:
        df_plot = df[df[class_column] == class_filter]
        title_suffix = f"Class {class_filter}"
        output_name = f"Histogram_Flares_Times_{class_filter}.png"
    else:
        df_plot = df
        title_suffix = "All Classes"
        output_name = f"Histogram_Flares_Times_All.png"

    if df_plot.empty:
        print(f"No flares found for {title_suffix}.")
        return

    # Crear la figura con 3 subplots
    plt.figure(figsize=(12,4))
    time_cols = ['StartPeak', 'PeakEnd', 'StartEnd']

    for i, col in enumerate(time_cols, 1):
        plt.subplot(1, 3, i)
        plt.hist(df_plot[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"{col} ({title_suffix})")
        plt.xlabel('Minutes')
        plt.ylabel('Number of Flares')

    plt.tight_layout()

    # Guardar figura
    output_path = os.path.join(graphics_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFile saved to: {output_path}")

    plt.close()

# Gráfica: Histograma de flares por clases
def plot_flare_distribution(df, column="ClassGroup", color_mode="palette:viridis", graphics_dir=".", df_label=None):
    """
    Genera una gráfica de distribución de clases de flares (conteo y porcentaje)
    usando la columna especificada (por defecto 'ClassGroup' o 'ClassLetter').
    """

    if column not in df.columns:
        raise ValueError(f"El DataFrame debe contener la columna '{column}'.")

    # -------------------------------
    # Conteos y porcentajes
    # -------------------------------
    counts = df[column].value_counts().sort_index()
    total_flares = counts.sum()
    percentages = (counts / total_flares * 100).round(1)

    # -------------------------------
    # Imprimir estadísticas
    # -------------------------------
    print(f"\nStatistics for {column}:")
    print(f"{'Class':<10} {'Count':>8} {'Percentage (%)':>15}")
    for cls, cnt, pct in zip(counts.index, counts.values, percentages.values):
        print(f"{cls:<10} {cnt:>8} {pct:>15.1f}")
    print(f"{'Total':<10} {total_flares:>8} {percentages.sum():>15.1f}\n")

    classes = counts.index.tolist()
    values = counts.values.tolist()
    n_bars = len(classes)

    # -------------------------------
    # Colores
    # -------------------------------
    if color_mode.startswith("palette:"):
        cmap_name = color_mode.split(":", 1)[1]
        cmap = plt.cm.get_cmap(cmap_name, n_bars)
        colors = [cmap(i) for i in range(n_bars)]
    elif color_mode.startswith("single:"):
        single = color_mode.split(":", 1)[1]
        colors = [single] * n_bars
    else:
        colors = ["skyblue"] * n_bars

    # -------------------------------
    # Layout
    # -------------------------------
    chart_width = max(8, n_bars * 1.2)
    chart_height = 6
    fontsize = 8
    item_width = 0.25 * fontsize
    usable_width = chart_width * 0.9
    max_cols = int(usable_width / item_width)
    #ncol = min(n_bars, max_cols)
    ncol = 4
    n_rows = int(np.ceil(n_bars / ncol))
    legend_height_ratio = max(0.55, 0.55 * n_rows)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
        "legend.fontsize": 14
    })

    fig = plt.figure(figsize=(chart_width, chart_height))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, legend_height_ratio], hspace=0.5)

    # -------------------------------
    # Gráfico principal
    # -------------------------------
    ax = fig.add_subplot(gs[0])
    bars = ax.bar(range(n_bars), values, color=colors, edgecolor="black", width=0.9)

    ax.set_xlim(-0.5, n_bars - 0.5)
    ax.set_xticks(range(n_bars))
    ax.set_xticklabels(classes, rotation=0, ha="center")

    y_max = max(values)
    ax.set_ylim(0, y_max * 1.15)

    # Etiquetas de porcentaje sobre barras
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=14)

    ax.set_ylabel("Number of flares")
    ax.set_xlabel("Flare Class")
    ax.set_title(f"Distribution of Solar Flares by {column}", pad=20)

    ax.text(0.98, 0.95, f"Total flares: {total_flares}",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # -------------------------------
    # Leyenda inferior
    # ------------------------
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")

    legend_labels = [f"{cls} (n={counts[cls]})" for cls in classes]
    handles = [plt.Line2D([], [], marker="s", markersize=8, linestyle="",
                          color=colors[i], label=legend_labels[i])
               for i in range(n_bars)]

    ax_leg.legend(handles=handles,
                  loc="center",
                  ncol=ncol,
                  frameon=False,
                  title=f"Number of flares per {column}:",
                  title_fontsize=15)

    # Guardar figura
    suffix = f"_{df_label}" if df_label else ""
    output_name = f"flare_distribution_{column}{suffix}.png"
    output_path = os.path.join(graphics_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFile saved to: {output_path}")

    plt.close()

# Función: Filtro de Flares por clases
def filter_flares_by_class(df_flares, include_classes=None, exclude_classes=None, output_dir=None, output_name=None):
    """
    Filtra flares según la letra de la clase o grupo.

    Parameters
    ----------
    df_flares : pd.DataFrame
        DataFrame de flares, debe contener 'ClassLetter' o 'ClassGroup'.
    include_classes : list of str, optional
        Lista de letras o grupos a incluir (p. ej. ['C','M','X']). Si se especifica,
        solo se mantienen estas clases.
    exclude_classes : list of str, optional
        Lista de letras o grupos a excluir (p. ej. ['A','B']). Si se especifica,
        se eliminan estas clases.
    output_dir : str, optional
        Carpeta donde guardar CSV filtrado.
    output_name : str, optional
        Nombre del CSV filtrado.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    df = df_flares.copy()

    # Prioridad: include_classes
    if include_classes is not None:
        df = df[df['ClassLetter'].isin(include_classes)]
    elif exclude_classes is not None:
        df = df[~df['ClassLetter'].isin(exclude_classes)]

    print(f"Flares después del filtro por clase: {len(df)} de {len(df_flares)}")

    # Guardar CSV si se indica
    if output_dir and output_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, output_name)
        df.to_csv(path, index=False)
        print(f"📁 Archivo filtrado por clase guardado: {path}")

    return df

# Función: Filtro de flares solapados
# Función: Filtro de flares solapados considerando start, peak y end
def filter_overlapping_flares(df_flares, start_col="StartTime", peak_col="PeakTime", end_col="EndTime",
                              min_gap_minutes=30, output_dir=None, output_name=None):
    """
    Filtra flares que se solapan o cuyos start, peak o end caen dentro de flares anteriores,
    o cuyo inicio está demasiado cerca del final del flare anterior.
    """

    df = df_flares.copy()

    # Asegurar datetime
    for col in [start_col, peak_col, end_col]:
        df[col] = pd.to_datetime(df[col])

    # Ordenar por inicio
    df = df.sort_values(by=start_col).reset_index(drop=True)

    keep_indices = []
    last_flare_end = None

    for idx, row in df.iterrows():
        if last_flare_end is None:
            keep_indices.append(idx)
            last_flare_end = row[end_col]
        else:
            # Chequear solapamiento completo/parcial
            # Si start, peak o end del flare actual está dentro del rango del flare anterior
            overlap = ((row[start_col] <= last_flare_end) or
                       (row[peak_col] <= last_flare_end and row[peak_col] >= row[start_col]) or
                       (row[end_col] <= last_flare_end and row[end_col] >= row[start_col]))

            # Chequear si el inicio está suficientemente separado del flare anterior
            gap = (row[start_col] - last_flare_end).total_seconds() / 60

            if overlap or gap < min_gap_minutes:
                # Flare eliminado
                continue
            else:
                keep_indices.append(idx)
                last_flare_end = row[end_col]

    df_filtered = df.loc[keep_indices].reset_index(drop=True)
    print(f"Flares después de eliminar solapamientos: {len(df_filtered)} de {len(df_flares)}")

    # Guardar CSV
    if output_dir and output_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, output_name)
        df_filtered.to_csv(path, index=False)
        print(f"📁 Archivo filtrado por solapamiento guardado: {path}")

    return df_filtered

# Función: Filtrar flares al inicio del día
def filter_flares_start_of_day(df_flares, start_col="StartTime", min_start_minutes=60,
                                output_dir=None, output_name=None):
    """
    Elimina flares cuyo inicio ocurre en los primeros minutos del día.

    Parameters
    ----------
    df_flares : pd.DataFrame
        DataFrame de flares con columna de inicio.
    start_col : str
        Nombre de la columna de inicio del flare.
    min_start_minutes : int
        Tiempo mínimo (en minutos) desde el inicio del día permitido para el start del flare.
    output_dir : str, optional
        Carpeta para guardar CSV filtrado.
    output_name : str, optional
        Nombre del CSV filtrado.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """

    df = df_flares.copy()
    df[start_col] = pd.to_datetime(df[start_col])

    # Extraer minutos desde inicio del día
    minutes_from_day_start = (df[start_col] - df[start_col].dt.normalize()).dt.total_seconds() / 60

    # Filtrar
    df_filtered = df[minutes_from_day_start >= min_start_minutes].reset_index(drop=True)
    print(f"Flares después de filtrar los primeros {min_start_minutes} min del día: {len(df_filtered)} de {len(df_flares)}")

    # Guardar CSV
    if output_dir and output_name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, output_name)
        df_filtered.to_csv(path, index=False)
        print(f"📁 Archivo filtrado por inicio del día guardado: {path}")

    return df_filtered
