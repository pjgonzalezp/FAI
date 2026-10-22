from datetime import datetime, timedelta
import os
import pandas as pd
import time
start_time = time.time() 

def select_all_dates(start_date, end_date, output_dir="."):
    """
    Genera todas las fechas diarias entre start_date y end_date (incluyendo ambos),
    imprime un resumen y guarda un CSV con las fechas.

    Parameters:
    - start_date (str): Fecha inicial 'YYYY-MM-DD'
    - end_date (str): Fecha final 'YYYY-MM-DD'
    - output_dir (str): Carpeta donde guardar el CSV

    Returns:
    - list of str: Lista de fechas en 'YYYY-MM-DD'
    - str: Ruta de la carpeta donde se guardó el CSV
    """
    # Convertir a datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generar todas las fechas
    delta = end - start
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(delta.days + 1)]

    ndays = len(dates)

    # Crear carpeta como en la función original
    output_folder = os.path.join(output_dir, f"Data_for_{ndays}_days")
    os.makedirs(output_folder, exist_ok=True)

    # Guardar CSV
    csv_path = os.path.join(output_folder, f"selected_dates_all_{ndays}.csv")
    pd.DataFrame({"date": dates}).to_csv(csv_path, index=False)

    # Resumen
    print(f"\n--- Selección de fechas (todas) ---")
    print(f"Fecha de inicio: {start_date}")
    print(f"Fecha de fin:    {end_date}")
    print(f"Número total de días: {ndays}")
    print(f"CSV guardado en: {csv_path}\n")

    return dates, output_folder


def select_dates(start_date, end_date, n=10, output_dir="."):
    """
    Selecciona n fechas distribuidas uniformemente entre start_date y end_date,
    imprime un resumen y guarda un CSV con las fechas seleccionadas.

    Parameters:
    - start_date (str): Fecha inicial 'YYYY-MM-DD'
    - end_date (str): Fecha final 'YYYY-MM-DD'
    - n (int): Número de fechas a seleccionar
    - output_dir (str): Carpeta donde guardar el CSV

    Returns:
    - list of str: Lista de fechas en 'YYYY-MM-DD'
    """
    # Convertir a datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_days = (end - start).days

    if n < 2:
        dates = [start_date]
        step = 0
    else:
        step = max(1, int(round(total_days / (n - 1))))
        dates = [start + timedelta(days=i*step) for i in range(n)]

    # Convertir a strings
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]

    # Crear carpeta Data_for_{n}_days fuera de All_Data
    output_folder = os.path.join(output_dir, f"Data_for_{n}_days")
    os.makedirs(output_folder, exist_ok=True)

    # Guardar CSV
    csv_path = os.path.join(output_folder, f"selected_dates_{n}.csv")
    pd.DataFrame({"date": dates_str}).to_csv(csv_path, index=False)

    # Imprimir resumen
    print(f"\n--- Selección de fechas ---")
    print(f"Fecha de inicio: {start_date}")
    print(f"Fecha de fin:    {end_date}")
    print(f"Número de fechas: {n}")
    print(f"Paso entre fechas (días): {step}")
    print(f"CSV con fechas guardado en: {csv_path}\n")

    return dates_str, output_folder


def select_dates_step(start_date, end_date, step_days=10, output_dir="."):
    """
    Selecciona fechas con un paso de 'step_days' entre start_date y end_date,
    imprime un resumen y guarda un CSV con las fechas seleccionadas.

    Parameters:
    - start_date (str): Fecha inicial 'YYYY-MM-DD'
    - end_date (str): Fecha final 'YYYY-MM-DD'
    - step_days (int): Número de días entre cada fecha
    - output_dir (str): Carpeta donde guardar el CSV

    Returns:
    - tuple: (lista de fechas en 'YYYY-MM-DD', carpeta donde se guardó el CSV)
    """
    # Convertir a datetime
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if step_days < 1:
        raise ValueError("step_days debe ser al menos 1")

    # Generar fechas con paso de step_days
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=step_days)

    # Convertir a strings
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]

    # Crear carpeta Data_for_{step_days}_days
    output_folder = os.path.join(output_dir, f"Data_for_{step_days}_days")
    os.makedirs(output_folder, exist_ok=True)

    # Guardar CSV
    csv_path = os.path.join(output_folder, f"selected_dates_{step_days}.csv")
    pd.DataFrame({"date": dates_str}).to_csv(csv_path, index=False)

    # Imprimir resumen
    print(f"\n--- Selección de fechas ---")
    print(f"Fecha de inicio: {start_date}")
    print(f"Fecha de fin:    {end_date}")
    print(f"Paso entre fechas (días): {step_days}")
    print(f"Número de fechas seleccionadas: {len(dates_str)}")
    print(f"CSV con fechas guardado en: {csv_path}\n")

    return dates_str, output_folder

def collect_selected_dates(
        data_dir, 
        selected_dates, 
        file_prefix="df_full",
        date_column="date",
        output_dir_base=None
    ):
    """
    Filtra archivos mensuales {file_prefix}_YYYY_MM.csv y guarda un único CSV
    con solo las filas correspondientes a las fechas seleccionadas en una
    carpeta nueva 'Data_for_{ndays}_days'.

    Parameters
    ----------
    output_dir_base : str, optional
        Carpeta base donde se creará 'Data_for_{ndays}_days'. 
        Si no se proporciona, se usa `data_dir`.
    """
    selected_dates = set(selected_dates)
    print(f"Buscando {len(selected_dates)} fechas en archivos que empiezan con '{file_prefix}'...")

    all_rows = []

    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith(file_prefix + "_") and fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            print(f"Revisando: {fname}")

            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  Error al leer {fname}: {e}")
                continue

            # Si la columna de fecha no existe, intentar usar la primera columna
            if date_column not in df.columns:
                if df.columns[0].startswith("Unnamed"):
                    df.rename(columns={df.columns[0]: date_column}, inplace=True)
                    print(f"  Columna de fecha sin nombre renombrada a '{date_column}'")
                else:
                    print(f"  Advertencia: {fname} no contiene la columna '{date_column}'")
                    continue

            # Convertir a datetime para filtrar correctamente
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df_filtered = df[df[date_column].dt.strftime("%Y-%m-%d").isin(selected_dates)]

            if not df_filtered.empty:
                print(f"  -> {len(df_filtered)} filas seleccionadas")
                all_rows.append(df_filtered)

    if not all_rows:
        print("No se encontró ninguna de las fechas seleccionadas en estos archivos.")
        return None

    final_df = pd.concat(all_rows, ignore_index=True)

    # Crear carpeta de salida Data_for_{n}_days en la ruta que se indique o en data_dir
    ndays = len(selected_dates)
    base_dir = output_dir_base if output_dir_base is not None else data_dir
    output_dir = os.path.join(base_dir, f"Data_for_{ndays}_days")
    os.makedirs(output_dir, exist_ok=True)

    # Guardar archivo final
    output_path = os.path.join(
        output_dir, 
        f"all_{file_prefix}_{ndays}.csv"
    )
    final_df.to_csv(output_path, index=False)

    print(f"\nArchivo final creado en: {output_path}")

    return final_df


data_dir="All_Data"
# Carpeta fuera de All_Data
output_base = "./" # en esta misma carpeta

dates, selected_folder = select_all_dates("1980-01-05", "2025-11-15", output_dir=output_base)
#dates, selected_folder = select_dates_step("1980-01-05", "2025-11-15", step_days=5, output_dir=output_base)

# Para df_full
df_full_selected = collect_selected_dates(
    data_dir=data_dir,
    selected_dates=dates,
    file_prefix="df_full",
    date_column="date",
    output_dir_base=output_base
)

# Para AR
df_ar_selected = collect_selected_dates(
    data_dir=data_dir,
    selected_dates=dates,
    file_prefix="df_AR",
    date_column="StartTime",
    output_dir_base=output_base
)

# Para flares
df_flares_selected = collect_selected_dates(
    data_dir=data_dir,
    selected_dates=dates,
    file_prefix="df_flare_data",
    date_column="PeakTime",
    output_dir_base=output_base
)

end_time = time.time()
elapsed = end_time - start_time

print("\n⏱️ Tiempo total de ejecución:")
mins = int(elapsed // 60)
secs = int(elapsed % 60)

print(f"\n⏱️ Tiempo total: {mins} min {secs} sec")