# main_years_download.py

import os
import pandas as pd
from datetime import datetime, timedelta
from functions_download import (
    Download_Data,
    running_difference,
    Calculate_Tem_EM,
    build_full_dataframe,
    get_flares,
    get_active_regions
)

# --------------------------------------------------------
# Crear DataFrame con todos los días del intervalo
# --------------------------------------------------------
def all_dates_dataframe(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_days = pd.DataFrame({
        'start_time': dates,
        'end_time': dates + pd.Timedelta(hours=23, minutes=59, seconds=59)
    })
    return df_days

# --------------------------------------------------------
# Pipeline diario GOES + Flares + AR con manejo de logs
# --------------------------------------------------------
def download_goes_flare_AR_data(start_time, end_time, resolution="avg1m",
                                Dif_time=5, plot_diff=True,
                                nc_dir=None, plot_dir=None,
                                month_log=None, year_log=None):
    """
    Descarga diaria de GOES, calcula T/EM, descarga flares y AR.
    """
    try:
        # --- GOES ---
        goes_ts, observatory = Download_Data(
            start_time, end_time,
            resolution=resolution,
            output_dir=nc_dir
        )

        if goes_ts is None:
            msg = f"No hay datos GOES para {start_time}."
            print(msg)
            if month_log: open(month_log, "a").write(msg + "\n")
            if year_log:  open(year_log, "a").write(msg + "\n")
            return None

        # --- Background removal / running difference ---
        goes_ts_corrected_diff = running_difference(
            goes_ts,
            Dif_time=Dif_time,
            plot=plot_diff,
            block_dir=plot_dir,
            start_time=start_time
        )

        # --- Cálculo de T y EM ---
        temp_em_cor  = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='coronal')
        temp_em_phot = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='photospheric')

        # --- Construir dataframe completo ---
        df_full = build_full_dataframe(
            goes_ts,
            goes_ts_corrected_diff,
            temp_em_cor,
            temp_em_phot,
            clip_negative=True,
            normalize_em=True
        )

        df_full["observatory"] = observatory if observatory else "Unknown"
        # mover columna observatory al inicio
        cols = ["observatory"] + [c for c in df_full.columns if c != "observatory"]
        df_full = df_full[cols]

        # --- Descargar flares ---
        df_flare = get_flares(start_time, end_time, output_dir=nc_dir)
        if df_flare is None:
            df_flare = pd.DataFrame()

        # --- Descargar AR ---
        df_AR = get_active_regions(start_time, end_time, output_dir=nc_dir)
        if df_AR is None:
            df_AR = pd.DataFrame()

        return {
            "df_full": df_full,
            "df_flare_data": df_flare,
            "df_AR_data": df_AR
        }

    except Exception as e:
        msg = f"❌ Error en {start_time}: {e}"
        print(msg)
        if month_log: open(month_log, "a").write(msg + "\n")
        if year_log:  open(year_log, "a").write(msg + "\n")
        return None

# --------------------------------------------------------
# Verificación si un archivo anual está completo (Opción C)
# --------------------------------------------------------
def check_annual_completeness(year_dir, year):
    """
    Verifica si el archivo anual contiene todos los meses disponibles.
    Si falta alguno → returns False.
    """
    annual_file = os.path.join(year_dir, f"df_full_{year}.csv")
    if not os.path.exists(annual_file):
        return False  # No existe el archivo → está incompleto

    df_annual = pd.read_csv(annual_file)
    if "month" not in df_annual.columns:
        return False  # No se puede verificar

    months_present = set(df_annual["month"].astype(str).str.zfill(2))
    
    # meses que existen en carpetas
    month_folders = [
        f for f in os.listdir(year_dir)
        if os.path.isdir(os.path.join(year_dir, f)) and f.isdigit()
    ]
    month_folders = sorted(month_folders)

    # Si falta alguno → incompleto
    for m in month_folders:
        file_month = os.path.join(year_dir, m, f"df_full_{year}_{m}.csv")
        if os.path.exists(file_month) and m not in months_present:
            return False

    return True

# --------------------------------------------------------
# Reconstrucción anual desde archivos mensuales
# --------------------------------------------------------
def rebuild_annual_files(year_dir, year):
    print(f"\n🔄 Reconstruyendo archivos anuales para {year}...")

    files_full = []
    files_flare = []
    files_AR = []

    for month in sorted(os.listdir(year_dir)):
        month_path = os.path.join(year_dir, month)
        if not os.path.isdir(month_path) or not month.isdigit():
            continue

        f_full  = os.path.join(month_path, f"df_full_{year}_{month}.csv")
        f_flare = os.path.join(month_path, f"df_flare_data_{year}_{month}.csv")
        f_AR    = os.path.join(month_path, f"df_AR_{year}_{month}.csv")

        if os.path.exists(f_full):
            df = pd.read_csv(f_full, index_col=0)
            df["month"] = month  # PARA CHEQUEO DE COMPLETITUD
            files_full.append(df)

        if os.path.exists(f_flare):
            files_flare.append(pd.read_csv(f_flare))

        if os.path.exists(f_AR):
            files_AR.append(pd.read_csv(f_AR))

    if files_full:
        pd.concat(files_full).to_csv(
            os.path.join(year_dir, f"df_full_{year}.csv"),
            index=True
        )
        print(f"   ✔ df_full_{year}.csv generado.")

    if files_flare:
        pd.concat(files_flare).to_csv(
            os.path.join(year_dir, f"df_flare_data_{year}.csv"),
            index=False
        )
        print(f"   ✔ df_flare_data_{year}.csv generado.")

    if files_AR:
        pd.concat(files_AR).to_csv(
            os.path.join(year_dir, f"df_AR_{year}.csv"),
            index=False
        )
        print(f"   ✔ df_AR_{year}.csv generado.")

# --------------------------------------------------------
# Procesamiento general año por año
# --------------------------------------------------------
def process_goes_by_year(start_year, end_year, base_output_dir="GOES_data"):

    for year in range(start_year, end_year + 1):
        print(f"\n============================")
        print(f"=== Procesando año {year} ===")
        print("============================")

        year_dir = os.path.join(base_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        annual_full_file = os.path.join(year_dir, f"df_full_{year}.csv")

        # --- Opción C: reconstruir solo si está incompleto ---
        if os.path.exists(annual_full_file) and check_annual_completeness(year_dir, year):
            print(f"⏭️ Archivo anual {year} completo. No se procesarán meses, pero se actualizará/creará el resumen.")
            need_process_months = False
        else:
            print("⚠️ Archivo anual está incompleto o no existe → se procesarán meses faltantes.")
            need_process_months = True

        # =====================================================================
        # PROCESAR DÍAS Y MESES SOLO SI FALTAN DATOS
        # =====================================================================
        if need_process_months:

            # --- Logs del año ---
            logs_dir = os.path.join(year_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            year_log = os.path.join(logs_dir, f"errores_{year}.log")

            # --- Generar días del año ---
            df_days = all_dates_dataframe(f"{year}-01-01", f"{year}-12-31")
            df_days["month"] = df_days["start_time"].dt.strftime("%m")

            dias_sin_datos = []
            total_flares = 0
            total_AR = 0

            # -----------------------------------------------------------------
            # PROCESAMIENTO MENSUAL
            # -----------------------------------------------------------------
            for month, df_month in df_days.groupby("month"):

                month_dir = os.path.join(year_dir, month)
                os.makedirs(month_dir, exist_ok=True)

                monthly_full_file = os.path.join(month_dir, f"df_full_{year}_{month}.csv")

                # Si ya existe, no reprocesar ese mes
                if os.path.exists(monthly_full_file):
                    print(f"   ⏭️ Mes {year}-{month} ya procesado.")
                    continue

                print(f"\n📅 Procesando mes {year}-{month}...")
                nc_dir = os.path.join(month_dir, "nc_files")
                plot_dir = os.path.join(month_dir, "plots")
                os.makedirs(nc_dir, exist_ok=True)
                os.makedirs(plot_dir, exist_ok=True)

                month_log = os.path.join(month_dir, f"errores_{year}_{month}.log")

                list_df_full = []
                list_df_flare = []
                list_df_AR = []

                # -------------------------------------------------------------
                # PROCESAMIENTO DIARIO
                # -------------------------------------------------------------
                for _, row in df_month.iterrows():

                    start_time = row["start_time"].strftime("%Y-%m-%d %H:%M:%S")
                    end_time   = row["end_time"].strftime("%Y-%m-%d %H:%M:%S")

                    print(f"   Día {start_time[:10]}...")

                    result = download_goes_flare_AR_data(
                        start_time, end_time,
                        resolution="avg1m",
                        Dif_time=5,
                        plot_diff=True,
                        nc_dir=nc_dir,
                        plot_dir=plot_dir,
                        month_log=month_log,
                        year_log=year_log
                    )

                    if result is None:
                        dias_sin_datos.append(start_time[:10])
                        continue

                    if not result["df_full"].empty:
                        list_df_full.append(result["df_full"])

                    if not result["df_flare_data"].empty:
                        list_df_flare.append(result["df_flare_data"])
                        total_flares += len(result["df_flare_data"])

                    if not result["df_AR_data"].empty:
                        list_df_AR.append(result["df_AR_data"])
                        total_AR += len(result["df_AR_data"])

                # --- Guardar mensuales ---
                if list_df_full:
                    pd.concat(list_df_full).to_csv(monthly_full_file, index=True)

                if list_df_flare:
                    pd.concat(list_df_flare).to_csv(
                        os.path.join(month_dir, f"df_flare_data_{year}_{month}.csv"),
                        index=False
                    )

                if list_df_AR:
                    pd.concat(list_df_AR).to_csv(
                        os.path.join(month_dir, f"df_AR_{year}_{month}.csv"),
                        index=False
                    )

        # =====================================================================
        # RECONSTRUCCIÓN ANUAL (SIEMPRE)
        # =====================================================================
        rebuild_annual_files(year_dir, year)

        # =====================================================================
        # CREAR O REESCRIBIR RESUMEN ANUAL
        # =====================================================================
        resumen_file = os.path.join(year_dir, f"resumen_{year}.txt")

        if os.path.exists(resumen_file):
            print(f"⚠️ El archivo {resumen_file} ya existía y será reescrito.")
        else:
            print(f"📄 Creando nuevo archivo de resumen: {resumen_file}")

        # Contar días totales del año
        df_days = all_dates_dataframe(f"{year}-01-01", f"{year}-12-31")
        total_dias = len(df_days)

        # Recalcular días con datos desde archivos mensuales
        dias_con_datos_set = set()
        total_flares = 0
        total_AR = 0

        for month in sorted(os.listdir(year_dir)):
            month_path = os.path.join(year_dir, month)
            if not os.path.isdir(month_path) or not month.isdigit():
                continue

            # Días con datos GOES
            monthly_full_file = os.path.join(month_path, f"df_full_{year}_{month}.csv")
            if os.path.exists(monthly_full_file):
                df_m = pd.read_csv(monthly_full_file, index_col=0)
                dias_index = pd.to_datetime(df_m.index, errors="coerce").strftime("%Y-%m-%d")
                dias_con_datos_set.update(dias_index)

            # flares
            file_fl = os.path.join(month_path, f"df_flare_data_{year}_{month}.csv")
            if os.path.exists(file_fl):
                total_flares += len(pd.read_csv(file_fl))

            # AR
            file_AR = os.path.join(month_path, f"df_AR_{year}_{month}.csv")
            if os.path.exists(file_AR):
                total_AR += len(pd.read_csv(file_AR))

        # Días sin datos
        all_days_set = set(df_days["start_time"].dt.strftime("%Y-%m-%d"))
        dias_sin_datos_final = sorted(all_days_set - dias_con_datos_set)

        # --- Guardar archivo resumen ---
        with open(resumen_file, "w") as f:
            f.write(f"Resumen anual GOES {year}\n")
            f.write("=================================\n")
            f.write(f"Días del año: {total_dias}\n")
            f.write(f"Días con datos GOES: {len(dias_con_datos_set)}\n")
            f.write(f"Días sin datos GOES: {len(dias_sin_datos_final)}\n\n")
            f.write(f"Número total de flares: {total_flares}\n")
            f.write(f"Número total de AR: {total_AR}\n")

            if dias_sin_datos_final:
                f.write("\nDías sin datos:\n")
                for d in dias_sin_datos_final:
                    f.write(f"{d}\n")

        print(f"✅ Resumen anual guardado en {resumen_file}")



# --------------------------------------------------------
# Ejecución directa
# --------------------------------------------------------
if __name__ == "__main__":
    base_output_dir = "GOES_data"
    os.makedirs(base_output_dir, exist_ok=True)
    
    start_year = 2024
    end_year   = 2025

    process_goes_by_year(start_year, end_year, base_output_dir=base_output_dir)

