# main_years_download.py

import os
import pandas as pd
from datetime import datetime, timedelta
from functions_download import Download_Data, running_difference, Calculate_Tem_EM, build_full_dataframe, get_flares, get_active_regions

# --- Crear DataFrame de días ---
def all_dates_dataframe(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_days = pd.DataFrame({'start_time': dates, 'end_time': dates + pd.Timedelta(hours=23, minutes=59, seconds=59)})
    return df_days

# --- Pipeline diario con logs, nc_files y plots ---
def download_goes_flare_AR_data(start_time, end_time, resolution="avg1m", Dif_time=5, plot_diff=True,
                                nc_dir=None, plot_dir=None, month_log=None, year_log=None):
    """
    Descarga diaria de GOES, calcula T/EM, descarga flares y AR.
    """
    try:
        # --- GOES ---
        goes_ts, observatory = Download_Data(start_time, end_time, resolution=resolution, output_dir=nc_dir)
        if goes_ts is None:
            msg = f"No hay datos GOES para {start_time}."
            print(msg)
            if month_log:  open(month_log, "a").write(msg + "\n")
            if year_log:   open(year_log, "a").write(msg + "\n")
            return None

        # --- Restar background y guardar plots ---
        goes_ts_corrected_diff = running_difference(goes_ts, Dif_time=Dif_time, plot=plot_diff,
                                                    block_dir=plot_dir, start_time=start_time)

        # --- Calcular T y EM ---
        temp_em_cor = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='coronal')
        temp_em_phot = Calculate_Tem_EM(goes_ts_corrected_diff, abundance='photospheric')

        # --- Construir df_full ---
        df_full = build_full_dataframe(goes_ts, goes_ts_corrected_diff, temp_em_cor, temp_em_phot,
                                       clip_negative=True, normalize_em=True)

        df_full["observatory"] = observatory if observatory else "Unknown"
        cols = ["observatory"] + [col for col in df_full.columns if col != "observatory"]
        df_full = df_full[cols]

        # --- Descargar flares ---
        df_flare_data = get_flares(start_time, end_time, output_dir=nc_dir)
        if df_flare_data is None:
            df_flare_data = pd.DataFrame()

        # --- Descargar Active Regions ---
        df_AR_data = get_active_regions(start_time, end_time, output_dir=nc_dir)
        if df_AR_data is None:
            df_AR_data = pd.DataFrame()

        return {"df_full": df_full, "df_flare_data": df_flare_data, "df_AR_data": df_AR_data}

    except Exception as e:
        msg = f"❌ Error en {start_time}: {e}"
        print(msg)
        if month_log: open(month_log, "a").write(msg + "\n")
        if year_log:  open(year_log, "a").write(msg + "\n")
        return None

# --- Función principal por año ---
def process_goes_by_year(start_year, end_year, base_output_dir="GOES_data"):
    for year in range(start_year, end_year + 1):
        print(f"\n=== Procesando año {year} ===")
        year_dir = os.path.join(base_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        # ======================================================
        # 🚫 1) SALTAR AÑO COMPLETO SI YA EXISTE df_full_YYYY.csv
        # ======================================================
        annual_full_file = os.path.join(year_dir, f"df_full_{year}.csv")

        if os.path.exists(annual_full_file):
            print(f"⏭️ Año {year} ya procesado anteriormente. Se omite completamente.")
            continue

        # --- Logs anuales ---
        logs_dir = os.path.join(year_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        year_log = os.path.join(logs_dir, f"errores_{year}.log")

        # --- Listas anuales ---
        list_year_full  = []
        list_year_flare = []
        list_year_AR    = []

        # --- Generar días del año ---
        df_days = all_dates_dataframe(f"{year}-01-01", f"{year}-12-31")
        df_days["month"] = df_days["start_time"].dt.strftime("%m")

        dias_sin_datos = []
        total_flares = 0
        total_AR = 0

        # --- Loop por mes ---
        for month, df_month in df_days.groupby("month"):
            month_dir = os.path.join(year_dir, month)
            os.makedirs(month_dir, exist_ok=True)

            # ==========================================================
            # 🚫 2) SALTAR MES COMPLETO SI YA EXISTE df_full_YYYY_MM.csv
            # ==========================================================
            monthly_full_file = os.path.join(month_dir, f"df_full_{year}_{month}.csv")

            if os.path.exists(monthly_full_file):
                print(f"   ⏭️ Mes {year}-{month} ya estaba procesado. Se omite.")
                continue
            
            nc_dir = os.path.join(month_dir, "nc_files")
            os.makedirs(nc_dir, exist_ok=True)
            plot_dir = os.path.join(month_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            month_log = os.path.join(month_dir, f"errores_{year}_{month}.log")

            list_df_full = []
            list_df_flare = []
            list_df_AR = []

            for idx, row in df_month.iterrows():
                start_time = row["start_time"].strftime("%Y-%m-%d %H:%M:%S")
                end_time   = row["end_time"].strftime("%Y-%m-%d %H:%M:%S")

                print(f"\nProcesando {start_time[:10]}...")
                result = download_goes_flare_AR_data(
                    start_time, end_time, resolution="avg1m", Dif_time=5, plot_diff=True,
                    nc_dir=nc_dir, plot_dir=plot_dir,
                    month_log=month_log, year_log=year_log
                )

                if result is None:
                    dias_sin_datos.append(start_time[:10])
                    continue

                # --- Acumulación mensual ---
                if not result["df_full"].empty:
                    list_df_full.append(result["df_full"])
                    list_year_full.append(result["df_full"])

                if not result["df_flare_data"].empty:
                    list_df_flare.append(result["df_flare_data"])
                    list_year_flare.append(result["df_flare_data"])
                    total_flares += len(result["df_flare_data"])

                if not result["df_AR_data"].empty:
                    list_df_AR.append(result["df_AR_data"])
                    list_year_AR.append(result["df_AR_data"])
                    total_AR += len(result["df_AR_data"])

            # --- Guardar CSV mensual ---
            if list_df_full:
                pd.concat(list_df_full).to_csv(
                    os.path.join(month_dir, f"df_full_{year}_{month}.csv"), 
                    index=True
                )

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

        # ================================
        # 🟩 GUARDADO ANUAL COMPLETO
        # ================================
        if list_year_full:
            pd.concat(list_year_full).to_csv(os.path.join(year_dir, f"df_full_{year}.csv"), index=True)

        if list_year_flare:
            pd.concat(list_year_flare).to_csv(os.path.join(year_dir, f"df_flare_data_{year}.csv"), index=False)

        if list_year_AR:
            pd.concat(list_year_AR).to_csv(os.path.join(year_dir, f"df_AR_{year}.csv"), index=False)

        print(f"\n✔️ Archivos anuales guardados en {year_dir}")

        # --- Resumen anual ---
        resumen_file = os.path.join(year_dir, f"resumen_{year}.txt")
        with open(resumen_file, "w") as f:
            f.write(f"Resumen anual GOES {year}\n")
            f.write(f"========================\n")
            f.write(f"Días del año procesados: {len(df_days)}\n")
            f.write(f"Días sin datos GOES: {len(dias_sin_datos)}\n")
            f.write(f"Días con datos GOES: {len(df_days) - len(dias_sin_datos)}\n")
            f.write(f"Número total de flares descargados: {total_flares}\n")
            f.write(f"Número total de AR descargadas: {total_AR}\n")
            if dias_sin_datos:
                f.write(f"Días sin datos: {', '.join(dias_sin_datos)}\n")

        print(f"\n✅ Resumen anual guardado: {resumen_file}")

    print("\n🚀 Descarga completa con archivos anuales y mensuales.")



# --- Ejecución directa ---
if __name__ == "__main__":
    base_output_dir = "GOES_data"
    os.makedirs(base_output_dir, exist_ok=True)
    start_year = 2024
    end_year = 2025

    process_goes_by_year(start_year, end_year, base_output_dir=base_output_dir)


