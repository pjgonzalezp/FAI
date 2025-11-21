# calculate_fai_utils.py
from datetime import datetime, timedelta
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO


# Función: Calcula los FAI alets
def fai_from_df(df_full: pd.DataFrame,
                fai_temp_range: tuple = (7, 14),
                fai_em_threshold: float = 0.005,
                date_column: str = "date",
                duration: bool = True,
                FAI_duration: int = 3,
                filter_flare_coincidence: bool = True,
                df_flares: pd.DataFrame = None,
                flare_peak_col: str = "peak_time",
                flare_end_col: str = "end_time",
                verbose: bool = True,
                output_dir: str = None
            ) -> dict:
    """
    Calcula los FAI en un DataFrame y opcionalmente guarda los resultados y un resumen en TXT.
    """

    # --- Validaciones y formato de fechas (igual que antes) ---
    required_cols = ["T_cor", "EM_cor_norm", date_column]
    for col in required_cols:
        if col not in df_full.columns:
            raise ValueError(f"Falta la columna requerida: '{col}'")

    if filter_flare_coincidence:
        if df_flares is None:
            raise ValueError("Cuando filter_flare_coincidence=True, debe proporcionar df_flares")
        required_flare_cols = [flare_peak_col, flare_end_col]
        for col in required_flare_cols:
            if col not in df_flares.columns:
                raise ValueError(f"Falta la columna requerida en df_flares: '{col}'")

    df_full = df_full.copy()
    df_full[date_column] = pd.to_datetime(df_full[date_column], errors="coerce")
    if filter_flare_coincidence and df_flares is not None:
        df_flares = df_flares.copy()
        df_flares[flare_peak_col] = pd.to_datetime(df_flares[flare_peak_col], errors="coerce")
        df_flares[flare_end_col] = pd.to_datetime(df_flares[flare_end_col], errors="coerce")

    # --- FAI_alert y FAI_true ---
    df_full["FAI_alert"] = (df_full["T_cor"].between(*fai_temp_range) & 
                            (df_full["EM_cor_norm"] > fai_em_threshold))
    df_full["FAI_true"] = False

    if duration and FAI_duration > 1:
        df_full["delta_min"] = df_full[date_column].diff().dt.total_seconds().div(60).fillna(1)
        df_full["group_id"] = ((df_full["FAI_alert"] != df_full["FAI_alert"].shift()) |
                               (df_full["delta_min"] > 1)).cumsum()
        df_full["duration_from_start"] = df_full.groupby("group_id")[date_column] \
                                             .transform(lambda x: (x - x.iloc[0]).dt.total_seconds()/60)
        df_full["FAI_true"] = df_full["FAI_alert"] & (df_full["duration_from_start"] >= (FAI_duration - 1))

    df_fai_all = df_full[df_full["FAI_alert"]].copy()
    df_fai_true = df_full[df_full["FAI_true"]].copy()
    df_fai_filtered = df_fai_true.copy()

    if filter_flare_coincidence and df_flares is not None and len(df_fai_true) > 0:
        mask_no_flare_coincidence = pd.Series(True, index=df_fai_true.index)
        for _, flare in df_flares.iterrows():
            peak_time = flare[flare_peak_col]
            end_time = flare[flare_end_col]
            if pd.isna(peak_time) or pd.isna(end_time):
                continue
            flare_mask = (df_fai_true[date_column] >= peak_time) & (df_fai_true[date_column] <= end_time)
            mask_no_flare_coincidence &= ~flare_mask
        df_fai_filtered = df_fai_true.loc[mask_no_flare_coincidence].copy()

    if verbose:
        print(f"✅ FAI_alert: {len(df_fai_all)} puntos candidatos")
        if duration and FAI_duration > 1:
            print(f"✅ FAI_true: {len(df_fai_true)} puntos con duración mínima {FAI_duration}")
        if filter_flare_coincidence:
            filtered_count = len(df_fai_true) - len(df_fai_filtered)
            print(f"✅ FAI filtrados por coincidencia con flares: {filtered_count}, quedan {len(df_fai_filtered)}")

    # --- Guardar CSV y resumen TXT ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fai_em_threshold_d = str(fai_em_threshold).replace(".", "d")
        file_full = os.path.join(output_dir, f"df_fai_full_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")
        file_all = os.path.join(output_dir, f"df_fai_all_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")
        file_true = os.path.join(output_dir, f"df_fai_true_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")
        file_filtered = os.path.join(output_dir, f"df_fai_filtered_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")

        df_full.to_csv(file_full, index=False)
        df_fai_all.to_csv(file_all, index=False)
        df_fai_true.to_csv(file_true, index=False)
        df_fai_filtered.to_csv(file_filtered, index=False)

        # Resumen en TXT
        summary_txt = os.path.join(output_dir, f"FAI_summary_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.txt")
        with open(summary_txt, "w") as f:
            f.write(f"FAI Calculation Summary\n")
            f.write(f"--------------------\n")
            f.write(f"Total points in df_full: {len(df_full)}\n")
            f.write(f"FAI_alert points: {len(df_fai_all)}\n")
            f.write(f"FAI_true points: {len(df_fai_true)}\n")
            f.write(f"FAI_filtered points: {len(df_fai_filtered)}\n")
            f.write(f"\nSaved CSVs:\n{file_full}\n{file_all}\n{file_true}\n{file_filtered}\n")
        print(f"\n✅ CSVs guardados en '{output_dir}' y resumen en '{summary_txt}'")

    return {
        "df_fai_full": df_full,
        "df_fai_all": df_fai_all,
        "df_fai_true": df_fai_true,
        "df_fai_filtered": df_fai_filtered
    }

# Función: Calcula la anticipación del FAI - revisa cuáles flares están o no siento anticipados (False negatives)
def anticipation_fai_analysis(
        df_fai_selected,
        df_flare_data,
        start_col="StartTime",
        peak_col="PeakTime",
        end_col="EndTime",
        window_minutes=30,
        max_prev_flare_minutes=180,
        save_csv=False,
        save_summary=False,
        output_dir=None,
        method="default",
        fai_temp_range=(0, 100),
        fai_em_threshold_d=0,
        FAI_duration=30
    ):
    """
    Analiza FAIs alrededor de cada flare y calcula varias métricas.
    Incluye estadísticas globales y tiempos relativos normalizados por StartPeak.
    """
    print("\n=== Ejecutando anticipation_fai_analysis: Flares que están o no siendo anticipados ===")

    # ---------------------------------------------------------
    # 0) VALIDACIONES INICIALES
    # ---------------------------------------------------------

    # DF flare vacío
    if df_flare_data is None or len(df_flare_data) == 0:
        print("❌ ERROR: df_flare_data está vacío. No se puede realizar el análisis.")
        return pd.DataFrame(), None

    # DF FAI vacío
    if df_fai_selected is None or len(df_fai_selected) == 0:
        print("⚠️ Advertencia: df_fai_selected está vacío. No habrá anticipación posible.")
        # Igual devolvemos los flares sin columnas extra
        return df_flare_data.copy(), None
    print(f"✔️ FAIs recibidos: {len(df_fai_selected)}")
    # Columnas obligatorias
    required_cols = [start_col, peak_col, end_col]
    missing = [c for c in required_cols if c not in df_flare_data.columns]

    if missing:
        print(f"❌ ERROR: faltan columnas obligatorias en df_flare_data: {missing}")
        return pd.DataFrame(), None

    if "date" not in df_fai_selected.columns:
        print("❌ ERROR: df_fai_selected debe contener una columna llamada 'date'.")
        return pd.DataFrame(), None

    # ---------------------------------------------------------
    # 1) PREPARACIÓN
    # ---------------------------------------------------------
    df = df_flare_data.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[peak_col] = pd.to_datetime(df[peak_col])
    df[end_col] = pd.to_datetime(df[end_col])
    fai_times = pd.to_datetime(df_fai_selected["date"]).sort_values()

    df = df.sort_values(peak_col).reset_index(drop=True)

    # Diccionario de listas
    results = {
        "FAIalerts_W": [],
        "FAIalerts_WStart": [],
        "FAIalerts_WPeak": [],
        "FAIalerts_StartPeak": [],
        "FAIalerts_PeakEnd": [],
        "FAIalerts_startEnd": [],
        "AnticipationStart": [],
        "AnticipationPeak": [],
        "Peak_to_lastFAI": [],
        "Time_since_prev_flare_end": [],
        "Time_since_prev_flare_peak": []
    }

    # ---------------------------------------------------------
    # 2) LOOP PRINCIPAL: cálculo por flare
    # ---------------------------------------------------------
    for i, row in df.iterrows():
        start_t = row[start_col]
        peak_t = row[peak_col]
        end_t = row[end_col]

        prev_end = df.loc[i-1, end_col] if i > 0 else pd.Timestamp.min   # flare anterior: EndTime
        prev_peak = df.loc[i-1, peak_col] if i > 0 else pd.Timestamp.min # flare anterior: PeakTime

        #lower_limit = max(prev_end, peak_t - timedelta(minutes=window_minutes))
        lower_limit = max(prev_end, start_t - timedelta(minutes=window_minutes))  # límite inferior end del flare anterior o 30 min antes del start
        #lower_limit = max(prev_peak, peak_t - timedelta(minutes=window_minutes))  # límite inferior peak del flare anterior
        #lower_limit =  peak_t - timedelta(minutes=window_minutes)                # límite inferior 30 min antes del peak
        upper_limit = end_t

        valid_fais = fai_times[(fai_times >= lower_limit) & (fai_times <= upper_limit)]

        # Conteos
        w_total = len(valid_fais)
        w_start = len(valid_fais[valid_fais < start_t])
        w_peak = len(valid_fais[valid_fais < peak_t])
        start_peak = len(valid_fais[(valid_fais >= start_t) & (valid_fais < peak_t)])
        peak_end = len(valid_fais[(valid_fais >= peak_t) & (valid_fais <= end_t)])
        start_end = len(valid_fais[(valid_fais >= start_t) & (valid_fais <= end_t)])

        results["FAIalerts_W"].append(w_total)
        results["FAIalerts_WStart"].append(w_start)
        results["FAIalerts_WPeak"].append(w_peak)
        results["FAIalerts_StartPeak"].append(start_peak)
        results["FAIalerts_PeakEnd"].append(peak_end)
        results["FAIalerts_startEnd"].append(start_end)

        # Anticipaciones
        if not valid_fais.empty:
            earliest_fai = valid_fais.min()
            last_fai = valid_fais.max()

            anticipation_start = (start_t - earliest_fai).total_seconds()/60 if earliest_fai < start_t else None
            anticipation_peak = (peak_t - earliest_fai).total_seconds()/60 if earliest_fai < peak_t else None
            peak_to_last_fai = (peak_t - last_fai).total_seconds()/60  # puede ser negativo
        else:
            anticipation_start = anticipation_peak = peak_to_last_fai = None

        results["AnticipationStart"].append(anticipation_start)
        results["AnticipationPeak"].append(anticipation_peak)
        results["Peak_to_lastFAI"].append(peak_to_last_fai)

        # Tiempo desde flare anterior
        if i > 0:
            delta_end = (peak_t - prev_end).total_seconds()/60
            delta_peak = (peak_t - prev_peak).total_seconds()/60
            time_since_prev_end = delta_end if delta_end <= max_prev_flare_minutes else None
            time_since_prev_peak = delta_peak if delta_peak <= max_prev_flare_minutes else None
        else:
            time_since_prev_end = None
            time_since_prev_peak = None

        # Guardar los valores en results
        results["Time_since_prev_flare_end"].append(time_since_prev_end)
        results["Time_since_prev_flare_peak"].append(time_since_prev_peak)

    # Añadir columnas al DF
    for col, values in results.items():
        df[col] = values

    # ---------------------------------------------------------
    # 3) CÁLCULO DE TIEMPOS RELATIVOS (normalizados por el rise time StartPeak)
    # ---------------------------------------------------------
    if "StartPeak" in df.columns:
        df["RelAnticipation_Peak"] = df["AnticipationPeak"] / df["StartPeak"]
        df["RelAnticipation_Start"] = df["AnticipationStart"] / df["StartPeak"]
        print("✅ Columnas 'RelAnticipation_Peak' y 'RelAnticipation_Start' añadidas (Δt / StartPeak).")
    else:
        print("⚠️ No se encontró la columna 'StartPeak'. No se calcularon tiempos relativos.")

    # ---------------------------------------------------------
    # 4) ESTADÍSTICAS GLOBALES
    # ---------------------------------------------------------
    total_fais = len(df_fai_selected)
    total_flares = len(df)
    total_fais_in_windows = df["FAIalerts_W"].sum()

    flares_with_fais = (df["FAIalerts_W"] > 0).sum()
    flares_without_fais = total_flares - flares_with_fais  # falsos negativos = dlares sin FAI
    flares_with_fai_before_start = (df["FAIalerts_WStart"] > 0).sum()
    flares_without_fai_before_start = total_flares - flares_with_fai_before_start
    flares_with_fai_before_peak = (df["FAIalerts_WPeak"] > 0).sum()
    flares_without_fai_before_peak = total_flares - flares_with_fai_before_peak

    pct_flares_with_fais = 100 * flares_with_fais / total_flares if total_flares > 0 else 0
    pct_flares_without_fais = 100 * flares_without_fais / total_flares if total_flares > 0 else 0

    pct_flares_with_fai_before_start = 100 * flares_with_fai_before_start / total_flares if total_flares > 0 else 0
    pct_flares_without_fai_before_start = 100 * flares_without_fai_before_start / total_flares if total_flares > 0 else 0
    pct_flares_with_fai_before_peak = 100 * flares_with_fai_before_peak / total_flares if total_flares > 0 else 0
    pct_flares_without_fai_before_peak = 100 * flares_without_fai_before_peak / total_flares if total_flares > 0 else 0

    # Totales por subventana
    fai_WStart = df["FAIalerts_WStart"].sum()
    fai_WPeak = df["FAIalerts_WPeak"].sum()
    fai_StartPeak = df["FAIalerts_StartPeak"].sum()
    fai_PeakEnd = df["FAIalerts_PeakEnd"].sum()
    fai_StartEnd = df["FAIalerts_startEnd"].sum()

    def pct(x): return 100 * x / total_fais if total_fais > 0 else 0

    mean_fais_per_flare = df["FAIalerts_W"].mean()
    anticipations = df["AnticipationPeak"].dropna()
    mean_anticipation = anticipations.mean() if not anticipations.empty else np.nan
    min_anticipation = anticipations.min() if not anticipations.empty else np.nan
    max_anticipation = anticipations.max() if not anticipations.empty else np.nan

    # ---------------------------------------------------------
    # 5) GUARDAR CSV
    # ---------------------------------------------------------
    if save_csv:
        if output_dir is None:
            raise ValueError("Debe proporcionar 'output_dir' si save_csv=True")
        file_csv = os.path.join(output_dir,
            f"df_anticipation_time_{method}_(W_{window_minutes})_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")
        df.to_csv(file_csv, index=False)
        print(f"💾 CSV guardado en:\n   {file_csv}")

    # ---------------------------------------------------------
    # 6) GUARDAR RESUMEN TXT
    # ---------------------------------------------------------
    if save_summary:
        if output_dir is None:
            raise ValueError("Debe proporcionar 'output_dir' si save_summary=True")
        summary_file = os.path.join(output_dir,
            f"summary_flare_anticipation_{method}_(W_{window_minutes})_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv")
        with open(summary_file, "w") as f:
            f.write("📊 RESUMEN DE ANÁLISIS FAI–FLARE\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total flares analizados: {total_flares}\n")
            f.write(f"Total FAIs analizados: {total_fais}\n")
            f.write(f"FAIs en ventanas: {total_fais_in_windows} ({pct(total_fais_in_windows):.1f}%)\n")
            f.write(f"Promedio de FAIs por flare: {mean_fais_per_flare:.2f}\n\n")
            f.write(f"Flares con ≥1 FAI: {flares_with_fais} ({pct_flares_with_fais:.1f}%)\n")
            f.write(f"Flares SIN FAI (falsos negativos): {flares_without_fais} ({pct_flares_without_fais:.1f}%)\n")
            f.write(f"Flares con FAI antes del inicio: {flares_with_fai_before_start} ({pct_flares_with_fai_before_start:.1f}%)\n")
            f.write(f"Flares SIN FAI antes del inicio: {flares_without_fai_before_start} ({pct_flares_without_fai_before_start:.1f}%)")
            f.write(f"Flares con FAI antes del pico: {flares_with_fai_before_peak} ({pct_flares_with_fai_before_peak:.1f}%)\n")
            f.write(f"Flares SIN FAI antes del pico: {flares_without_fai_before_peak} ({pct_flares_without_fai_before_peak:.1f}%)")
            f.write("\nDistribución de FAIs por ventana:\n")
            f.write(f"  • Antes del inicio (WStart): {fai_WStart} ({pct(fai_WStart):.1f}%)\n")
            f.write(f"  • Antes del pico (WPeak): {fai_WPeak} ({pct(fai_WPeak):.1f}%)\n")
            f.write(f"  • Entre inicio(inclusive) y pico: {fai_StartPeak} ({pct(fai_StartPeak):.1f}%)\n")
            f.write(f"  • Entre pico(inclusive) y fin(inclusive): {fai_PeakEnd} ({pct(fai_PeakEnd):.1f}%)\n")
            f.write(f"  • Entre inicio(inclusive) y fin(inclusive): {fai_StartEnd} ({pct(fai_StartEnd):.1f}%)\n")
            if not anticipations.empty:
                f.write(f"\nAnticipación media (min): {mean_anticipation:.1f}\n")
                f.write(f"Anticipación mínima (min): {min_anticipation:.1f}\n")
                f.write(f"Anticipación máxima (min): {max_anticipation:.1f}\n")
        print(f"📝 Resumen guardado en:\n   {summary_file}")

    return df, pct_flares_without_fais

# Función: Revisa cuales FAIs están asociados o no a flares (False positives)
def associate_fai_to_flare_dataframes( df_fai_selected, 
                                        df_flares, 
                                        window_minutes=30, 
                                        include_inside=True,
                                        save_csv=True,
                                        save_summary=True,
                                        output_dir=None,
                                        method=None,
                                        fai_temp_range=None,
                                        fai_em_threshold_d=None,
                                        FAI_duration=None
                                    ):
    """
    Asocia FAIs a flares y guarda el resultado en:
    - analysis_esp/fai_flare_association.csv
    - analysis_esp/fai_flare_association_report.txt
    """
    print("\n=== Ejecutando associate_fai_to_flare_dataframes: FAIs que están o no siendo asociados a flares ===")
    # Función interna para imprimir y guardar logs
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    df_fai = df_fai_selected.copy()
    df_flares_copy = df_flares.copy()

    # Detectar la columna de tiempo FAI
    time_col_fai = None
    for col in ['date', 'Unnamed: 0']:
        if col in df_fai.columns:
            time_col_fai = col
            break
    if time_col_fai is None:
        raise ValueError("No se pudo identificar la columna de tiempo en df_fai_selected")

    log(f"Usando columna de tiempo FAI: {time_col_fai}")

    # Convertir tiempos a datetime
    df_fai['Time_FAI'] = pd.to_datetime(df_fai[time_col_fai])
    for c in ['StartTime', 'PeakTime', 'EndTime']:
        df_flares_copy[c] = pd.to_datetime(df_flares_copy[c])

    # Inicializar columnas nuevas
    for col in [
        'Associated_Flare', 'Flare_ID', 'F_StartTime', 'F_PeakTime', 'F_EndTime',
        'F_Class', 'F_ClassLetter', 'F_ClassNumber', 'F_ClassGroup',
        'F_Observatory', 'F_StartPeak', 'F_PeakEnd', 'F_StartEnd',
        'Association_Type', 'Time_to_flare', 'FAI_to_start',
        'FAI_to_peak', 'FAI_to_end']:
        df_fai[col] = None
    df_fai['Associated_Flare'] = False

    window = pd.Timedelta(minutes=window_minutes)
    associated_count = 0

    log(f"Procesando {len(df_fai)} alertas FAI...")

    # --- Bucle principal ---
    for idx, row in df_fai.iterrows():
        fai_time = row['Time_FAI']
        flare = None
        ref_type = None

        # 1) Buscar flare cuyo StartTime esté después del FAI
        mask_start = (df_flares_copy['StartTime'] >= fai_time) & \
                     (df_flares_copy['StartTime'] <= fai_time + window)
        candidate_start = df_flares_copy[mask_start].sort_values('StartTime')

        if not candidate_start.empty:
            flare = candidate_start.iloc[0]
            ref_type = "StartTime"

        else:
            # 2) Buscar flare cuyo PeakTime esté después del FAI
            mask_peak = (df_flares_copy['PeakTime'] >= fai_time) & \
                        (df_flares_copy['PeakTime'] <= fai_time + window)
            candidate_peak = df_flares_copy[mask_peak].sort_values('PeakTime')

            if not candidate_peak.empty:
                flare = candidate_peak.iloc[0]
                ref_type = "PeakTime"

            else:
                # 3) Buscar flare cuyo EndTime esté después del FAI
                mask_end = (df_flares_copy['EndTime'] >= fai_time) & \
                           (df_flares_copy['EndTime'] <= fai_time + window)
                candidate_end = df_flares_copy[mask_end].sort_values('EndTime')

                if not candidate_end.empty:
                    flare = candidate_end.iloc[0]
                    ref_type = "EndTime"

                # 4) FAI dentro de un flare activo
                elif include_inside:
                    mask_inside = (df_flares_copy['StartTime'] <= fai_time) & \
                                   (df_flares_copy['EndTime'] >= fai_time)
                    candidate_inside = df_flares_copy[mask_inside].sort_values('StartTime')

                    if not candidate_inside.empty:
                        flare = candidate_inside.iloc[0]
                        ref_type = "Inside"

        # Si no hay flare → continuar
        if flare is None:
            continue

        # Asociar información
        df_fai.at[idx, 'Associated_Flare'] = True
        df_fai.at[idx, 'Flare_ID'] = flare['Flare_ID']
        df_fai.at[idx, 'F_StartTime'] = flare['StartTime']
        df_fai.at[idx, 'F_PeakTime'] = flare['PeakTime']
        df_fai.at[idx, 'F_EndTime'] = flare['EndTime']
        df_fai.at[idx, 'F_Class'] = flare['Class']
        df_fai.at[idx, 'F_ClassLetter'] = flare['ClassLetter']
        df_fai.at[idx, 'F_ClassNumber'] = flare['ClassNumber']
        df_fai.at[idx, 'F_ClassGroup'] = flare['ClassGroup']
        df_fai.at[idx, 'F_Observatory'] = flare['Observatory'] if 'Observatory' in flare else None
        df_fai.at[idx, 'F_StartPeak'] = flare['StartPeak']
        df_fai.at[idx, 'F_PeakEnd'] = flare['PeakEnd']
        df_fai.at[idx, 'F_StartEnd'] = flare['StartEnd']
        df_fai.at[idx, 'Association_Type'] = ref_type

        # Calcular tiempos relativos (minutos)
        df_fai.at[idx, 'FAI_to_start'] = (flare['StartTime'] - fai_time).total_seconds() / 60
        df_fai.at[idx, 'FAI_to_peak'] = (flare['PeakTime'] - fai_time).total_seconds() / 60
        df_fai.at[idx, 'FAI_to_end'] = (flare['EndTime'] - fai_time).total_seconds() / 60

        if ref_type == "StartTime":
            df_fai.at[idx, 'Time_to_flare'] = df_fai.at[idx, 'FAI_to_start']
        elif ref_type == "PeakTime":
            df_fai.at[idx, 'Time_to_flare'] = df_fai.at[idx, 'FAI_to_peak']
        elif ref_type == "EndTime":
            df_fai.at[idx, 'Time_to_flare'] = df_fai.at[idx, 'FAI_to_end']
        else:  # Inside
            df_fai.at[idx, 'Time_to_flare'] = df_fai.at[idx, 'FAI_to_peak']  # me va a mostrar el tiempo del FAI al peak

        associated_count += 1

    # --- Estadísticas ---
    total_fai = len(df_fai)
    pct = (associated_count / total_fai) * 100
    false_positives = total_fai - associated_count
    pct_false_positives = (false_positives / total_fai) * 100


    log("\n--- Estadísticas de Asociación FAI-Flare ---")
    log(f"Total FAIs: {total_fai}")
    log(f"FAIs asociados: {associated_count} ({pct:.1f}%)")
    log(f"FAIs NO asociados (falsos positivos): {false_positives} ({pct_false_positives:.1f}%)")
    log(f"Ventana: {window_minutes} minutos hacia adelante")
    log(f"Incluir FAIs dentro de flares activos: {include_inside}")

    log("\nDistribución por tipo de asociación:")
    log(str(df_fai['Association_Type'].value_counts(dropna=True)))
    
    valid_times = df_fai['Time_to_flare'].dropna() # Calcular distribución de tiempos a flare
    if associated_count > 0 and valid_times.size > 0:
        log(f"Tiempo medio a flare: {valid_times.mean():.1f} min")
        log(f"Tiempo mínimo: {valid_times.min():.1f} min")
        log(f"Tiempo máximo: {valid_times.max():.1f} min")
        # Distribución por clase de flare
        class_dist = df_fai[df_fai['Associated_Flare']]['F_Class'].value_counts()
        log("\nDistribución por clase:")
        for cls, n in class_dist.items():
            log(f"  {cls}: {n}")

    # Guardar archivo CSV
    if save_csv and output_dir:
        file_fai_assoc = os.path.join(
            output_dir,
            f"df_fai_assoc_{method}_(W_{window_minutes})_T{fai_temp_range[0]}-{fai_temp_range[1]}_EM{fai_em_threshold_d}_dur{FAI_duration}min.csv"
        )
        df_fai.to_csv(file_fai_assoc, index=False)
        log(f"\n✅ DataFrame guardado en:\n{file_fai_assoc}")
    
    # Guardar archivo resumen TXT
    if save_summary and output_dir:
        summary_file = os.path.join(
            output_dir,
            f"Summary_assoc_{method}_W{window_minutes}.txt"
        )
        with open(summary_file, "w") as f:
            for line in logs:
                f.write(line + "\n")
        log(f"📄 Resumen guardado en:\n{summary_file}")

    return df_fai, pct_false_positives
