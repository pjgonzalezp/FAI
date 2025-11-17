# functions_download.py
import os
from sunpy.net import Fido, attrs as a
from sunpy.timeseries import TimeSeries
from sunpy.timeseries.sources.goes import XRSTimeSeries
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pandas as pd
from sunkit_instruments.goes_xrs import calculate_temperature_em
from sunpy.data import manager
import logging

# 1. Resolución:
def ensure_1min_resolution(ts):
    """
    Revisa si el TimeSeries está en resolución de 1 minuto.
    Si no, lo re-muestrea a 1 min con la media.

    Check if the TimeSeries has a resolution of 1 minute.
    If no, it will be shown again in 1 minute with the media
    """
    # Pasar a DataFrame
    df = ts.to_dataframe()
    
    # Calcular la resolución actual (diferencia entre los 2 primeros tiempos)
    current_res = (df.index[1] - df.index[0]).total_seconds()
    
    if abs(current_res - 60) < 1:  # ya es 1 min (tolerancia de 1s)
        print("Resolution = 1 minute")
        return ts
    else:
        print(f"Resolution detected: {current_res:.2f} s → resampling at 1 min")
        df_resampled = df.resample("1min").mean()
        units={'xrsa': u.W/u.m**2, 'xrsb': u.W/u.m**2}
        ts_resampled=XRSTimeSeries(df_resampled, units=units, meta=ts.meta)
        
        return ts_resampled
    
# 2. Descarga datos de rayos X blandos de GOES:
def Download_Data(start_time, end_time, resolution="avg1m", output_dir="GOES_data", log_file=None):

    # ===== Helpers internos =====
    def write_log(message):
        """Escribe mensajes en el log."""
        with open(log_file, "a") as f:
            f.write(message + "\n")

    def extract_satellite(meta):
        """Extrae número de GOES desde platform o telescop."""
        for key in ("platform", "telescop"):
            value = meta.get(key)
            if value:
                digits = "".join(filter(str.isdigit, value))
                if digits:
                    return int(digits)
        return None

    try:
        # ===== Directorios =====
        os.makedirs(output_dir, exist_ok=True)
        data_dir = os.path.join(output_dir, "data")
        error_dir = os.path.join(output_dir, "error")
        graph_dir = os.path.join(output_dir, "data_graphs")

        for d in (data_dir, error_dir, graph_dir):
            os.makedirs(d, exist_ok=True)

        # ===== Log =====
        log_file = os.path.join(error_dir, "errores_goes.log")

        # ===== Validar resolución =====
        if resolution not in ("flx1s", "avg1m"):
            raise ValueError("Resolución inválida. Usa 'flx1s' o 'avg1m'.")

        print(f"Buscando datos GOES: {start_time} a {end_time}")
        result = Fido.search(
            a.Time(start_time, end_time),
            a.Instrument.goes,
            a.Resolution(resolution)
        )

        if len(result[0]) == 0:
            msg = f"No hay datos GOES para {start_time} - {end_time}. Día saltado."
            print(msg)
            write_log(msg)
            return None, None

        # ===== Descargar archivos =====
        files_all = Fido.fetch(result, path=os.path.join(data_dir, "{file}"))

        if len(files_all) == 0:
            msg = f"No se descargaron archivos GOES para {start_time}."
            print(msg)
            write_log(msg)
            return None, None

        # ===== Elegir satélite más reciente usando metadata rápida =====
        sat_files = []
        for f in files_all:
            try:
                # cargar solo el header mínimo
                ts = TimeSeries(f, source="XRS")
                meta = ts.meta.metas[0]
                sat = extract_satellite(meta)
                if sat:
                    sat_files.append((sat, f))
            except Exception as e:
                write_log(f"Error leyendo metadata de {f}: {e}")

        if not sat_files:
            msg = f"No se pudo determinar el satélite para los archivos de {start_time}."
            print(msg)
            write_log(msg)
            return None, None

        latest_sat, latest_file = max(sat_files, key=lambda x: x[0])
        print(f"Satélite más reciente: GOES-{latest_sat}")

        # ===== TimeSeries del archivo correcto =====
        ts = TimeSeries(latest_file, source="XRS")

        # ===== Asegurar resolución 1 min =====
        goes_ts = ensure_1min_resolution(ts)

        # ===== Guardar gráfica =====
        fig, ax = plt.subplots(figsize=(12, 6))
        goes_ts.plot(axes=ax)
        safe_time = start_time.replace(":", "-").replace(" ", "_")
        output_png = os.path.join(graph_dir, f"GOES_{safe_time}.png")
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Gráfica guardada en: {output_png}")

        # ===== Observatorio =====
        meta0 = goes_ts.meta.metas[0]
        sat_final = extract_satellite(meta0)
        observatory = f"GOES-{sat_final}" if sat_final else None

        print(f"Observatorio detectado: {observatory}")

        return goes_ts, observatory

    except Exception as e:
        print(f"Error en Download_Data: {e}")
        write_log(str(e))
        return None, None

# 3. Substracción de backgroun por diferencias:
def running_difference(goes_ts, Dif_time=5, plot=False, block_dir=None, start_time=None):
    """
    Calcula las diferencias de flujo de rayos X GOES a un intervalo definido (default 5 min).
    
    Parámetros
    ----------
    goes_ts : XRSTimeSeries
        Serie temporal original de GOES.
    Dif_time : int, optional
        Intervalo de diferencia en número de pasos para restar flux (default=5).
    plot : bool, optional
        Si True, guarda gráficas comparativas original vs corregido.
    block_dir : str, optional
        Carpeta base donde guardar gráficas.
    start_time : str, optional
        Tiempo inicial usado para nombrar archivos.
        
    Retorna
    -------
    goes_diff_ts : XRSTimeSeries
        Serie temporal corregida con las diferencias.
    """

    import matplotlib.dates as mdates
    from matplotlib.ticker import LogFormatterMathtext

    # --- 1. Extraer datos ---
    df = goes_ts.to_dataframe()
    flux_xrsa = df["xrsa"]
    flux_xrsb = df["xrsb"]
    npts = len(df)

    # --- 2. Calcular diferencias ---
    diffa = np.array(flux_xrsa[Dif_time:]) - np.array(flux_xrsa[:npts - Dif_time])
    diffb = np.array(flux_xrsb[Dif_time:]) - np.array(flux_xrsb[:npts - Dif_time])

    # --- 3. Llenar arreglos completos ---
    diffa_full = np.zeros(npts)
    diffb_full = np.zeros(npts)
    diffa_full[Dif_time:] = diffa
    diffb_full[Dif_time:] = diffb

    # --- 4. Crear DataFrame corregido ---
    df_diff = pd.DataFrame({'xrsa': diffa_full, 'xrsb': diffb_full}, index=df.index)

    # --- 5. Crear TimeSeries corregida ---
    units = {'xrsa': u.W / u.m**2, 'xrsb': u.W / u.m**2}
    goes_diff_ts = XRSTimeSeries(df_diff, units=units, meta=goes_ts.meta)

    # --- 6. Función auxiliar para graficar ---
    def save_plot(df_orig, df_corr, output_file, title="", logscale=False, positive_only=False):
        df_o, df_c = df_orig.copy(), df_corr.copy()
        if positive_only:
            df_o = df_o.clip(lower=1e-9)
            df_c = df_c.clip(lower=1e-9)

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df_o.index, df_o['xrsa'], label='XRSA (original)', color='blue')
        ax.plot(df_c.index, df_c['xrsa'], label='XRSA (corrected)', color='blue', linestyle='--')
        ax.plot(df_o.index, df_o['xrsb'], label='XRSB (original)', color='red')
        ax.plot(df_c.index, df_c['xrsb'], label='XRSB (corrected)', color='red', linestyle='--')

        date_only = df_orig.index[0].strftime("%Y-%m-%d")
        ax.set_xlabel(f"Time (UTC) — {date_only}")
        ax.set_ylabel("Flux [W/m²]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        if logscale:
            ax.set_yscale('log', base=10)
            ax.yaxis.set_major_formatter(LogFormatterMathtext())

        plt.tight_layout()
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved at: {output_file}")

    # --- 7. Guardar gráficas si plot=True ---
    if plot and block_dir is not None and start_time is not None:
        graph_dir = os.path.join(block_dir, "data_graphs")
        os.makedirs(graph_dir, exist_ok=True)

        safe_time = start_time.replace(':','-').replace(' ','_')
        df_corr = df_diff

        # Linear
        save_plot(df, df_corr, os.path.join(graph_dir, f"GOES_diff_linear_{safe_time}.png"),
                  title=f"GOES Data Comparison (Δt={Dif_time} steps)")

        # Logarithmic
        save_plot(df, df_corr, os.path.join(graph_dir, f"GOES_diff_log_{safe_time}.png"),
                  title=f"GOES Data Comparison (Δt={Dif_time} steps) [Log Y]",
                  logscale=True)

        # Positive only
        save_plot(df, df_corr, os.path.join(graph_dir, f"GOES_diff_positive_{safe_time}.png"),
                  title=f"GOES Data Comparison [Positive only, Log Y]",
                  logscale=True,
                  positive_only=True)

    return goes_diff_ts

# 4. Calcular la Temperatura y Medida de emisión con los datos corregidos:
#Calculate_Tem_EM(goes_flare_corrected, abundance='photospheric')
#Calculate_Tem_EM(goes_flare_corrected, abundance='coronal')
def Calculate_Tem_EM(goes_flare_corrected, abundance='coronal'):
    
    """
    Entrada / Input:
        goes_flare_corrected: Objeto TimeSeries corregido que contiene los datos del flare solar en los
        canales XRSA y XRSB. Debe ser compatible con la función `calculate_temperature_em` de SunPy.
        abundance (str): Tipo de abundancia elemental a usar en el cálculo (por defecto: 'coronal').
                         Otras opciones posibles incluyen 'photospheric', dependiendo del modelo de SunPy.

    Salida / Output:
        temp_em: Objeto que contiene la temperatura (T) y medida de emisión (EM) derivadas a partir de los
        datos GOES corregidos.

    Descripción / Description:
        Esta función utiliza los datos corregidos del satélite GOES para calcular la temperatura del plasma
        y la medida de emisión (EM) durante un evento de fulguración solar. Permite especificar el modelo
        de abundancia elemental a utilizar en el cálculo.

        This function uses corrected GOES data to compute the plasma temperature and emission measure (EM)
        during a solar flare. It allows specifying the elemental abundance model to be used in the calculation.

    Notas / Notes:
        - Usa la función `calculate_temperature_em` de SunPy.
        - El parámetro `abundance` controla el modelo de abundancias (por ejemplo, 'coronal' o 'photospheric').
        - Se desactiva temporalmente la verificación del hash de calibración del instrumento.
        - Los datos deben estar previamente corregidos y limpios.
    """

    print(f'Ahora vamos a calcular la T y EM con el modelo de abundancias:{abundance}')
    #  Saltar la verificación del hash temporalmente
    with manager.skip_hash_check():
        #temp_em = calculate_temperature_em(goes_flare_corrected, abundance='coronal')
        temp_em = calculate_temperature_em(goes_flare_corrected, abundance)
    
    print(f'se calculó T y EM con el modelo de abundancias:{abundance}')
    #print(temp_em)
    return temp_em

# 5. Construir dataframe completo y normalizar EM y T:
def build_full_dataframe(goes_ts, goes_corrected, temp_em_cor, temp_em_phot,
                         clip_negative=True, normalize_em=False):
    """
    Combina datos originales, corregidos y parámetros de temperatura/EM
    en un solo DataFrame.

    Parámetros
    ----------
    goes_ts : sunpy.timeseries.TimeSeries
        Serie temporal GOES remuestreada (contiene 'xrsa', 'xrsb').
    goes_corrected : sunpy.timeseries.TimeSeries
        Serie temporal con GOES corregido (xrsa, xrsb corregidos).
    temp_em_cor : sunpy.timeseries.TimeSeries
        Serie temporal con temperatura y EM coronal.
    temp_em_phot : sunpy.timeseries.TimeSeries
        Serie temporal con temperatura y EM fotosférica.
    clip_negative : bool, opcional
        Si True, reemplaza valores negativos con NaN (en lugar de 0).
    normalize_em : bool, opcional
        Si True, normaliza EM a unidades de 1e49 cm^-3.

    Retorna
    -------
    pd.DataFrame
        DataFrame combinado con todas las columnas.
    """

    # Originales
    df_original = goes_ts.to_dataframe()[['xrsa', 'xrsb']]

    # Corregidos
    df_corr = goes_corrected.to_dataframe().rename(
        columns={'xrsa': 'xrsa_corr', 'xrsb': 'xrsb_corr'}
    )

    # Coronal
    df_cor = temp_em_cor.to_dataframe()[['temperature', 'emission_measure']].rename(
        columns={'temperature': 'T_cor', 'emission_measure': 'EM_cor'}
    )

    # Fotosférica
    df_phot = temp_em_phot.to_dataframe()[['temperature', 'emission_measure']].rename(
        columns={'temperature': 'T_phot', 'emission_measure': 'EM_phot'}
    )

    # Combinar todo
    df_full = pd.concat([df_original, df_corr, df_cor, df_phot], axis=1)

    # Opcional: reemplazar valores negativos
    if clip_negative:
        df_full = df_full.mask(df_full < 0, np.nan)   # ahora quedan NaN y no 0

    # Opcional: normalizar EM
    if normalize_em:
        df_full['EM_cor_norm'] = df_full['EM_cor'] / 1e49
        df_full['EM_phot_norm'] = df_full['EM_phot'] / 1e49

    return df_full

# 6. Descargar Flares del HEK:
def get_flares(start_time, end_time, output_dir=None):
    """
    Busca fulguraciones solares reportadas por GOES y SDO en el intervalo dado.
    y guarda un log si no se encuentran flares o si hay errores.
    
    Parameters
    ----------
    start_time : str
        Tiempo inicial (YYYY-MM-DD o compatible con SunPy).
    end_time : str
        Tiempo final (YYYY-MM-DD o compatible con SunPy).
    log_file : str o None, opcional
        Ruta del archivo .log donde guardar mensajes si no hay flares o hay errores.
        Si es None, no se escribe log en disco (solo en consola).

    
    Returns
    -------
    pd.DataFrame or None
        DataFrame con columnas ['StartTime', 'EndTime', 'Class', 'Observatory', 'PeakTime'] 
        o None si no se encontraron flares.
    """
    # Configurar logging básico
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        logging.info(f"Searching for GOES flares between {start_time} and {end_time}...")
        result = Fido.search(a.Time(start_time, end_time), a.hek.FL)

        if not result or len(result) == 0 or len(result[0]) == 0:
            msg = f"No solar flares found between {start_time} and {end_time}."
            logging.info(msg)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                log_path = os.path.join(output_dir, "no_flares.log")
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            return None

        # Filtrar columnas 1D
        names = [name for name in result[0].colnames if len(result[0][name].shape) <= 1]
        table = result[0][names].to_pandas()

        # Seleccionar columnas de interés
        flare_data = table[[

            # Tiempos
            "event_starttime",
            "event_peaktime",
            "event_endtime",

            # Magnitud
            "fl_goescls", 
            "fl_peakflux",

            # Observatorio / instrumento
            "obs_observatory",
            "obs_instrument",

            # Región activa
            "ar_numspots",
            "ar_noaanum", 
            "ar_noaaclass",
            "ar_zurichcls", 
            "ar_mcintoshcls", 
            "ar_mtwilsoncls",

            # Coordenadas
            "hpc_x", "hpc_y",
            "hgs_x", "hgs_y",

            # Metadatos de evento
            "event_type", 
            "eventtype"
        ]]

        # Renombrar columnas
        flare_data.columns = [
            "StartTime", "PeakTime", "EndTime",
            "Class", "PeakFlux",
            "Observatory", "Instrument",
            "NumSpots", "NOAA_AR", "NOAA_Class",
            "Zurich", "McIntosh", "MtWilson",
            "HPC_X", "HPC_Y",
            "HGS_Long", "HGS_Lat",
            "EventType", "EventType2"
        ]

        # Filtrar filas que tengan clase asignada
        flare_data = flare_data[flare_data["Class"].str.strip() != ""]
        flare_data = flare_data.dropna(subset=["Class"])


        # Priorizar observatorio GOES y eliminar duplicados por PeakTime
        flare_data = flare_data.sort_values(by=['PeakTime', 'Observatory'], 
                                            key=lambda x: x.map(lambda v: 0 if v == 'GOES' else 1))
        flare_data = flare_data.drop_duplicates(subset='PeakTime', keep='first')

        logging.info(f"Found {len(flare_data)} GOES solar flares between {start_time} and {end_time}.")
        return flare_data

    except Exception as e:
        msg = f"❌ Error retrieving flares between {start_time} and {end_time}: {e}"
        logging.error(msg)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "no_flares.log")
            with open(log_path, "a") as f:
                f.write(msg + "\n")
        return None

# 7. Descargar AR:
def get_active_regions(start_time, end_time, output_dir=None):
    """
    Descarga regiones activas (AR) desde el HEK filtradas por NOAA SWPC.

    Parameters
    ----------
    start_time : str
        Tiempo inicial (YYYY-MM-DD o compatible con SunPy)
    end_time : str
        Tiempo final (YYYY-MM-DD o compatible con SunPy)
    output_dir : str or None
        Directorio donde guardar logs. Si None, no guarda logs.

    Returns
    -------
    pd.DataFrame or None
        DataFrame con columnas limpias de AR o None si no hay datos.
    """
    import os
    import logging
    import pandas as pd
    from sunpy.net import Fido, attrs as a

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        logging.info(f"Searching HEK Active Regions from {start_time} to {end_time}...")

        # --- 1. Búsqueda HEK usando Fido ---
        result = Fido.search(
            a.Time(start_time, end_time),
            a.hek.AR,
            a.hek.FRM.Name == "NOAA SWPC Observer"
        )

        if len(result) == 0 or len(result[0]) == 0:
            msg = f"No active regions found between {start_time} and {end_time}."
            logging.info(msg)

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                log_path = os.path.join(output_dir, "no_active_regions.log")
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            return None

        tbl = result["hek"]

        # --- 2. Quitar columnas multidimensionales (bbox, chaincode, etc.) ---
        names = [c for c in tbl.colnames if len(tbl[c].shape) <= 1]
        tbl = tbl[names]

        # Convertir a pandas
        df = tbl.to_pandas()

        
        # Seleccionar solo columnas existentes
        keep = [
            "event_starttime", "event_endtime",
            "ar_noaanum", "ar_noaaclass", "ar_zurichcls", "ar_mcintoshcls", "ar_mtwilsoncls",
            "ar_numspots", "ar_area", "ar_numpoints",
            "hpc_x", "hpc_y", "hgs_x", "hgs_y",
            "obs_observatory", "obs_instrument", "event_type"
        ]

        keep = [c for c in keep if c in df.columns]  # filtrar solo las columnas presentes
        df = df[keep]

        # Renombrar automáticamente según las columnas que realmente existen
        rename_dict = {
            "event_starttime": "StartTime",
            "event_endtime": "EndTime",
            "ar_noaanum": "NOAA_AR",
            "ar_noaaclass": "NOAA_Classification",
            "ar_zurichcls": "Zurich",
            "ar_mcintoshcls": "McIntosh",
            "ar_mtwilsoncls": "MtWilson",
            "ar_numspots": "NumSpots",
            "ar_area": "Area",
            "ar_numpoints": "NumPoints",
            "hpc_x": "HPC_X",
            "hpc_y": "HPC_Y",
            "hgs_x": "HGS_Long",
            "hgs_y": "HGS_Lat",
            "obs_observatory": "Observatory",
            "obs_instrument": "Instrument",
            "event_type": "EventType"
        }

        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})

        logging.info(f"Found {len(df)} active regions between {start_time} and {end_time}.")
        return df

    except Exception as e:
        msg = f"❌ Error retrieving active regions between {start_time} and {end_time}: {e}"
        logging.error(msg)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_path = os.path.join(output_dir, "no_active_regions.log")
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        return None

# Este bloque evita que el código se ejecute al importae este módulo
if __name__ == "__main__":
    print("Este archivo solo define funciones, no se ejecuta directamente.")
