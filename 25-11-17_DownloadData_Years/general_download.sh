#!/bin/bash
# ==========================================================
# Script: general_download.sh
# Descripción: Descarga datos por año con timeout y watchdog.
# ==========================================================

# --- Configuración ---
START_YEAR=2020
END_YEAR=2025
PY_SCRIPT="run_download.py"
LOG_FILE="general_log.txt"
TIMEOUT_PER_YEAR="2h"
WATCHDOG_INTERVAL=900   # 15 min sin actividad

echo "=== INICIO DEL PROCESO: $(date) ===" | tee -a "$LOG_FILE"

watchdog() {
    local pid=$1
    local outfile=$2
    local last_size=$(stat -c%s "$outfile" 2>/dev/null || echo 0)
    local elapsed=0

    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        elapsed=$((elapsed + 60))
        local new_size=$(stat -c%s "$outfile" 2>/dev/null || echo 0)
        if [ "$new_size" -eq "$last_size" ]; then
            if [ "$elapsed" -ge "$WATCHDOG_INTERVAL" ]; then
                echo "[$(date)] 🛑 Proceso $pid colgado más de $((WATCHDOG_INTERVAL/60)) min. Se fuerza cierre." | tee -a "$LOG_FILE"
                kill -9 "$pid"
                return 1
            fi
        else
            last_size=$new_size
            elapsed=0
        fi
    done
    return 0
}

# --- Bucle principal ---
for year in $(seq $START_YEAR $END_YEAR); do
    echo "[$(date)] Iniciando descarga para el año $year..." | tee -a "$LOG_FILE"

    OUTFILE="output_${year}.txt"
    ERRFILE="output_error_${year}.txt"

    timeout $TIMEOUT_PER_YEAR python3 "$PY_SCRIPT" "$year" > "$OUTFILE" 2> "$ERRFILE" &
    PY_PID=$!

    watchdog $PY_PID "$OUTFILE"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] ✅ Año $year completado correctamente." | tee -a "$LOG_FILE"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] ⚠️ Año $year excedió el tiempo máximo de $TIMEOUT_PER_YEAR y se detuvo." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ⚠️ Año $year terminó por inactividad o error (código $EXIT_CODE)." | tee -a "$LOG_FILE"
        echo "Revisar archivo: $ERRFILE" | tee -a "$LOG_FILE"
    fi

    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "=== FIN DEL PROCESO: $(date) ===" | tee -a "$LOG_FILE"
