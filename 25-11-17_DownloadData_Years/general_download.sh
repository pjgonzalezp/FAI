#!/bin/bash
# ==========================================================
# Script: general_download.sh
# Descripción: Descarga datos por año con watchdog 15 min.
# ==========================================================

START_YEAR=2015
END_YEAR=2019
PY_SCRIPT="run_download.py"
LOG_FILE="general_log.txt"
TIMEOUT_PER_YEAR="2h"
WATCHDOG_INTERVAL=900   # 15 min de inactividad

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
                echo "[$(date)] 🛑 Proceso $pid colgado >=15 min. Se cierra." | tee -a "$LOG_FILE"
                kill -9 "$pid" 2>/dev/null
                wait "$pid" 2>/dev/null
                return 137
            fi
        else
            last_size=$new_size
            elapsed=0
        fi
    done

    return 0
}

for year in $(seq $START_YEAR $END_YEAR); do
    echo "[$(date)] Iniciando descarga para el año $year..." | tee -a "$LOG_FILE"

    OUTFILE="output_${year}.txt"
    ERRFILE="output_error_${year}.txt"

    timeout $TIMEOUT_PER_YEAR python3 "$PY_SCRIPT" "$year" > "$OUTFILE" 2> "$ERRFILE" &
    PY_PID=$!

    watchdog $PY_PID "$OUTFILE"
    WD_STATUS=$?

    wait $PY_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] ✅ Año $year completado correctamente." | tee -a "$LOG_FILE"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "[$(date)] ⚠️ Timeout anual ($TIMEOUT_PER_YEAR)." | tee -a "$LOG_FILE"
    elif [ $WD_STATUS -eq 137 ]; then
        echo "[$(date)] 🛑 Finalizado por inactividad >=15min." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] ⚠️ Error (EXIT $EXIT_CODE / WD $WD_STATUS)" | tee -a "$LOG_FILE"
    fi

    YEAR_DIR="GOES_data/$year"
    mkdir -p "$YEAR_DIR"
    mv "$OUTFILE" "$YEAR_DIR/" 2>/dev/null
    mv "$ERRFILE" "$YEAR_DIR/" 2>/dev/null

    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "=== FIN DEL PROCESO: $(date) ===" | tee -a "$LOG_FILE"
