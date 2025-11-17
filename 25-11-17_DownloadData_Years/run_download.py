#run_download.py

from main_years_download import process_goes_by_year
import os
import sys

def main():
    # Revisar argumentos
    if len(sys.argv) < 2:
        print("Uso: python run_download.py <AÑO>")
        sys.exit(1)

    year = int(sys.argv[1])

    base_output_dir = "GOES_data"
    os.makedirs(base_output_dir, exist_ok=True)

    # Descargar solo ese año
    process_goes_by_year(year, year, base_output_dir=base_output_dir)

if __name__ == "__main__":
    main()
