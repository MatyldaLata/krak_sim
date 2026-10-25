import argparse
import warnings
import logging
from pathlib import Path
import sys

from .config import load_params, make_samples
from .monte_carlo import run_nominal, run_monte_carlo
from .visualization import generate_report
from .engineering_sweep import sweep_fins, sweep_parachutes, sweep_sensitivity

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logger = logging.getLogger("KRAK9_SIM")
    
    BASE_DIR = Path(__file__).parent
    DEFAULT_PARAMS = BASE_DIR / "rocket_params.json"
    
    ap = argparse.ArgumentParser(description="Symulacja rakiety KRAK-9 (EuRoC L9) w RocketPy")
    # Dodano tryb "all" i ustawiono go jako domyślny
    ap.add_argument("--mode", type=str, default="all", 
                    choices=["all", "mc", "nominal", "sweep-fins", "sweep-chutes", "sweep-params"],
                    help="Wybierz tryb działania (domyślnie 'all' - uruchamia wszystko po kolei)")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS), type=str, help="Ścieżka do pliku JSON")
    ap.add_argument("--n", type=int, default=None, help="Liczba lotów MC")
    ap.add_argument("--workers", type=int, default=4, help="Liczba rdzeni do obliczeń MC")
    ap.add_argument("--out", default="results", type=str, help="Folder wyjściowy na raporty")
    ap.add_argument("--profile", default="L9", choices=["L9", "L3"], help="Profil misji")
    
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Rozpoczęcie symulacji KRAK-9. Wczytany profil: {args.profile}")
    p = load_params(args.params, args.profile)
    
    # Logika potoku (Pipeline)
    if args.mode == "all":
        modes_to_run = ["sweep-fins", "sweep-chutes", "sweep-params", "nominal", "mc"]
        logger.info("Wybrano tryb 'ALL'. Rozpoczynam pełen potok inżynierski...")
    else:
        modes_to_run = [args.mode]

    # Sekwencyjne uruchamianie zadań
    for current_mode in modes_to_run:

        logger.info(f" URUCHAMIANIE ETAPU: {current_mode.upper()} ")

        
        if current_mode == "sweep-fins":
            sweep_fins(p, out_dir)
            
        elif current_mode == "sweep-chutes":
            sweep_parachutes(p, out_dir)
            
        elif current_mode == "sweep-params":
            sweep_sensitivity(p, out_dir)
            
        elif current_mode == "nominal":
            run_nominal(p, out_dir)
            
        elif current_mode == "mc":
            n_runs = args.n or p["monte_carlo"]["n_runs"]
            seed = p["monte_carlo"]["seed"]
            logger.info(f"Przygotowanie Monte Carlo: {n_runs} lotów (seed: {seed})")
            
            samples = make_samples(p, n_runs, seed)
            df_results = run_monte_carlo(p, samples, args.workers)
            
            csv_path = out_dir / "monte_carlo_results.csv"
            df_results.to_csv(csv_path, index=False)
            logger.info(f"Wyniki tabelaryczne zapisano do {csv_path}")
            
            if (df_results["error"] == "").sum() > 0:
                logger.info("Generowanie zaawansowanego raportu wizualnego...")
                generate_report(p, df_results, out_dir)
                logger.info("Raport MC wygenerowany pomyślnie.")
            else:
                logger.error("Brak udanych lotów do wygenerowania raportu MC.")
                
        logger.info(f"\nZAKOŃCZONO ETAP: {current_mode.upper()}\n")

    logger.info("POTOK ZAKOŃCZONY. Wszystkie raporty znajdują się w folderze wyjściowym.")

if __name__ == "__main__":
    main()