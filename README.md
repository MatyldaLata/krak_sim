# KRAK-9 – symulacja rakiety na paliwo ciekłe (EuRoC, kategoria L9)

Symulacja trajektorii rakiety KRAK w RocketPy 1.13: silnik ciekły N₂O/etanol (3,5 kN, 32 kN·s),
cel 9000 m AGL. Projekt zawiera lot nominalny, dobór lotek (stateczność + flatter), dobór spadochronów,
analizę wrażliwości i Monte Carlo (200 lotów).

## Uruchomienie

```bash
pip install -r krak_sim/requirements.txt
python -m krak_sim                       # cały potok: sweepy, lot nominalny, Monte Carlo
python -m krak_sim --mode nominal        # tylko lot nominalny (+ checklista wymagań)
python -m krak_sim --mode mc --n 200 --workers 4
python -m krak_sim --profile L3          # ten sam pojazd z częściowym tankowaniem pod 3000 m
```

Uruchamiaj z katalogu **nad** `krak_sim/` (to pakiet Pythona).
Jeśli `pip` nie zbuduje zależności `simplekml` (Python 3.13), zainstaluj ją osobno:
`pip install simplekml --no-build-isolation`. Możesz też zainstalować RocketPy z `--no-deps`.

## Struktura

| Plik | Rola |
|---|---|
| `rocket_params.json` | Wszystkie parametry wejściowe ze źródłem każdego (`_provenance`) |
| `config.py` | Wczytanie parametrów, losowanie próbek MC |
| `rocket_model.py` | Atmosfera (gradient ISA), silnik ciekły (2 zbiorniki), rakieta, spadochrony |
| `monte_carlo.py` | Lot nominalny, metryki, sprawdzanie wymagań EuRoC, Monte Carlo |
| `engineering_sweep.py` | Dobór lotek + flatter (NACA TN 4197), spadochrony, wrażliwość |
| `visualization.py`, `plot_style.py` | Wykresy (wspólny styl, wymagania zaznaczone na każdym wykresie) |

## Wyniki (folder `results/`)

| Plik | Co pokazuje |
|---|---|
| `requirements_check.md` | Tabela wymagań EuRoC: limit, wynik, status |
| `nominal_trajectory.png` | Wysokość w czasie (fazy lotu) i ślad na ziemi |
| `nominal_ascent.png` | Prędkość, przyspieszenie, margines stateczności |
| `nominal_descent.png` | Prędkość opadania na drogue/main vs wysokość |
| `nominal_motor.png` | Krzywa ciągu i masa rakiety |
| `opt_fins_sweep.png` | Rozpiętość lotek → stateczność, apogeum, zapas flatteru |
| `opt_parachute_sweep.png` | Cd·S drogue/main → prędkości opadania |
| `opt_sensitivity_sweep.png` | Wpływ masy, ciągu, krzywej ciągu, Cd, kąta, temperatury, wiatru na apogeum |
| `mc_*.png`, `monte_carlo_*.csv` | Rozkład apogeum, mapa lądowań, odzysk, wrażliwość, spełnienie wymagań |

## Najważniejsze założenia modelu

* Atmosfera: gradient −6,5 K/km od temperatury przy ziemi, tropopauza 216,65 K; wiatr według profilu potęgowego.
* Silnik: ciąg stały z rampami 0,3 s. Impuls liczony z masy **spalonej** (bez 3% pozostałości), Isp 205 s.
* Opór: krzywa Cd(Mach) typowa dla smukłej rakiety z dziobem von Kármán. Nie pochodzi z CFD ani z RASAero.
* Spadochrony: Cd·S stałe, opóźnienie otwarcia, drogue w apogeum, main na 400 m AGL.


