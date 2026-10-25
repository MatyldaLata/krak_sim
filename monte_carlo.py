import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rocketpy import Flight, Environment

from .rocket_model import build_environment, build_motor, build_rocket, motor_numbers
from .config import nominal_sample

logger = logging.getLogger("KRAK9_SIM.MC")


def make_flight(p: dict, s: dict, **kw) -> tuple:
    env = build_environment(p, s)
    motor, _ = build_motor(p, s)
    rocket = build_rocket(p, s, motor)
    flight = Flight(rocket=rocket, environment=env, rail_length=p["launch"]["rail_length_m"],
                    inclination=s["inclination_deg"], heading=s["heading_deg"], **kw)
    return flight, env, rocket, motor


def stability_range(flight: Flight) -> tuple:
    """Min/max marginesu statycznego [kalibry] od zejścia z szyny do apogeum."""
    sm = np.array(flight.stability_margin.source)
    sel = sm[(sm[:, 0] >= flight.out_of_rail_time) & (sm[:, 0] <= flight.apogee_time)]
    if len(sel) == 0:
        return float("nan"), float("nan")
    return float(sel[:, 1].min()), float(sel[:, 1].max())


def descent_speeds(p: dict, flight: Flight, elev: float) -> dict:
    """Prędkości opadania po rozwinięciu drogue (zakres) i maina (ustalona)."""
    sol = np.array(flight.solution)
    t, z, vz = sol[:, 0], sol[:, 3] - elev, -sol[:, 6]
    ev = {e[1].name: e[0] for e in flight.parachute_events}
    out = {"drogue_speed_max_m_s": np.nan, "drogue_speed_min_m_s": np.nan,
           "main_deploy_agl_m": np.nan, "main_speed_m_s": np.nan}
    if "drogue" in ev:
        t_open = ev["drogue"] + p["recovery"]["drogue"].get("lag_s", 0.0) + 5.0   # po ustabilizowaniu
        t_end = ev.get("main", t[-1])
        m = (t >= t_open) & (t <= t_end)
        if m.any():
            out["drogue_speed_max_m_s"] = float(vz[m].max())
            out["drogue_speed_min_m_s"] = float(vz[m].min())
    if "main" in ev:
        out["main_deploy_agl_m"] = float(np.interp(ev["main"], t, z))
        m = t >= ev["main"] + p["recovery"]["main"].get("lag_s", 0.0) + 5.0
        if m.any():
            out["main_speed_m_s"] = float(np.median(vz[m]))
    return out


def extract_metrics(p: dict, flight: Flight, env: Environment) -> dict:
    rq = p["requirements"]
    elev = env.elevation
    x, y = float(flight.x_impact), float(flight.y_impact)
    sm_min, sm_max = stability_range(flight)
    m = {
        "apogee_agl_m": float(flight.apogee - elev),
        "apogee_time_s": float(flight.apogee_time),
        "apogee_x_m": float(flight.apogee_x), "apogee_y_m": float(flight.apogee_y),
        "max_speed_m_s": float(flight.max_speed),
        "max_mach": float(flight.max_mach_number),
        "max_acceleration_m_s2": float(flight.max_acceleration),
        "rail_exit_velocity_m_s": float(flight.out_of_rail_velocity),
        "stability_min_cal": sm_min, "stability_max_cal": sm_max,
        "impact_x_m": x, "impact_y_m": y,
        "drift_m": float(math.hypot(x, y)),
        "landing_speed_m_s": abs(float(flight.impact_velocity)),
    }
    m.update(descent_speeds(p, flight, elev))

    # Flagi wymagań EuRoC (SRD) — każda z numerem wymagania w raporcie
    lo, hi = rq["drogue_speed_range_m_s"]
    m["ok_rail_exit"] = m["rail_exit_velocity_m_s"] > rq["rail_exit_min_m_s"]
    m["ok_stability_min"] = sm_min >= rq["stability_min_cal"]
    m["ok_stability_max"] = sm_max <= rq["stability_max_cal"]
    m["ok_drogue_speed"] = (lo <= m["drogue_speed_min_m_s"]) and (m["drogue_speed_max_m_s"] <= hi)
    m["ok_main_deploy"] = m["main_deploy_agl_m"] <= rq["main_deploy_max_agl_m"] + 1.0
    m["ok_main_speed"] = m["landing_speed_m_s"] < rq["main_speed_max_m_s"]
    m["ok_landing_radius"] = m["drift_m"] <= rq["landing_radius_max_m"]
    return m


def _run_single_flight(args: tuple) -> dict:
    p, s, idx = args
    try:
        flight, env, _, _ = make_flight(p, s)
        out = extract_metrics(p, flight, env)
        out["error"] = ""
    except Exception as e:  # pojedynczy nieudany lot nie zatrzymuje całego MC
        out = {"error": str(e)}
    out.update({"run_id": idx, **{f"in_{k}": v for k, v in s.items()}})
    return out


def run_monte_carlo(p: dict, samples: list, workers: int) -> pd.DataFrame:
    jobs = [(p, s, i) for i, s in enumerate(samples)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_single_flight, j) for j in jobs]
        for k, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if k % 25 == 0 or k == len(jobs):
                logger.info(f"Postęp: {k}/{len(jobs)} symulacji wykonano.")
    return pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)


def requirements_table(p: dict, m: dict, s: dict) -> list:
    """Wiersze: (wymaganie, źródło, limit, wartość, OK?)."""
    rq = p["requirements"]
    mn = motor_numbers(p, s)
    lo, hi = rq["drogue_speed_range_m_s"]
    return [
        ("Impuls całkowity", "CR 4.3", f"≤ {rq['total_impulse_max_ns']:.0f} N·s", f"{mn['impulse']:.0f} N·s",
         mn["impulse"] <= rq["total_impulse_max_ns"]),
        ("Masa ładunku", "RQT-0860", f"≥ {rq['payload_min_kg']:.2f} kg", f"{p['rocket']['payload_mass_kg']:.2f} kg",
         p["rocket"]["payload_mass_kg"] >= rq["payload_min_kg"]),
        ("Kąt startu", "RQT-1000", "84° ± 1°", f"{s['inclination_deg']:.1f}°", abs(s["inclination_deg"] - 84) <= 1),
        ("Prędkość zejścia z szyny", "RQT-1100", f"> {rq['rail_exit_min_m_s']:.0f} m/s",
         f"{m['rail_exit_velocity_m_s']:.1f} m/s", m["ok_rail_exit"]),
        ("Margines stateczności (min)", "RQT-1030", f"≥ {rq['stability_min_cal']:.1f} cal",
         f"{m['stability_min_cal']:.2f} cal", m["ok_stability_min"]),
        ("Margines stateczności (max)", "RQT-1030 (uwaga)", f"≤ {rq['stability_max_cal']:.1f} cal",
         f"{m['stability_max_cal']:.2f} cal", m["ok_stability_max"]),
        ("Drogue: otwarcie w apogeum", "RQT-0350", "trigger = apogee", p["recovery"]["drogue"]["deploy"],
         p["recovery"]["drogue"]["deploy"] == "apogee"),
        ("Drogue: prędkość opadania", "RQT-0360", f"{lo:.0f}–{hi:.0f} m/s",
         f"{m['drogue_speed_min_m_s']:.1f}–{m['drogue_speed_max_m_s']:.1f} m/s", m["ok_drogue_speed"]),
        ("Main: wysokość otwarcia", "RQT-0370", f"≤ {rq['main_deploy_max_agl_m']:.0f} m AGL",
         f"{m['main_deploy_agl_m']:.0f} m", m["ok_main_deploy"]),
        ("Main: prędkość opadania", "RQT-0380", f"< {rq['main_speed_max_m_s']:.0f} m/s",
         f"{m['landing_speed_m_s']:.2f} m/s", m["ok_main_speed"]),
        ("Odległość lądowania", "kryterium zespołu", f"≤ {rq['landing_radius_max_m']:.0f} m",
         f"{m['drift_m']:.0f} m", m["ok_landing_radius"]),
    ]


def run_nominal(p: dict, out_dir: Path) -> dict:
    from .visualization import plot_nominal  # import lokalny: unikamy cykli
    s = nominal_sample(p)
    flight, env, rocket, motor = make_flight(p, s)
    m = extract_metrics(p, flight, env)

    target = p["mission"]["target_apogee_agl_m"]
    logger.info("WYNIKI LOTU NOMINALNEGO ")
    logger.info(f"Apogeum (AGL):            {m['apogee_agl_m']:.1f} m (cel {target} m, różnica {m['apogee_agl_m'] - target:+.0f} m)")
    logger.info(f"Maks. prędkość / Mach:    {m['max_speed_m_s']:.0f} m/s / {m['max_mach']:.2f}")
    rows = requirements_table(p, m, s)
    for name, src, lim, val, ok in rows:
        logger.info(f"  [{'OK ' if ok else 'NIE'}] {name:<30} {val:>16}   (wym. {lim}, {src})")

    md = ["| Wymaganie | Źródło | Limit | Wynik | Status |", "|---|---|---|---|---|"]
    md += [f"| {n} | {src} | {lim} | {val} | {'✅ spełnione' if ok else '❌ NIE'} |" for n, src, lim, val, ok in rows]
    (out_dir / "requirements_check.md").write_text(
        f"# Lot nominalny – zgodność z wymaganiami\n\nApogeum: **{m['apogee_agl_m']:.0f} m AGL** "
        f"(cel {target} m)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame([m]).to_csv(out_dir / "nominal_metrics.csv", index=False)

    plot_nominal(p, flight, env, motor, m, out_dir)
    logger.info("Wykresy lotu nominalnego zapisane (nominal_*.png).")
    return m
