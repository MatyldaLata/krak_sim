"""Analizy projektowe: dobór lotek (stateczność + flatter), dobór spadochronów, wrażliwość apogeum."""
import copy
import logging
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rocketpy import Flight

from . import plot_style as ps
from .config import nominal_sample
from .monte_carlo import make_flight, stability_range
from .rocket_model import build_environment, build_motor, build_rocket

logger = logging.getLogger("KRAK9_SIM.SWEEP")
ps.apply_style()


def _apogee_flight(p, s):
    f, env, rocket, motor = make_flight(p, s, terminate_on_apogee=True, max_time=150)
    return f, env


# ------------------------------------------------------------------ FLATTER (NACA TN 4197)
def flutter_speed(p_fins: dict, pressure: float, sound: float) -> float:
    """Prędkość flatteru lotki trapezowej wg NACA TN 4197 (wzór Martina, uproszczony).

    V_f = a * sqrt( G / ( 1.337 AR^3 P (λ+1) / (2 (AR+2) (t/c)^3) ) )
    """
    cr, ct, b = p_fins["root_chord_m"], p_fins["tip_chord_m"], p_fins["span_m"]
    t = p_fins.get("thickness_m", 0.004)
    G = p_fins.get("shear_modulus_pa", 5e9)
    S = 0.5 * (cr + ct) * b
    AR = b * b / S
    lam = ct / cr
    denom = 1.337 * AR ** 3 * pressure * (lam + 1) / (2 * (AR + 2) * (t / cr) ** 3)
    return sound * math.sqrt(G / denom)


def flutter_margin(p: dict, flight: Flight, env) -> float:
    """Najmniejszy stosunek V_flatter / V_lotu w trakcie wznoszenia."""
    ts = np.linspace(flight.out_of_rail_time, flight.apogee_time, 200)
    best = np.inf
    for t in ts:
        z = flight.z(t)
        v = flight.speed(t)
        if v < 30:
            continue
        vf = flutter_speed(p["rocket"]["fins"], env.pressure(z), env.speed_of_sound(z))
        best = min(best, vf / v)
    return float(best)


# ------------------------------------------------------------------ LOTKI
def sweep_fins(p: dict, out_dir: Path):
    logger.info("Dobór lotek: rozpiętość vs stateczność, apogeum i flatter...")
    s = nominal_sample(p)
    rq = p["requirements"]
    span0 = p["rocket"]["fins"]["span_m"]
    spans = np.round(np.arange(0.08, 0.181, 0.01), 3)
    res = []
    for span in spans:
        pt = copy.deepcopy(p)
        pt["rocket"]["fins"]["span_m"] = float(span)
        env = build_environment(pt, s)
        motor, _ = build_motor(pt, s)
        rocket = build_rocket(pt, s, motor)
        sm0 = float(rocket.static_margin(0))
        if sm0 < 0.5:
            # rakieta niestateczna już na starcie: lot koziołkujący nie ma sensu (i trwa bardzo długo)
            res.append((sm0, sm0, np.nan, np.nan))
            logger.info(f"  rozpiętość {span * 100:4.1f} cm: niestateczna (margines statyczny {sm0:.2f} cal), lot pominięty")
            continue
        f, env = _apogee_flight(pt, s)
        smin, smax = stability_range(f)
        res.append((smin, smax, f.apogee - env.elevation, flutter_margin(pt, f, env)))
        logger.info(f"  rozpiętość {span * 100:4.1f} cm: stateczność {smin:.2f}–{smax:.2f} cal, "
                    f"apogeum {res[-1][2]:.0f} m, zapas flatteru {res[-1][3]:.2f}")
    smin, smax, apo, flut = map(np.array, zip(*res))
    x = spans * 100
    i0 = int(np.argmin(abs(spans - span0)))

    ok = (smin >= rq["stability_min_cal"]) & (smax <= rq["stability_max_cal"]) & (np.nan_to_num(flut) >= 1.5)
    fins = p["rocket"]["fins"]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5), sharex=True)
    ax = axs[0]
    ax.plot(x, smin, marker="o", color=ps.SERIES[0], label="minimum (zejście z szyny)")
    ax.plot(x, smax, marker="s", color=ps.SERIES[2], label="maksimum (lot naddźwiękowy)")
    ax.set_ylim(min(0, np.nanmin(smin) - 0.3), max(np.nanmax(smax), rq["stability_max_cal"]) + 1)
    ps.forbidden_below(ax, rq["stability_min_cal"], f"min. EuRoC {rq['stability_min_cal']} cal")
    ps.forbidden_above(ax, rq["stability_max_cal"], f"maks. projektowe {rq['stability_max_cal']} cal")
    ax.set_title("1. Margines stateczności"); ax.set_ylabel("Margines [kalibry]")
    ax.legend(loc="upper left")

    ax = axs[1]
    ax.plot(x, flut, marker="o", color=ps.SERIES[1])
    ax.set_ylim(0, max(np.nanmax(flut) * 1.1, 2))
    ps.forbidden_below(ax, 1.5, "zalecany zapas 1,5")
    ps.forbidden_below(ax, 1.0, "flatter")
    ps.mark_nominal(ax, x[i0], flut[i0], f"{flut[i0]:.2f}")
    ax.set_title(f"2. Zapas flatteru (NACA TN 4197, t = {fins.get('thickness_m', 0.004) * 1000:.0f} mm, "
                 f"G = {fins.get('shear_modulus_pa', 5e9) / 1e9:.0f} GPa)")
    ax.set_ylabel("V_flatter / V_lotu [–]")

    ax = axs[2]
    ax.plot(x, apo, marker="o", color=ps.SERIES[0])
    ps.target_line(ax, p["mission"]["target_apogee_agl_m"], "cel")
    ps.mark_nominal(ax, x[i0], apo[i0], f"{apo[i0]:.0f} m")
    ax.set_ylim(np.nanmin(apo) - 30, np.nanmax(apo) + 30)
    ax.set_title("3. Apogeum")
    ax.set_ylabel("Apogeum AGL [m]")
    ax.text(0.98, 0.04, "Cd(Mach) jest stałe dla całej rakiety –\nopór samych lotek nie zmienia się z rozpiętością,\n"
            "więc różnice < 20 m wynikają tylko z kątów natarcia.", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=ps.INK2)
    for a in axs:
        a.axvline(x[i0], color=ps.MUTED, ls=":", lw=1.2)
        a.set_xlabel("Rozpiętość lotki [cm]")
        if ok.any():
            a.axvspan(x[ok].min() - 0.5, x[ok].max() + 0.5, color=ps.GOOD, alpha=0.10, lw=0)
    if ok.any():
        axs[0].text((x[ok].min() + x[ok].max()) / 2, axs[0].get_ylim()[0], "okno\nOK", ha="center", va="bottom",
                    color=ps.GOOD, fontsize=10.5, fontweight="bold")
    win = f"{x[ok].min():.0f}–{x[ok].max():.0f} cm" if ok.any() else "brak"
    fig.suptitle(f"{p['rocket']['name']} – dobór rozpiętości lotek (cięciwy {fins['root_chord_m'] * 100:.0f}/"
                 f"{fins['tip_chord_m'] * 100:.0f} cm, skos {fins['sweep_length_m'] * 100:.0f} cm). "
                 f"Zielone pole = wszystkie warunki spełnione: {win}; projekt {span0 * 100:.0f} cm",
                 x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    ps.save(fig, out_dir / "opt_fins_sweep.png")
    logger.info("Zapisano opt_fins_sweep.png")


# ------------------------------------------------------------------ SPADOCHRONY
def sweep_parachutes(p: dict, out_dir: Path):
    logger.info("Dobór spadochronów (drogue i main)...")
    s = nominal_sample(p)
    rq = p["requirements"]
    f, env, rocket, motor = make_flight(p, s)
    m_desc = float(rocket.total_mass(f.t_final))     # masa na spadochronie (sucha + resztki)
    g = p["site"]["gravity_m_s2"]
    elev = env.elevation
    apo = f.apogee
    z_main = elev + p["recovery"]["main"]["deploy_altitude_agl_m"]

    def v_term(cds, z):
        return np.sqrt(2 * m_desc * g / (env.density(z) * cds))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    # Drogue — analitycznie (prędkość graniczna), na wysokości apogeum i otwarcia maina
    cds = np.linspace(0.3, 1.6, 200)
    lo, hi = rq["drogue_speed_range_m_s"]
    a1.plot(cds, v_term(cds, apo), color=ps.SERIES[0], label=f"tuż po otwarciu ({apo - elev:.0f} m)")
    a1.plot(cds, v_term(cds, z_main), color=ps.SERIES[1], label=f"przed otwarciem main ({z_main - elev:.0f} m)")
    a1.set_ylim(10, 80)
    ps.allowed_band(a1, lo, hi, f"dozwolone {lo:.0f}–{hi:.0f} m/s (RQT-0360)")
    c_ok = cds[(v_term(cds, apo) <= hi) & (v_term(cds, z_main) >= lo)]
    if len(c_ok):
        a1.axvspan(c_ok.min(), c_ok.max(), color=ps.GOOD, alpha=0.06, lw=0)
        a1.text((c_ok.min() + c_ok.max()) / 2, 12, f"poprawne Cd·S: {c_ok.min():.2f}–{c_ok.max():.2f} m²",
                ha="center", color=ps.GOOD, fontsize=10.5)
    c0 = p["recovery"]["drogue"]["cd_s_m2"]
    ps.mark_nominal(a1, c0, float(v_term(c0, apo)), f"projekt {c0:.2f} m²: {v_term(c0, apo):.0f} m/s")
    ps.mark_nominal(a1, c0, float(v_term(c0, z_main)), f"{v_term(c0, z_main):.0f} m/s", dy=-20)
    a1.set_title("Spadochron hamujący (drogue)")
    a1.set_xlabel("Cd·S drogue [m²]"); a1.set_ylabel("Prędkość opadania [m/s]")
    a1.legend(loc="upper right")

    # Main — symulacja pełnych lotów + krzywa analityczna
    cds_main = np.linspace(4, 20, 9)
    v_sim = []
    for c in cds_main:
        pt = copy.deepcopy(p)
        pt["recovery"]["main"]["cd_s_m2"] = float(c)
        fl, *_ = make_flight(pt, s)
        v_sim.append(abs(fl.impact_velocity))
    cc = np.linspace(3.5, 20.5, 200)
    a2.plot(cc, v_term(cc, elev), color=ps.SERIES[2], label="prędkość graniczna (analitycznie)")
    a2.plot(cds_main, v_sim, "o", color=ps.SERIES[0], label="symulacja RocketPy")
    a2.set_ylim(0, max(v_sim) * 1.1)
    ps.forbidden_above(a2, rq["main_speed_max_m_s"], f"limit {rq['main_speed_max_m_s']:.0f} m/s (RQT-0380)")
    c0 = p["recovery"]["main"]["cd_s_m2"]
    need = 2 * m_desc * g / (env.density(elev) * rq["main_speed_max_m_s"] ** 2)
    a2.axvline(need, color=ps.CRIT, lw=1, ls=":")
    a2.text(need + 0.2, max(v_sim), f"min. Cd·S = {need:.1f} m²", color=ps.CRIT, fontsize=10.5, va="top")
    ps.mark_nominal(a2, c0, float(v_term(c0, elev)), f"projekt {c0:.0f} m² → {v_term(c0, elev):.1f} m/s\n"
                                                      f"({p['recovery']['main'].get('type', '')})")
    a2.set_title(f"Spadochron główny (masa na spadochronie {m_desc:.1f} kg)")
    a2.set_xlabel("Cd·S main [m²]"); a2.set_ylabel("Prędkość przyziemienia [m/s]")
    a2.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, out_dir / "opt_parachute_sweep.png")
    logger.info("Zapisano opt_parachute_sweep.png")


# ------------------------------------------------------------------ WRAŻLIWOŚĆ
def sweep_sensitivity(p: dict, out_dir: Path):
    logger.info("Analiza wrażliwości apogeum (masa, ciąg, krzywa ciągu, Cd, kąt, atmosfera, wiatr)...")
    s0 = nominal_sample(p)
    target = p["mission"]["target_apogee_agl_m"]
    head = p["launch"]["heading_deg"]

    def apogee(**over):
        s = copy.deepcopy(s0)
        s.update(over)
        f, env = _apogee_flight(p, s)
        return f.apogee - env.elevation

    base = apogee()
    cases = [
        ("Masa konstrukcji", "Zmiana masy [kg]", np.linspace(-5, 5, 9), lambda v: dict(mass_offset_kg=v), 0.0, "{:+.0f} kg"),
        ("Ciąg (impuls) silnika", "Ciąg względem nominalnego [%]", np.linspace(90, 110, 9),
         lambda v: dict(thrust_scale=v / 100), 100.0, "{:.0f}%"),
        ("Kształt krzywej ciągu (ten sam impuls)", "Ciąg końcowy / początkowy [–]", np.linspace(0.5, 1.0, 6),
         lambda v: dict(thrust_end_ratio=v), 1.0, "{:.1f}"),
        ("Napełnienie zbiorników", "Masa propelantów [%]", np.linspace(85, 100, 7),
         lambda v: dict(propellant_scale=v / 100), 100.0, "{:.0f}%"),
        ("Współczynnik oporu", "Cd względem nominalnego [%]", np.linspace(85, 115, 7),
         lambda v: dict(cd_scale=v / 100), 100.0, "{:.0f}%"),
        ("Kąt wyrzutni", "Kąt od poziomu [°]", np.linspace(75, 87, 7),
         lambda v: dict(inclination_deg=v), s0["inclination_deg"], "{:.0f}°"),
        ("Temperatura przy ziemi", "Temperatura [°C]", np.linspace(5, 35, 7),
         lambda v: dict(temperature_k=v + 273.15), s0["temperature_k"] - 273.15, "{:.0f} °C"),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5))
    for ax, (title, xl, xs, fn, x0, fmt) in zip(axs.flat, cases):
        ys = np.array([apogee(**fn(v)) for v in xs])
        ax.plot(xs, ys, marker="o", color=ps.SERIES[0])
        ps.target_line(ax, target, "cel")
        ax.axvline(x0, color=ps.MUTED, ls=":", lw=1.2)
        slope = np.polyfit(xs, ys, 1)[0]
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xl)
        ax.text(0.97, 0.05, f"Δ ≈ {slope:+.0f} m / jedn.\nzakres {ys.min():.0f}–{ys.max():.0f} m",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color=ps.INK2,
                bbox=dict(boxstyle="round,pad=0.3", fc=ps.SURFACE, ec=ps.GRID))
        if title == "Kąt wyrzutni":
            ax.axvspan(83, 85, color=ps.GOOD, alpha=0.08, lw=0)
            ax.text(84, ax.get_ylim()[1], "84±1°", ha="center", va="top", fontsize=10, color=ps.GOOD)
        logger.info(f"  {title}: {ys.min():.0f}–{ys.max():.0f} m")
    # wiatr: z wiatrem i pod wiatr (względem kierunku startu)
    ax = axs.flat[7]
    ws = np.linspace(0, 10, 6)
    for lab, d, c in (("wieje w kierunku startu", head, ps.SERIES[0]), ("wieje przeciwnie", head + 180, ps.SERIES[1])):
        ys = [apogee(ground_wind_speed_m_s=w, wind_direction_deg_toward=d % 360) for w in ws]
        ax.plot(ws, ys, marker="o", color=c, label=lab)
    ps.target_line(ax, target, "cel")
    ax.axvline(s0["ground_wind_speed_m_s"], color=ps.MUTED, ls=":", lw=1.2)
    ax.set_title("Wiatr przy ziemi", fontsize=13); ax.set_xlabel("Prędkość wiatru [m/s]")
    ax.legend(loc="lower left", fontsize=10)
    for ax in axs[:, 0]:
        ax.set_ylabel("Apogeum AGL [m]")
    fig.suptitle(f"{p['rocket']['name']} – wrażliwość apogeum (nominalnie {base:.0f} m; kropkowana pionowa = wartość nominalna)",
                 x=0.01, ha="left")
    fig.tight_layout()
    ps.save(fig, out_dir / "opt_sensitivity_sweep.png")
    logger.info("Zapisano opt_sensitivity_sweep.png")
