"""Wykresy: lot nominalny i raport Monte Carlo. Każdy wykres = jedno pytanie, osobny plik PNG."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from . import plot_style as ps

ps.apply_style()

NAMES = {  # czytelne nazwy wejść MC
    "ground_wind_speed_m_s": "Wiatr przy ziemi",
    "wind_direction_deg_toward": "Kierunek wiatru",
    "upper_wind_factor": "Wiatr na wysokości (mnożnik)",
    "temperature_k": "Temperatura powietrza",
    "inclination_deg": "Kąt wyrzutni",
    "heading_deg": "Azymut startu",
    "thrust_scale": "Ciąg silnika",
    "propellant_scale": "Masa propelantów",
    "mass_offset_kg": "Masa konstrukcji",
    "com_offset_m": "Położenie środka masy",
    "cd_scale": "Współczynnik oporu Cd",
    "drogue_cds_scale": "Cd·S drogue",
    "main_cds_scale": "Cd·S main",
}


def _src(f):
    a = np.array(f.source)
    return a[:, 0], a[:, 1]


# ============================================================ LOT NOMINALNY
def plot_nominal(p, flight, env, motor, m, out_dir: Path):
    name = p["rocket"]["name"]
    elev = env.elevation
    rq = p["requirements"]
    t_burn = motor.burn_out_time
    ev = {e[1].name: e[0] for e in flight.parachute_events}
    t_apo, t_end = flight.apogee_time, flight.t_final
    t_rail = flight.out_of_rail_time

    # ---------- 1. Profil wysokości + ślad na ziemi
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw=dict(width_ratios=[1.35, 1]))
    t, z = _src(flight.z)
    a1.plot(t, z - elev, color=ps.SERIES[0])
    phases = [(0, t_burn, "Praca silnika", ps.SERIES[1]), (t_burn, t_apo, "Lot bierny", ps.MUTED),
              (t_apo, ev.get("main", t_end), "Drogue", ps.SERIES[2]), (ev.get("main", t_end), t_end, "Main", ps.SERIES[6])]
    ymax = (z - elev).max() * 1.12
    from matplotlib.patches import Patch
    handles = []
    for t0, t1, lab, c in phases:
        a1.axvspan(t0, t1, color=c, alpha=0.10, lw=0)
        handles.append(Patch(color=c, alpha=0.35, label=f"{lab} ({t0:.0f}–{t1:.0f} s)"))
    a1.legend(handles=handles, loc="upper right", title="Fazy lotu", title_fontsize=11)
    a1.set_ylim(0, ymax)
    a1.set_xlim(0, t_end)
    ps.mark_nominal(a1, t_apo, m["apogee_agl_m"], f"Apogeum {m['apogee_agl_m']:.0f} m  (t = {t_apo:.1f} s)")
    if "main" in ev:
        ps.mark_nominal(a1, ev["main"], m["main_deploy_agl_m"], f"Main {m['main_deploy_agl_m']:.0f} m", dx=-10, dy=10)
        a1.texts[-1].set_ha("right")
    a1.set_title("Wysokość nad poziomem startu")
    a1.set_xlabel("Czas [s]")
    a1.set_ylabel("Wysokość AGL [m]")

    _, x = _src(flight.x)
    _, y = _src(flight.y)
    r_max = rq["landing_radius_max_m"]
    a2.add_patch(Circle((0, 0), r_max, fc=ps.GOOD, alpha=0.07, lw=0))
    a2.add_patch(Circle((0, 0), r_max, fill=False, ec=ps.GOOD, ls="--", lw=1.5))
    a2.annotate(f"Limit {r_max:.0f} m", xy=(0, r_max), xytext=(0, 4), textcoords="offset points",
                ha="center", color=ps.GOOD, fontsize=10.5)
    i_apo = np.searchsorted(t, t_apo)
    a2.plot(x[:i_apo + 1], y[:i_apo + 1], color=ps.SERIES[1], label="Wznoszenie")
    a2.plot(x[i_apo:], y[i_apo:], color=ps.SERIES[0], label="Opadanie")
    a2.plot(0, 0, marker="^", ms=11, color=ps.INK, ls="none")
    a2.annotate("Wyrzutnia", (0, 0), xytext=(-10, 6), textcoords="offset points", fontsize=10.5, ha="right")
    ps.mark_nominal(a2, m["apogee_x_m"], m["apogee_y_m"], "Apogeum", dx=10, dy=-4)
    ps.mark_nominal(a2, m["impact_x_m"], m["impact_y_m"], f"Lądowanie\n{m['drift_m']:.0f} m od wyrzutni",
                    color=ps.SERIES[0])
    lim = max(r_max, abs(x).max(), abs(y).max()) * 1.15
    a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim); a2.set_aspect("equal")
    w = p["environment"]["wind"]
    a2.set_title(f"Ślad na ziemi (wiatr {w['ground_speed_mean_m_s']:.0f} m/s → {w['nominal_direction_deg_toward']:.0f}°)")
    a2.set_xlabel("Wschód [m]"); a2.set_ylabel("Północ [m]")
    a2.legend(loc="lower left")
    fig.suptitle(f"{name} – lot nominalny", x=0.01, ha="left", y=1.02)
    ps.save(fig, out_dir / "nominal_trajectory.png")

    # ---------- 2. Wznoszenie: prędkość, przyspieszenie, stateczność (wspólna oś czasu)
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    tt, v = _src(flight.speed)
    sel = tt <= t_apo
    axs[0].plot(tt[sel], v[sel], color=ps.SERIES[0])
    i = int(np.argmax(v[sel]))
    ps.mark_nominal(axs[0], tt[sel][i], v[sel][i], f"{v[sel][i]:.0f} m/s  (Ma {m['max_mach']:.2f})")
    axs[0].set_ylabel("Prędkość [m/s]")
    axs[0].set_title("Prędkość całkowita")

    ta, acc = _src(flight.acceleration)
    sel = ta <= t_apo
    axs[1].plot(ta[sel], acc[sel] / 9.81, color=ps.SERIES[1])
    i = int(np.argmax(acc[sel]))
    ps.mark_nominal(axs[1], ta[sel][i], acc[sel][i] / 9.81, f"maks. {acc[sel][i] / 9.81:.1f} g", dx=12, dy=-18)
    axs[1].set_ylabel("Przyspieszenie [g]")
    axs[1].set_title("Przyspieszenie (moduł; po wyłączeniu silnika = hamowanie oporem)")

    ts, sm = _src(flight.stability_margin)
    sel = (ts >= t_rail) & (ts <= t_apo)
    ax = axs[2]
    ax.plot(ts[sel], sm[sel], color=ps.SERIES[2])
    ax.set_ylim(0, max(rq["stability_max_cal"], sm[sel].max()) + 1.0)
    ps.forbidden_below(ax, rq["stability_min_cal"], f"Min. EuRoC {rq['stability_min_cal']} cal (RQT-1030)")
    ps.forbidden_above(ax, rq["stability_max_cal"], f"Maks. projektowe {rq['stability_max_cal']} cal")
    ps.mark_nominal(ax, t_rail, m["stability_min_cal"],
                    f"Zejście z szyny: {m['rail_exit_velocity_m_s']:.1f} m/s, {m['stability_min_cal']:.2f} cal", dy=-26)
    ax.set_ylabel("Margines [kalibry]")
    ax.set_title("Margines stateczności statycznej (od zejścia z szyny do apogeum)")
    ax.set_xlabel("Czas [s]")
    for a in axs:
        a.axvline(t_burn, color=ps.MUTED, lw=1, ls=":")
    axs[0].annotate("koniec pracy silnika", xy=(t_burn, axs[0].get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=10, color=ps.INK2)
    axs[2].set_xlim(0, t_apo)
    fig.suptitle(f"{name} – faza wznoszenia", x=0.01, ha="left")
    fig.tight_layout()
    ps.save(fig, out_dir / "nominal_ascent.png")

    # ---------- 3. Opadanie: prędkość pionowa vs wysokość
    fig, ax = plt.subplots(figsize=(11, 7))
    tv, vz = _src(flight.vz)
    tz, zz = _src(flight.z)
    zi = np.interp(tv, tz, zz) - elev
    sel = tv >= t_apo + 1
    lo, hi = rq["drogue_speed_range_m_s"]
    ax.axvspan(lo, hi, color=ps.GOOD, alpha=0.08, lw=0)
    ax.axvline(rq["main_speed_max_m_s"], color=ps.CRIT, ls="--", lw=1.5)
    ax.plot(-vz[sel], zi[sel], color=ps.SERIES[0])
    ax.set_ylim(0, m["apogee_agl_m"] * 1.05)
    ax.set_xlim(0, max(55, np.nanmax(-vz[sel]) * 1.1))
    ax.text((lo + hi) / 2, m["apogee_agl_m"] * 0.30, f"Drogue:\n{lo:.0f}–{hi:.0f} m/s\n(RQT-0360)",
            ha="center", va="center", color=ps.GOOD, fontsize=10.5)
    ax.text(rq["main_speed_max_m_s"] + 0.6, m["apogee_agl_m"] * 0.55, f"Main < {rq['main_speed_max_m_s']:.0f} m/s\n(RQT-0380)",
            color=ps.CRIT, fontsize=10.5, va="center")
    ax.axhline(rq["main_deploy_max_agl_m"], color=ps.MUTED, ls=":", lw=1.2)
    ax.text(ax.get_xlim()[1], rq["main_deploy_max_agl_m"], f"maks. wysokość otwarcia main {rq['main_deploy_max_agl_m']:.0f} m ",
            ha="right", va="bottom", fontsize=10, color=ps.INK2)
    dsel = (tv >= t_apo + p["recovery"]["drogue"].get("lag_s", 0) + 5) & (tv <= ev.get("main", t_end))
    if dsel.any():
        vd, zd = -vz[dsel], zi[dsel]
        i_hi, i_lo = int(np.argmax(vd)), int(np.argmin(vd))
        ps.mark_nominal(ax, vd[i_hi], zd[i_hi], f"{vd[i_hi]:.1f} m/s (drogue, wysoko)")
        ps.mark_nominal(ax, vd[i_lo], zd[i_lo], f"{vd[i_lo]:.1f} m/s (drogue, nisko)", dy=14)
    ps.mark_nominal(ax, m["landing_speed_m_s"], 0, f"Przyziemienie {m['landing_speed_m_s']:.1f} m/s", dx=10, dy=-2)
    ax.texts[-1].set_va("bottom")
    ax.set_xlabel("Prędkość opadania [m/s]")
    ax.set_ylabel("Wysokość AGL [m]")
    ax.set_title(f"{name} – prędkość opadania na spadochronach")
    ps.save(fig, out_dir / "nominal_descent.png")

    # ---------- 4. Silnik: ciąg i masa
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(14, 5))
    tt, F = _src(motor.thrust)
    b1.plot(tt, F, color=ps.SERIES[1])
    b1.fill_between(tt, F, color=ps.SERIES[1], alpha=0.12, lw=0)
    b1.set_title(f"Krzywa ciągu – {p['motor']['name']}")
    b1.set_xlabel("Czas [s]"); b1.set_ylabel("Ciąg [N]")
    b1.text(0.5, 0.45, f"I = {motor.total_impulse / 1000:.2f} kN·s\nt = {motor.burn_out_time:.2f} s\n"
            f"Isp = {p['motor']['isp_s']:.0f} s,  O/F = {p['motor']['of_ratio']:.1f}",
            transform=b1.transAxes, ha="center", fontsize=11.5, color=ps.INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=ps.SURFACE, ec=ps.GRID))
    b1.set_ylim(0, F.max() * 1.15)
    tm = np.linspace(0, motor.burn_out_time + 1, 300)
    b2.plot(tm, [flight.rocket.total_mass(x) for x in tm], color=ps.SERIES[0])
    b2.set_title("Masa rakiety")
    b2.set_xlabel("Czas [s]"); b2.set_ylabel("Masa [kg]")
    fig.tight_layout()
    ps.save(fig, out_dir / "nominal_motor.png")


# ============================================================ MONTE CARLO
def generate_report(p: dict, df: pd.DataFrame, out_dir: Path):
    ok = df[df["error"] == ""].copy()
    name = p["rocket"]["name"]
    rq = p["requirements"]
    n = len(ok)
    target = p["mission"]["target_apogee_agl_m"]

    # ---------- 1. Apogeum
    fig, ax = plt.subplots(figsize=(11, 6))
    a = ok["apogee_agl_m"]
    ax.hist(a, bins=25, color=ps.SERIES[0], alpha=0.85, edgecolor=ps.SURFACE, linewidth=1.5)
    ps.target_line(ax, target, f"Cel {target:.0f} m", vertical=True)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    ax.axvspan(p5, p95, color=ps.SERIES[0], alpha=0.07, lw=0)
    ax.set_title(f"Rozkład apogeum – {n} lotów Monte Carlo")
    ax.set_xlabel("Apogeum AGL [m]"); ax.set_ylabel("Liczba lotów")
    ax.text(0.99, 0.97, f"mediana {p50:.0f} m\nśrednia {a.mean():.0f} m, σ = {a.std():.0f} m\n90% lotów: {p5:.0f}–{p95:.0f} m",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=ps.SURFACE, ec=ps.GRID))
    ps.save(fig, out_dir / "mc_apogee.png")

    # ---------- 2. Mapa lądowań
    fig, ax = plt.subplots(figsize=(10, 9))
    r_max = rq["landing_radius_max_m"]
    ax.add_patch(Circle((0, 0), r_max, fc=ps.GOOD, alpha=0.07, lw=0))
    ax.add_patch(Circle((0, 0), r_max, fill=False, ec=ps.GOOD, ls="--", lw=1.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("wind", plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)))
    sc = ax.scatter(ok["impact_x_m"], ok["impact_y_m"], c=ok["in_ground_wind_speed_m_s"], cmap=cmap,
                    vmin=0, vmax=ok["in_ground_wind_speed_m_s"].max(), s=34, edgecolor=ps.INK2, linewidth=0.4)
    ax.plot(0, 0, marker="^", ms=12, color=ps.INK, ls="none")
    ax.annotate("Wyrzutnia", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=10.5)
    inside = (ok["drift_m"] <= r_max).mean() * 100
    ax.annotate(f"Limit {r_max:.0f} m – w środku {inside:.0f}% lotów", xy=(0, r_max), xytext=(0, 5),
                textcoords="offset points", ha="center", color=ps.GOOD, fontsize=11)
    lim = max(r_max, ok["drift_m"].max()) * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Wiatr przy ziemi [m/s]")
    cb.outline.set_visible(False)
    ax.set_title(f"Miejsca lądowania (P95 odległości = {np.percentile(ok['drift_m'], 95):.0f} m)")
    ax.set_xlabel("Wschód [m]"); ax.set_ylabel("Północ [m]")
    ps.save(fig, out_dir / "mc_landing_map.png")

    # ---------- 3. Odzysk: drogue i main
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(14, 5.5))
    lo, hi = rq["drogue_speed_range_m_s"]
    b1.axvspan(lo, hi, color=ps.GOOD, alpha=0.08, lw=0)
    b1.hist(ok["drogue_speed_max_m_s"], bins=20, color=ps.SERIES[0], alpha=0.85, label="po otwarciu (wysoko)",
            edgecolor=ps.SURFACE)
    b1.hist(ok["drogue_speed_min_m_s"], bins=20, color=ps.SERIES[1], alpha=0.85, label="przed mainem (nisko)",
            edgecolor=ps.SURFACE)
    b1.set_xlim(min(lo - 3, ok["drogue_speed_min_m_s"].min() - 2), max(hi + 3, ok["drogue_speed_max_m_s"].max() + 2))
    b1.text((lo + hi) / 2, b1.get_ylim()[1] * 0.97, f"dozwolone {lo:.0f}–{hi:.0f} m/s", ha="center", va="top",
            color=ps.GOOD, fontsize=10.5)
    b1.set_title("Prędkość na drogue (RQT-0360)")
    b1.set_xlabel("Prędkość opadania [m/s]"); b1.set_ylabel("Liczba lotów")
    b1.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    b2.hist(ok["landing_speed_m_s"], bins=20, color=ps.SERIES[2], alpha=0.9, edgecolor=ps.SURFACE)
    b2.axvline(rq["main_speed_max_m_s"], color=ps.CRIT, ls="--", lw=1.5)
    b2.set_xlim(ok["landing_speed_m_s"].min() - 1, max(rq["main_speed_max_m_s"] + 1, ok["landing_speed_m_s"].max() + 0.5))
    b2.text(rq["main_speed_max_m_s"] - 0.1, b2.get_ylim()[1] * 0.97, f"limit {rq['main_speed_max_m_s']:.0f} m/s ",
            ha="right", va="top", color=ps.CRIT, fontsize=10.5)
    b2.set_title("Prędkość przyziemienia na mainie (RQT-0380)")
    b2.set_xlabel("Prędkość [m/s]"); b2.set_ylabel("Liczba lotów")
    fig.tight_layout()
    ps.save(fig, out_dir / "mc_recovery.png")

    # ---------- 4. Wrażliwość apogeum (korelacja Spearmana)
    ins = [c for c in ok.columns if c.startswith("in_") and ok[c].std() > 0]
    corr = pd.Series({NAMES.get(c[3:], c[3:]): ok[c].corr(ok["apogee_agl_m"], method="spearman") for c in ins})
    corr = corr.dropna()
    corr = corr[corr.abs() >= 0.02].sort_values(key=abs)
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(corr) + 1.8))
    colors = [ps.SERIES[1] if v < 0 else ps.SERIES[0] for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, height=0.6)
    for yi, v in enumerate(corr.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=10.5, color=ps.INK2)
    ax.axvline(0, color=ps.INK, lw=1)
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Co najbardziej wpływa na apogeum? (korelacja Spearmana)")
    ax.set_xlabel("← obniża apogeum      ρ      podnosi apogeum →")
    ps.save(fig, out_dir / "mc_sensitivity.png")

    # ---------- 5. Spełnienie wymagań
    flags = {"Prędkość zejścia z szyny": "ok_rail_exit", "Stateczność min.": "ok_stability_min",
             "Stateczność maks.": "ok_stability_max", "Prędkość na drogue": "ok_drogue_speed",
             "Wysokość otwarcia main": "ok_main_deploy", "Prędkość przyziemienia": "ok_main_speed",
             "Promień lądowania": "ok_landing_radius"}
    rate = pd.Series({k: ok[v].astype(bool).mean() * 100 for k, v in flags.items() if v in ok})[::-1]
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = [ps.GOOD if r >= 99.9 else (ps.WARN if r >= 90 else ps.CRIT) for r in rate.values]
    ax.barh(rate.index, rate.values, color=cols, height=0.6)
    for yi, r in enumerate(rate.values):
        mark = "✓" if r >= 99.9 else "!"
        ax.text(min(r, 100) - 1.5, yi, f"{mark} {r:.0f}%", va="center", ha="right", fontsize=11, color="white",
                fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Odsetek lotów MC spełniających wymagania (n = {n})")
    ax.set_xlabel("Loty spełniające wymaganie [%]")
    ps.save(fig, out_dir / "mc_requirements.png")

    summary = ok[[c for c in ok.columns if not c.startswith("in_") and not c.startswith("ok_")
                  and c not in ("error", "run_id")]].describe(percentiles=[0.05, 0.5, 0.95]).T
    summary.to_csv(out_dir / "monte_carlo_summary.csv")
