"""Budowa obiektów RocketPy (środowisko, silnik ciekły, rakieta) z pliku rocket_params.json.

Konwencje:
  * rakieta: tail_to_nose, z = 0 na płaszczyźnie wylotu dyszy,
  * silnik:  nozzle_to_combustion_chamber, z = 0 na wylocie dyszy,
  * wszystkie wysokości w atmosferze są n.p.m. (ASL), jak w RocketPy.
"""
import math
from typing import Tuple

import numpy as np
from rocketpy import (Environment, Fluid, LiquidMotor,
                      MassFlowRateBasedTank, CylindricalTank, Rocket)

G0 = 9.80665          # [m/s^2] standardowe przyspieszenie (definicja Isp)
R_AIR = 287.05        # [J/(kg K)]
T_TROPOPAUSE = 216.65  # [K] temperatura stratosfery wg ISA


def atmosphere_profile(p: dict, s: dict, h_max: float = 30000.0, dh: float = 50.0):
    """Profil T(h) i p(h) zgodny z ISA, ale przesunięty do temperatury i ciśnienia przy ziemi.

    Troposfera: T = T0 - L*(h - h0), p = p0*(T/T0)^(g/(R L)).
    Powyżej tropopauzy: T = 216.65 K, spadek ciśnienia wykładniczy.
    Stała temperatura na wszystkich wysokościach (poprzednia wersja) zaniżała gęstość
    na 9 km o ~22% i zawyżała apogeum o ~900 m.
    """
    env_p = p["environment"]
    h0 = p["site"]["elevation_m"]
    T0 = s["temperature_k"]
    L = env_p.get("lapse_rate_k_per_m", 0.0065)
    p0 = env_p.get("pressure_ground_pa", 101325.0 * (1 - 0.0065 * h0 / 288.15) ** 5.2559)
    n = G0 / (R_AIR * L)
    h_trop = h0 + (T0 - T_TROPOPAUSE) / L
    p_trop = p0 * (T_TROPOPAUSE / T0) ** n

    hs = np.arange(0.0, h_max + dh, dh)
    T = np.where(hs < h_trop, T0 - L * (hs - h0), T_TROPOPAUSE)
    P = np.where(hs < h_trop, p0 * (np.maximum(T, 1.0) / T0) ** n,
                 p_trop * np.exp(-G0 * (hs - h_trop) / (R_AIR * T_TROPOPAUSE)))
    return hs, T, P


def build_environment(p: dict, s: dict) -> Environment:
    site, env_p = p["site"], p["environment"]
    env = Environment(
        gravity=site["gravity_m_s2"], date=tuple(env_p["date"]),
        latitude=site["latitude"], longitude=site["longitude"],
        elevation=site["elevation_m"]
    )

    # Wiatr: profil mnożników względem prędkości przy ziemi; kierunek = dokąd wieje.
    w = env_p["wind"]
    a = math.radians(s["wind_direction_deg_toward"])
    heights = [0.0] + list(w["profile_heights_m"])
    mult = [1.0 + (m - 1.0) * s["upper_wind_factor"]
            for m in ([w["profile_multipliers"][0]] + list(w["profile_multipliers"]))]
    u = [(h, s["ground_wind_speed_m_s"] * m * math.sin(a)) for h, m in zip(heights, mult)]
    v = [(h, s["ground_wind_speed_m_s"] * m * math.cos(a)) for h, m in zip(heights, mult)]

    hs, T, P = atmosphere_profile(p, s)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=list(zip(hs, P)), temperature=list(zip(hs, T)),
        wind_u=u, wind_v=v,
    )
    return env


def motor_numbers(p: dict, s: dict) -> dict:
    """Bilans silnika: masy, czas pracy, impuls. Jedno źródło prawdy dla modelu i raportów."""
    m = p["motor"]
    ru, rd = m["thrust_curve"]["ramp_up_s"], m["thrust_curve"]["ramp_down_s"]
    f_nom = m["thrust_nominal_n"]
    load = m.get("propellant_loaded_fraction", 1.0) * s["propellant_scale"]
    m_full = p["rocket"]["propellant_mass_kg"]          # masa zbiorników pełnych (z pozostałością)
    m_loaded = m_full * load
    m_burned = m_loaded * (1.0 - m["residual_fraction"])  # to, co faktycznie wypłynie przez dyszę
    # Impuls liczony z masy SPALONEJ (wcześniej z załadowanej -> +3% impulsu)
    impulse_nom = m_burned * m["isp_s"] * G0
    burn = impulse_nom / f_nom + 0.5 * (ru + rd)
    return dict(ru=ru, rd=rd, f_nom=f_nom, f=f_nom * s["thrust_scale"], load=load,
                m_full=m_full, m_loaded=m_loaded, m_burned=m_burned, burn=burn,
                impulse=impulse_nom * s["thrust_scale"])


def build_motor(p: dict, s: dict) -> Tuple[LiquidMotor, tuple]:
    m = p["motor"]
    mn = motor_numbers(p, s)
    ru, rd, f, burn, load = mn["ru"], mn["rd"], mn["f"], mn["burn"], mn["load"]
    # Opcjonalny kształt krzywej: "thrust_end_ratio" < 1 daje spadek ciągu (blowdown).
    k_end = s.get("thrust_end_ratio", m["thrust_curve"].get("thrust_end_ratio", 1.0))
    # utrzymanie tego samego impulsu przy krzywej liniowo opadającej
    f_start = f * 2.0 / (1.0 + k_end)
    thrust = [[0.0, 0.0], [ru, f_start], [burn - rd, f_start * k_end], [burn, 0.0]]

    of = m["of_ratio"]
    ox_full, fuel_full = mn["m_full"] * of / (1.0 + of), mn["m_full"] / (1.0 + of)
    r_tank = m["tank_radius_m"]
    area = math.pi * r_tank ** 2
    ull = m["ullage_fraction"]

    def make_tank(name, fluid_cfg, m_full_i):
        rho_l, rho_g = fluid_cfg["liquid_density_kg_m3"], fluid_cfg["gas_density_kg_m3"]
        v_tank = (m_full_i / rho_l) / (1.0 - ull)   # ullage = ułamek objętości zbiornika
        height = v_tank / area
        m_liq = m_full_i * load
        v_gas = max(v_tank - m_liq / rho_l, 1e-4 * v_tank) * 0.999
        out_rate = m_liq * (1.0 - m["residual_fraction"]) / burn
        tank = MassFlowRateBasedTank(
            name=name,
            geometry=CylindricalTank(radius=r_tank, height=height, spherical_caps=False),
            flux_time=burn,
            liquid=Fluid(name=fluid_cfg.get("name", name), density=rho_l),
            gas=Fluid(name=fluid_cfg.get("gas_name", "N2"), density=rho_g),
            initial_liquid_mass=m_liq, initial_gas_mass=v_gas * rho_g,
            liquid_mass_flow_rate_in=0.0, gas_mass_flow_rate_in=0.0,
            liquid_mass_flow_rate_out=out_rate, gas_mass_flow_rate_out=0.0)
        return tank, height

    ox_tank, h_ox = make_tank("oxidizer tank", m["oxidizer"], ox_full)
    fu_tank, h_fu = make_tank("fuel tank", m["fuel"], fuel_full)

    motor = LiquidMotor(
        thrust_source=thrust, dry_mass=m["dry_mass_kg"], dry_inertia=tuple(m["dry_inertia_kg_m2"]),
        nozzle_radius=m["nozzle_radius_m"], center_of_dry_mass_position=m["center_of_dry_mass_from_nozzle_m"],
        nozzle_position=0, burn_time=burn, coordinate_system_orientation="nozzle_to_combustion_chamber"
    )
    motor.add_tank(tank=fu_tank, position=m["fuel_tank_center_from_nozzle_m"])
    motor.add_tank(tank=ox_tank, position=m["oxidizer_tank_center_from_nozzle_m"])
    return motor, (h_ox, h_fu)


def build_rocket(p: dict, s: dict, motor: LiquidMotor) -> Rocket:
    r = p["rocket"]
    mach = r["drag"]["mach"]
    cd_off = [[mm, c * s["cd_scale"]] for mm, c in zip(mach, r["drag"]["cd_power_off"])]
    cd_on = [[mm, c * s["cd_scale"]] for mm, c in zip(mach, r["drag"]["cd_power_on"])]

    rocket = Rocket(
        radius=r["radius_m"], mass=r["mass_without_motor_kg"] + s["mass_offset_kg"],
        inertia=tuple(r["inertia_without_motor_kg_m2"]),
        power_off_drag=cd_off, power_on_drag=cd_on,
        center_of_mass_without_motor=r["center_of_mass_without_motor_m"] + s["com_offset_m"],
        coordinate_system_orientation="tail_to_nose"
    )
    rocket.add_motor(motor, position=0)
    rocket.add_nose(length=r["nose"]["length_m"], kind=r["nose"]["kind"], position=r["total_length_m"])

    fn = r["fins"]
    rocket.add_trapezoidal_fins(n=fn["n"], root_chord=fn["root_chord_m"], tip_chord=fn["tip_chord_m"],
                                span=fn["span_m"], position=fn["top_of_root_from_tail_m"],
                                sweep_length=fn["sweep_length_m"])

    # Prowadnice: bez nich RocketPy uznaje zejście z szyny dopiero po przebyciu całej długości
    # szyny przez dyszę, co zawyża prędkość zejścia z szyny.
    rb = r.get("rail_buttons")
    if rb:
        rocket.set_rail_buttons(upper_button_position=rb["upper_from_tail_m"],
                                lower_button_position=rb["lower_from_tail_m"], angular_position=45)

    rc = p["recovery"]
    rate = rc.get("sampling_rate_hz", 100)
    rocket.add_parachute("drogue", cd_s=rc["drogue"]["cd_s_m2"] * s["drogue_cds_scale"], trigger="apogee",
                         sampling_rate=rate, lag=rc["drogue"].get("lag_s", 0.0))
    rocket.add_parachute("main", cd_s=rc["main"]["cd_s_m2"] * s["main_cds_scale"],
                         trigger=float(rc["main"]["deploy_altitude_agl_m"]),
                         sampling_rate=rate, lag=rc["main"].get("lag_s", 0.0))
    return rocket
