import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def load_params(path: str, profile: str = None) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    if profile:
        prof = p["profiles"][profile]
        p["motor"]["propellant_loaded_fraction"] = prof["propellant_loaded_fraction"]
        p["mission"]["target_apogee_agl_m"] = prof["target_apogee_agl_m"]
    return p

def draw_value(rng: np.random.Generator, d: Dict[str, Any]) -> float:
    kind = d["dist"]
    if kind == "uniform":
        return float(rng.uniform(d["min"], d["max"]))
    if kind == "normal":
        return float(rng.normal(d["mean"], d["std"]))
    if kind == "truncnorm":
        for _ in range(1000):
            x = rng.normal(d["mean"], d["std"])
            if d["min"] <= x <= d["max"]:
                return float(x)
        return float(np.clip(d["mean"], d["min"], d["max"]))
    raise ValueError(f"Nieznany rozkład: {kind}")

def make_samples(p: Dict[str, Any], n: int, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    dist = p["monte_carlo"]["distributions"]
    return [{k: draw_value(rng, d) for k, d in dist.items()} for _ in range(n)]

def nominal_sample(p: Dict[str, Any]) -> Dict[str, float]:
    w = p["environment"]["wind"]
    return {
        "ground_wind_speed_m_s": w["ground_speed_mean_m_s"],
        "wind_direction_deg_toward": w["nominal_direction_deg_toward"],
        "upper_wind_factor": 1.0,
        "temperature_k": p["environment"]["temperature_k"],
        "inclination_deg": p["launch"]["inclination_deg"],
        "heading_deg": p["launch"]["heading_deg"],
        "thrust_scale": 1.0, "propellant_scale": 1.0, "mass_offset_kg": 0.0,
        "com_offset_m": 0.0, "cd_scale": 1.0, "drogue_cds_scale": 1.0, "main_cds_scale": 1.0,
    }