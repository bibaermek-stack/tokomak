"""Map simulator snapshots onto IMAS-style paths.

Live (HUD 0-D) and compute (1.5-D solver) stay in separate envelopes so
the operator cannot mix their numbers in one payload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# HUD / solver field -> IMAS path, and how to present SI alongside HUD units.
_MAP = {
    "t": ("pulse_schedule.time", "s", 1.0, "s"),
    "Ip": ("summary.global_quantities.ip", "A", 1e6, "MA"),
    "Bt": ("summary.global_quantities.b0", "T", 1.0, "T"),
    "Te": ("summary.global_quantities.t_e_volume_average", "eV", 1e3, "keV"),
    "Ti": ("summary.global_quantities.t_i_volume_average", "eV", 1e3, "keV"),
    "ne": ("summary.global_quantities.n_e_volume_average", "m^-3", 1e20, "1e20 m^-3"),
    "nG": ("summary.global_quantities.n_e_greenwald", "m^-3", 1e20, "1e20 m^-3"),
    "fG": ("summary.global_quantities.greenwald_fraction", "1", 1.0, "1"),
    "Pfus": ("summary.fusion.power", "W", 1e6, "MW"),
    "Q": ("summary.global_quantities.q_plus", "1", 1.0, "1"),
    "tauE": ("summary.global_quantities.tau_energy", "s", 1.0, "s"),
    "H98": ("summary.global_quantities.h_98", "1", 1.0, "1"),
    "betaN": ("summary.global_quantities.beta_normal", "1", 1.0, "1"),
    "q95": ("summary.global_quantities.q_95", "1", 1.0, "1"),
    "Zeff": ("summary.global_quantities.zeff", "1", 1.0, "1"),
    "Pnbi": ("nbi.power_launched", "W", 1e6, "MW"),
    "Picr": ("ic_antennas.power_launched", "W", 1e6, "MW"),
    "Pecr": ("ec_launchers.power_launched", "W", 1e6, "MW"),
    "PnbiSet": ("nbi.power_launched", "W", 1e6, "MW"),
    "PicrSet": ("ic_antennas.power_launched", "W", 1e6, "MW"),
    "PecrSet": ("ec_launchers.power_launched", "W", 1e6, "MW"),
    "gasSet": ("pulse_schedule.density_control.gas_puff", "1", 1.0, "valve"),
    "IpSet": ("pulse_schedule.flux_control.i_plasma", "A", 1e6, "MA"),
    "qDiv": ("divertors.tungsten.target_heat_flux", "W.m^-2", 1e6, "MW/m^2"),
    "neutronRate": ("summary.fusion.neutron_power_total", "1/s", 1.0, "1/s"),
    "hMode": ("summary.global_quantities.h_mode", "1", 1.0, "bool"),
    "W": ("summary.global_quantities.energy_thermal", "J", 1e6, "MJ"),
    "Vloop": ("summary.global_quantities.v_loop", "V", 1.0, "V"),
    "Prad": ("summary.global_quantities.power_radiated", "W", 1e6, "MW"),
}


def map_snapshot(snapshot: Optional[dict], source: str) -> Dict[str, Any]:
    """Return IMAS-labelled scalars. `source` is 'live' or 'compute'."""
    snap = snapshot or {}
    out = {"source": source, "fields": []}
    for key, (path, si_unit, scale, hud_unit) in _MAP.items():
        if key not in snap:
            continue
        raw = snap[key]
        try:
            hud_val = float(raw) if not isinstance(raw, bool) else float(raw)
        except (TypeError, ValueError):
            continue
        out["fields"].append({
            "key": key,
            "path": path,
            "hud": hud_val,
            "hud_unit": hud_unit,
            "si": hud_val * scale,
            "si_unit": si_unit,
        })
    if "alarms" in snap:
        out["alarms"] = list(snap.get("alarms") or [])[-8:]
    if "mode" in snap:
        out["regime"] = snap.get("mode")
    if "disrupted" in snap:
        out["disrupted"] = bool(snap.get("disrupted"))
    return out


def pfus_path() -> str:
    return _MAP["Pfus"][0]


def telemetry_table(mapped: Dict[str, Any]) -> list:
    """Rows for an operator table: IMAS path, HUD value, unit."""
    rows = []
    for f in mapped.get("fields") or []:
        val = f.get("hud")
        if isinstance(val, float):
            shown = round(val, 4) if abs(val) < 100 else round(val, 2)
        else:
            shown = val
        rows.append({
            "path": f.get("path", ""),
            "value": shown,
            "unit": f.get("hud_unit", ""),
        })
    return rows
