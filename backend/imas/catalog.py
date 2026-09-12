"""Searchable IMAS field catalog.

The full Data Dictionary XML is not shipped. Entries below cover the IDS
families this simulator can speak to; `search` is token-based (no embeddings).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

_TOKEN = re.compile(r"[a-z0-9]+")
_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"

# ids, path, units, documentation
_SEED: List[tuple] = [
    ("summary", "summary.fusion.power", "W", "Total fusion power."),
    ("summary", "summary.fusion.neutron_power_total", "W", "Neutron power from fusion."),
    ("summary", "summary.global_quantities.ip", "A", "Plasma current."),
    ("summary", "summary.global_quantities.b0", "T", "Vacuum toroidal field at R0."),
    ("summary", "summary.global_quantities.q_95", "1", "Safety factor at 95% flux."),
    ("summary", "summary.global_quantities.beta_normal", "1", "Normalised beta."),
    ("summary", "summary.global_quantities.beta_tor", "1", "Toroidal beta."),
    ("summary", "summary.global_quantities.tau_energy", "s", "Energy confinement time."),
    ("summary", "summary.global_quantities.h_98", "1", "H98(y,2) confinement factor."),
    ("summary", "summary.global_quantities.zeff", "1", "Effective charge."),
    ("summary", "summary.global_quantities.t_e_volume_average", "eV", "Volume-average electron temperature."),
    ("summary", "summary.global_quantities.t_i_volume_average", "eV", "Volume-average ion temperature."),
    ("summary", "summary.global_quantities.n_e_volume_average", "m^-3", "Volume-average electron density."),
    ("summary", "summary.global_quantities.n_e_greenwald", "m^-3", "Greenwald density limit."),
    ("summary", "summary.global_quantities.greenwald_fraction", "1", "Line-average density over Greenwald."),
    ("summary", "summary.global_quantities.energy_thermal", "J", "Thermal stored energy."),
    ("summary", "summary.global_quantities.v_loop", "V", "Loop voltage."),
    ("summary", "summary.global_quantities.li_3", "1", "Internal inductance."),
    ("summary", "summary.global_quantities.q_plus", "1", "Fusion gain Q = P_fus / P_aux."),
    ("summary", "summary.global_quantities.h_mode", "1", "H-mode flag."),
    ("summary", "summary.global_quantities.power_radiated", "W", "Total radiated power."),
    ("core_profiles", "core_profiles.profiles_1d.electrons.temperature", "eV", "Electron temperature profile."),
    ("core_profiles", "core_profiles.profiles_1d.electrons.density", "m^-3", "Electron density profile."),
    ("core_profiles", "core_profiles.profiles_1d.ion.temperature", "eV", "Ion temperature profile."),
    ("core_profiles", "core_profiles.profiles_1d.q", "1", "Safety factor profile."),
    ("core_profiles", "core_profiles.profiles_1d.pressure_thermal", "Pa", "Thermal pressure profile."),
    ("core_profiles", "core_profiles.profiles_1d.zeff", "1", "Zeff profile."),
    ("core_profiles", "core_profiles.vacuum_toroidal_field.b0", "T", "Reference toroidal field."),
    ("equilibrium", "equilibrium.time_slice.global_quantities.ip", "A", "Equilibrium plasma current."),
    ("equilibrium", "equilibrium.time_slice.global_quantities.q_95", "1", "q95 from Grad-Shafranov."),
    ("equilibrium", "equilibrium.time_slice.global_quantities.q_axis", "1", "On-axis safety factor."),
    ("equilibrium", "equilibrium.time_slice.global_quantities.beta_normal", "1", "Normalised beta from equilibrium."),
    ("equilibrium", "equilibrium.time_slice.global_quantities.magnetic_axis.r", "m", "Magnetic axis major radius."),
    ("equilibrium", "equilibrium.time_slice.profiles_1d.q", "1", "q(psi) profile."),
    ("equilibrium", "equilibrium.time_slice.boundary.geometric_axis.r", "m", "Geometric axis R."),
    ("nbi", "nbi.unit.power_launched", "W", "Launched neutral-beam power."),
    ("nbi", "nbi.unit.energy", "eV", "Beam particle energy."),
    ("nbi", "nbi.unit.species.a", "u", "Beam ion mass number."),
    ("nbi", "nbi.power_launched", "W", "Total NBI launched power."),
    ("ic_antennas", "ic_antennas.antenna.power_launched", "W", "ICRF launched power per antenna."),
    ("ic_antennas", "ic_antennas.power_launched", "W", "Total ICRH launched power."),
    ("ic_antennas", "ic_antennas.antenna.frequency", "Hz", "ICRF frequency."),
    ("ec_launchers", "ec_launchers.beam.power_launched", "W", "ECRH launched power per beam."),
    ("ec_launchers", "ec_launchers.power_launched", "W", "Total ECRH launched power."),
    ("ec_launchers", "ec_launchers.beam.frequency", "Hz", "ECRH frequency."),
    ("pulse_schedule", "pulse_schedule.time", "s", "Pulse time base."),
    ("pulse_schedule", "pulse_schedule.density_control.n_e_line", "m^-2", "Requested line-average density."),
    ("pulse_schedule", "pulse_schedule.density_control.gas_puff", "Pa.m^3.s^-1", "Gas puff / valve command."),
    ("pulse_schedule", "pulse_schedule.flux_control.i_plasma", "A", "Requested plasma current."),
    ("pulse_schedule", "pulse_schedule.tf.b_field_tor_vacuum_r", "T.m", "Requested vacuum toroidal field*R."),
    ("gas_injection", "gas_injection.valve.flow_rate", "Pa.m^3.s^-1", "Gas valve flow rate."),
    ("pellets", "pellets.time_slice.pellet.velocity", "m.s^-1", "Pellet injection velocity."),
    ("pellets", "pellets.time_slice.pellet.species.fraction", "1", "Pellet composition fraction."),
    ("divertors", "divertors.tungsten.target_heat_flux", "W.m^-2", "Divertor target heat flux."),
    ("divertors", "divertors.tungsten.t_e_target", "eV", "Divertor target electron temperature."),
    ("edge_profiles", "edge_profiles.profiles_1d.t_e", "eV", "SOL / edge electron temperature."),
    ("radiation", "radiation.global_quantities.power_total", "W", "Total radiated power."),
    ("radiation", "radiation.global_quantities.power_prad_h", "W", "Hydrogen line radiation."),
    ("magnetics", "magnetics.b_field_pol_probe.field", "T", "Mirnov / poloidal probe field."),
    ("magnetics", "magnetics.flux_loop.flux", "Wb", "Poloidal flux loop."),
    ("tf", "tf.b_field_tor_vacuum_r", "T.m", "Vacuum toroidal field times R."),
    ("pf_active", "pf_active.coil.current", "A", "Poloidal-field coil current."),
    ("barometry", "barometry.gauge.pressure", "Pa", "Vessel pressure."),
    ("neutron_diagnostic", "neutron_diagnostic.field_of_view.detector.counts", "1", "Neutron detector counts."),
    ("wall", "wall.global_quantities.temperature", "K", "First-wall temperature."),
    ("disruption", "disruption.global_quantities.ip", "A", "Plasma current during disruption."),
    ("disruption", "disruption.global_quantities.runaway_current", "A", "Runaway electron current."),
    ("mhd", "mhd.time_slice.n_tor", "1", "Toroidal mode number."),
    ("core_sources", "core_sources.source.nbi.power", "W", "NBI power deposited in the core."),
    ("core_sources", "core_sources.source.ec.power", "W", "ECRH power deposited in the core."),
    ("core_sources", "core_sources.source.ic.power", "W", "ICRH power deposited in the core."),
    ("core_sources", "core_sources.source.ohmic.power", "W", "Ohmic heating power."),
    ("core_sources", "core_sources.source.fusion.power", "W", "Alpha heating power to the plasma."),
]


def _tokens(text: str) -> set:
    return set(_TOKEN.findall((text or "").lower()))


def _entry(ids: str, path: str, units: str, documentation: str) -> dict:
    return {
        "ids": ids,
        "path": path,
        "units": units,
        "documentation": documentation[:400],
        "data_type": "FLT_1D" if "profiles" in path else "FLT_0D",
    }


def seed_entries() -> List[dict]:
    return [_entry(*row) for row in _SEED]


def _write_json(entries: Iterable[dict]) -> None:
    payload = {
        "source": "IMAS Data Dictionary (ITER Organization, CC-BY-SA 4.0)",
        "entries": list(entries),
    }
    _CATALOG_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def load() -> List[dict]:
    if _CATALOG_PATH.is_file():
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        entries = data.get("entries", data if isinstance(data, list) else [])
        if entries:
            return entries
    entries = seed_entries()
    _write_json(entries)
    return entries


def search(q: str, limit: int = 12) -> List[dict]:
    query = (q or "").strip()
    if not query:
        return []
    qtok = _tokens(query)
    ql = query.lower()
    scored = []
    for e in load():
        blob = f"{e['ids']} {e['path']} {e['documentation']}".lower()
        etok = _tokens(blob)
        overlap = len(qtok & etok)
        path_hit = 3 if ql in e["path"].lower() else 0
        ids_hit = 2 if ql == e["ids"].lower() else 0
        score = overlap + path_hit + ids_hit
        if ql in blob:
            score += 1
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda x: (-x[0], x[1]["path"]))
    return [e for _, e in scored[: max(1, min(limit, 30))]]


def get(path: str) -> Optional[dict]:
    if not path:
        return None
    key = path.strip().lower()
    for e in load():
        if e["path"].lower() == key:
            return e
    return None
