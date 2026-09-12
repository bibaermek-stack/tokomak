"""Fuel cycle, blanket and plant power balance.

The question a burning-plasma model is ultimately asked is not "does it
reach Q = 10" but "does the plant put power on the grid and does it make
its own tritium".  Both are decided outside the separatrix:

* the blanket has to breed more tritium than the plasma burns, or the
  machine runs out of fuel;
* the recirculating power -- heating, cryoplant, coils, pumping -- comes off
  the gross electric output before anything reaches the grid, and the
  engineering gain Q_eng is what a utility cares about.

Q = P_fus / P_aux, the number a physicist quotes, ignores both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import E_FUSION_J, F_ALPHA, MW


@dataclass
class BlanketResult:
    P_neutron: float          # [MW]
    wall_load: float          # [MW/m^2] neutron loading on the first wall
    TBR: float                # tritium breeding ratio
    tritium_burn_rate: float  # [kg/day]
    tritium_bred: float       # [kg/day]
    burn_fraction: float      # fraction of injected tritium that fuses
    throughput: float         # [kg/day] through the fuelling system
    startup_inventory: float  # [kg]
    doubling_time_days: float
    energy_multiplication: float

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class PlantResult:
    P_fusion: float
    P_thermal: float          # [MW] into the power conversion system
    P_gross_electric: float
    P_recirculating: float
    P_net_electric: float
    Q_plasma: float           # P_fus / P_aux
    Q_engineering: float      # P_net / P_recirculating + 1
    breakdown: dict           # where the recirculating power goes

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "breakdown"}
        d = {k: float(v) for k, v in d.items()}
        d["breakdown"] = {k: float(v) for k, v in self.breakdown.items()}
        return d


# ---------------------------------------------------------------------------
def blanket(P_fusion: float, S_wall: float, *,
            tbr_design: float = 1.15, coverage: float = 0.85,
            energy_multiplication: float = 1.20,
            burn_fraction: float = 0.010,
            tritium_processing_days: float = 1.0,
            reserve_days: float = 3.0) -> BlanketResult:
    """Tritium breeding and neutron loading.

    ``tbr_design`` is the local breeding ratio of the blanket module;
    multiplied by the coverage fraction it gives the achieved TBR, which has
    to exceed 1 with margin for losses and decay.  ``burn_fraction`` is the
    fraction of injected tritium that actually fuses before being pumped out
    -- about 1% in ITER-class machines, which is why the throughput is a
    hundred times the burn rate and the startup inventory is measured in
    kilograms.
    """
    P_n = P_fusion * (1.0 - F_ALPHA)
    wall_load = P_n / max(S_wall, 1e-3)

    # 1 D-T reaction consumes one triton; 1 kg of T is 1.9928e26 atoms
    reactions_per_s = P_fusion * MW / E_FUSION_J
    kg_per_atom = 3.016 * 1.66054e-27
    burn_kg_day = reactions_per_s * kg_per_atom * 86400.0

    tbr = tbr_design * coverage
    bred_kg_day = burn_kg_day * tbr

    throughput = burn_kg_day / max(burn_fraction, 1e-4)
    # inventory held up in the fuel loop plus a reserve
    startup = throughput * (tritium_processing_days + reserve_days) * 0.35

    surplus = bred_kg_day - burn_kg_day
    doubling = (startup / surplus) if surplus > 1e-9 else float("inf")

    return BlanketResult(
        P_neutron=P_n, wall_load=wall_load, TBR=tbr,
        tritium_burn_rate=burn_kg_day, tritium_bred=bred_kg_day,
        burn_fraction=burn_fraction, throughput=throughput,
        startup_inventory=startup,
        doubling_time_days=min(doubling, 1e6),
        energy_multiplication=energy_multiplication,
    )


def plant_balance(P_fusion: float, P_aux: float, *,
                  P_ohmic: float = 0.0,
                  energy_multiplication: float = 1.20,
                  thermal_efficiency: float = 0.35,
                  heating_efficiency: float = 0.40,
                  P_cryo: float = 0.0, P_coils: float = 0.0,
                  P_pumping: float = 0.0, P_balance_of_plant: float = 0.0,
                  superconducting: bool = True,
                  R0: float = 6.2) -> PlantResult:
    """Gross and net electric power, and the engineering gain.

    Defaults are reactor-relevant: a 1.2 blanket energy multiplication, 35%
    thermal-to-electric conversion, and 40% wall-plug efficiency for the
    heating systems.  When the auxiliary loads are not given they are
    estimated from the machine size, which is enough to show the shape of
    the balance -- a machine can reach Q = 10 and still deliver no net
    electricity.
    """
    if P_cryo <= 0:
        P_cryo = (0.35 * R0 ** 1.4) if superconducting else 0.0
    if P_coils <= 0:
        P_coils = 0.0 if superconducting else 60.0 * (R0 / 6.2) ** 2
    if P_pumping <= 0:
        P_pumping = 0.02 * max(P_fusion, 1.0) + 5.0
    if P_balance_of_plant <= 0:
        P_balance_of_plant = 0.01 * max(P_fusion, 1.0) + 8.0

    P_thermal = P_fusion * energy_multiplication + P_aux + P_ohmic
    P_gross = P_thermal * thermal_efficiency

    P_heat_wall = P_aux / max(heating_efficiency, 1e-3)
    P_recirc = P_heat_wall + P_cryo + P_coils + P_pumping + P_balance_of_plant
    P_net = P_gross - P_recirc

    q_plasma = P_fusion / P_aux if P_aux > 1e-6 else float("inf")
    q_eng = P_gross / P_recirc if P_recirc > 1e-6 else float("inf")

    return PlantResult(
        P_fusion=P_fusion, P_thermal=P_thermal, P_gross_electric=P_gross,
        P_recirculating=P_recirc, P_net_electric=P_net,
        Q_plasma=q_plasma, Q_engineering=q_eng,
        breakdown={
            "қыздыру жүйелері": P_heat_wall,
            "криожүйе": P_cryo,
            "катушкалар": P_coils,
            "вакуум сорғылары": P_pumping,
            "станция жүйелері": P_balance_of_plant,
        },
    )


def neutron_damage(wall_load: float, years: float = 2.0,
                   availability: float = 0.3) -> dict:
    """First-wall displacement damage and helium production.

    ~10 dpa per MW-yr/m^2 in steel is the usual conversion; structural
    lifetime limits sit around 50-100 dpa, which together with the wall
    load is what sets how often the blanket has to be replaced.
    """
    fluence = wall_load * years * availability            # MW yr / m^2
    return {
        "fluence_MWyr_m2": fluence,
        "dpa": 10.0 * fluence,
        "he_appm": 130.0 * fluence,
        "lifetime_years": (50.0 / max(10.0 * wall_load * availability, 1e-6)),
    }
