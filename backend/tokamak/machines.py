"""Machine registry.

Every entry carries the parameters a transport / equilibrium calculation
needs, plus provenance.  ``kappa_x`` / ``delta_x`` describe the separatrix
(what a renderer draws); ``kappa_95`` / ``delta_95`` describe the 95% flux
surface, which is what the q95 and L-H parametrisations were fitted against.

The numbers are the machines' headline design or reference values.  They are
adequate for scoping and for testing whether a parametrisation transfers
between machines; they are not a substitute for a machine's own equilibrium
reconstruction, and ``source`` says where each set comes from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class Machine:
    """A tokamak's geometry, field, and installed capability."""

    key: str
    label: str
    org: str
    country: str

    R0: float                 # major radius [m]
    a: float                  # minor radius [m]
    kappa_x: float            # separatrix elongation
    delta_x: float            # separatrix triangularity
    kappa_95: float
    delta_95: float
    B0: float                 # toroidal field on axis [T]
    Ip: float                 # flat-top plasma current [MA]

    fuel: str = "D"           # "D", "D-T"
    a_mass: float = 2.0       # average ion mass [amu]
    n_tf: int = 16
    n_pf: int = 6
    n_cs: int = 1
    superconducting: bool = False
    diverted: bool = True

    p_nbi: float = 0.0        # installed NBI power [MW]
    p_icrf: float = 0.0
    p_ecrf: float = 0.0
    e_nbi_kev: float = 100.0  # beam energy [keV]

    pulse_s: float = 1.0      # flat-top duration [s]
    first_plasma: Optional[int] = None
    s_div: float = 1.0        # effective divertor wetted area [m^2]
    purpose: str = ""
    source: str = ""

    # filled in by geometry.attach_geometry
    V: float = field(default=0.0)
    S: float = field(default=0.0)
    L_pol: float = field(default=0.0)
    A_cs: float = field(default=0.0)
    kappa_a: float = field(default=0.0)

    @property
    def eps(self) -> float:
        """Inverse aspect ratio a / R0."""
        return self.a / self.R0

    @property
    def aspect(self) -> float:
        return self.R0 / self.a

    @property
    def p_aux(self) -> float:
        """Total installed auxiliary heating power [MW]."""
        return self.p_nbi + self.p_icrf + self.p_ecrf

    @property
    def n_greenwald(self) -> float:
        """Greenwald density limit [1e20 m^-3]."""
        import math
        return self.Ip / (math.pi * self.a * self.a)

    @property
    def is_spherical(self) -> bool:
        return bool(self.eps > 0.55)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(eps=self.eps, aspect=self.aspect, p_aux=self.p_aux,
                 n_greenwald=self.n_greenwald,
                 is_spherical=bool(self.is_spherical))
        return d


# ---------------------------------------------------------------------------
# Registry.  Conventional aspect ratio first, then low-A and spherical.
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Machine] = {}


def _add(m: Machine) -> Machine:
    _REGISTRY[m.key] = m
    return m


_add(Machine(
    key="iter", label="ITER", org="ITER Organization", country="Франция",
    R0=6.20, a=2.00, kappa_x=1.85, delta_x=0.45, kappa_95=1.70, delta_95=0.33,
    B0=5.30, Ip=15.0, fuel="D-T", a_mass=2.5,
    n_tf=18, n_pf=6, n_cs=6, superconducting=True,
    p_nbi=33.0, p_icrf=20.0, p_ecrf=20.0, e_nbi_kev=1000.0,
    pulse_s=400.0, first_plasma=2035, s_div=3.32,
    purpose="Q=10 жанатын плазма",
    source="ITER Physics Basis, Nucl. Fusion 39 (1999); ITER Research Plan",
))

_add(Machine(
    key="demo", label="EU-DEMO", org="EUROfusion", country="Еуропа",
    R0=9.07, a=2.93, kappa_x=1.65, delta_x=0.33, kappa_95=1.59, delta_95=0.27,
    B0=4.92, Ip=17.75, fuel="D-T", a_mass=2.5,
    n_tf=16, n_pf=6, n_cs=5, superconducting=True,
    p_nbi=30.0, p_icrf=20.0, p_ecrf=50.0, e_nbi_kev=800.0,
    pulse_s=7200.0, first_plasma=None, s_div=6.0,
    purpose="электр энергиясын өндіру демонстрациясы",
    source="EU-DEMO 2017 baseline (Federici et al.)",
))

_add(Machine(
    key="sparc", label="SPARC", org="Commonwealth Fusion Systems", country="АҚШ",
    R0=1.85, a=0.57, kappa_x=1.97, delta_x=0.54, kappa_95=1.75, delta_95=0.40,
    B0=12.2, Ip=8.7, fuel="D-T", a_mass=2.5,
    n_tf=18, n_pf=6, n_cs=3, superconducting=True,
    p_icrf=25.0, pulse_s=10.0, first_plasma=2027, s_div=0.9,
    purpose="ықшам жоғары өрісті Q>1",
    source="Creely et al., J. Plasma Phys. 86 (2020) 865860502",
))

_add(Machine(
    key="jet", label="JET", org="UKAEA / EUROfusion", country="Ұлыбритания",
    R0=2.96, a=1.25, kappa_x=1.70, delta_x=0.30, kappa_95=1.60, delta_95=0.22,
    B0=3.45, Ip=4.8, fuel="D-T", a_mass=2.5,
    n_tf=32, n_pf=8, n_cs=1, superconducting=False,
    p_nbi=34.0, p_icrf=10.0, e_nbi_kev=125.0,
    pulse_s=20.0, first_plasma=1983, s_div=1.8,
    purpose="D-T рекордтары, ITER алдындағы физика",
    source="JET design parameters, ITER Physics Basis",
))

_add(Machine(
    key="diiid", label="DIII-D", org="General Atomics", country="АҚШ",
    R0=1.67, a=0.67, kappa_x=2.00, delta_x=0.70, kappa_95=1.80, delta_95=0.45,
    B0=2.20, Ip=2.0, fuel="D", a_mass=2.0,
    n_tf=24, n_pf=18, n_cs=1, superconducting=False,
    p_nbi=20.0, p_ecrf=6.0, e_nbi_kev=80.0,
    pulse_s=6.0, first_plasma=1986, s_div=0.6,
    purpose="пішінді басқару, ELM және тасымал физикасы",
    source="DIII-D facility parameters",
))

_add(Machine(
    key="east", label="EAST", org="ASIPP, Хэфэй", country="Қытай",
    R0=1.85, a=0.45, kappa_x=1.90, delta_x=0.50, kappa_95=1.75, delta_95=0.35,
    B0=3.50, Ip=1.0, fuel="D", a_mass=2.0,
    n_tf=16, n_pf=14, n_cs=1, superconducting=True,
    p_nbi=8.0, p_icrf=12.0, p_ecrf=4.0, e_nbi_kev=80.0,
    pulse_s=1000.0, first_plasma=2006, s_div=0.5,
    purpose="ұзақ импульс, толық асқын өткізгіш",
    source="EAST facility parameters",
))

_add(Machine(
    key="kstar", label="KSTAR", org="KFE, Тэджон", country="Оңтүстік Корея",
    R0=1.80, a=0.50, kappa_x=2.00, delta_x=0.80, kappa_95=1.80, delta_95=0.50,
    B0=3.50, Ip=2.0, fuel="D", a_mass=2.0,
    n_tf=16, n_pf=14, n_cs=1, superconducting=True,
    p_nbi=12.0, p_ecrf=6.0, e_nbi_kev=100.0,
    pulse_s=300.0, first_plasma=2008, s_div=0.5,
    purpose="жоғары температуралы ұзақ разряд",
    source="KSTAR facility parameters",
))

_add(Machine(
    key="t15md", label="Т-15МД", org="Курчатов институты", country="Ресей",
    R0=1.48, a=0.67, kappa_x=1.80, delta_x=0.40, kappa_95=1.70, delta_95=0.30,
    B0=2.00, Ip=2.0, fuel="D", a_mass=2.0,
    n_tf=16, n_pf=8, n_cs=1, superconducting=False,
    p_nbi=6.0, p_icrf=6.0, p_ecrf=7.0, e_nbi_kev=75.0,
    pulse_s=30.0, first_plasma=2021, s_div=0.9,
    purpose="гибридті реактор физикасы, ұзақ импульс",
    source="Kurchatov Institute T-15MD design parameters",
))

_add(Machine(
    key="ktm", label="KTM", org="ҚР Ұлттық ядролық орталығы", country="Қазақстан",
    R0=0.86, a=0.43, kappa_x=1.70, delta_x=0.40, kappa_95=1.60, delta_95=0.30,
    B0=1.00, Ip=0.75, fuel="D", a_mass=2.0,
    n_tf=12, n_pf=6, n_cs=1, superconducting=False,
    p_nbi=2.0, p_icrf=3.0, p_ecrf=2.0, e_nbi_kev=40.0,
    pulse_s=5.0, first_plasma=2017, s_div=0.3,
    purpose="материалтану, дивертор мен бірінші қабырғаны сынау",
    source="NNC RK KTM design parameters (жыл шамамен)",
))

_add(Machine(
    key="nstx", label="NSTX", org="PPPL, Принстон", country="АҚШ",
    R0=0.85, a=0.67, kappa_x=2.20, delta_x=0.60, kappa_95=2.00, delta_95=0.45,
    B0=0.45, Ip=1.0, fuel="D", a_mass=2.0,
    n_tf=12, n_pf=8, n_cs=1, superconducting=False,
    p_nbi=7.0, p_icrf=6.0, e_nbi_kev=90.0,
    pulse_s=1.5, first_plasma=1999, s_div=0.6,
    purpose="сфералық тор физикасы, жоғары бета",
    source="NSTX facility parameters (PPPL)",
))

_add(Machine(
    key="nstxu", label="NSTX-U", org="PPPL, Принстон", country="АҚШ",
    R0=0.93, a=0.60, kappa_x=2.50, delta_x=0.60, kappa_95=2.20, delta_95=0.45,
    B0=1.00, Ip=2.0, fuel="D", a_mass=2.0,
    n_tf=12, n_pf=8, n_cs=1, superconducting=False,
    p_nbi=12.0, e_nbi_kev=90.0,
    pulse_s=5.0, first_plasma=2016, s_div=0.7,
    purpose="сфералық тордың жоғары өрістегі жалғасы",
    source="NSTX-U upgrade design parameters",
))

_add(Machine(
    key="mastu", label="MAST-U", org="UKAEA, Калхэм", country="Ұлыбритания",
    R0=0.85, a=0.65, kappa_x=2.50, delta_x=0.50, kappa_95=2.20, delta_95=0.40,
    B0=0.78, Ip=2.0, fuel="D", a_mass=2.0,
    n_tf=12, n_pf=12, n_cs=1, superconducting=False,
    p_nbi=5.0, e_nbi_kev=75.0,
    pulse_s=5.0, first_plasma=2020, s_div=0.6,
    purpose="Super-X дивертор, сфералық тор",
    source="MAST-U design parameters (UKAEA)",
))


# ---------------------------------------------------------------------------
def list_machines() -> Dict[str, Machine]:
    """All registered machines, keyed by short name."""
    return dict(_REGISTRY)


def get_machine(key: str) -> Machine:
    """Look a machine up by key, case-insensitively."""
    k = (key or "").strip().lower()
    if k not in _REGISTRY:
        raise KeyError(
            f"machine {key!r} is not registered; available: "
            + ", ".join(sorted(_REGISTRY))
        )
    from copy import deepcopy
    return deepcopy(_REGISTRY[k])


def custom_machine(**kw) -> Machine:
    """Build a machine from explicit parameters.

    Anything not given is taken from ITER, so a caller can vary one knob
    without restating a whole device.  ``key`` defaults to ``"custom"``.
    """
    base = get_machine(kw.pop("base", "iter"))
    base.key = kw.pop("key", "custom")
    base.label = kw.pop("label", "Custom")
    base.source = kw.pop("source", "пайдаланушы берген параметрлер")
    for name, value in kw.items():
        if not hasattr(base, name):
            raise KeyError(f"unknown machine parameter {name!r}")
        setattr(base, name, value)
    return base
