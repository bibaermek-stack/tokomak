"""Component validation against published values.

Each check exercises one physics module against a number that exists
independently of this code.  The point of keeping them together is that a
model is only as good as what it has been checked against, and the list of
what has *not* been checked is part of the result -- so ``report()`` returns
both.

Run with ``python -m tokamak.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import current as cur
from . import disruption as disr
from . import equilibrium as eqm
from . import fuelcycle as fc
from . import geometry as geo
from . import sol as sol_mod
from .machines import get_machine
from .reactivity import BOSCH_HALE_TABLE, sigma_v
from .scalings import (CONFINEMENT, ConfinementInputs, KZ1, ST_ANCHOR,
                       ST_ANCHOR_TAU, tau_energy)


@dataclass
class Check:
    name: str
    model: float
    reference: float
    unit: str
    tol_pct: float
    source: str
    note: str = ""

    @property
    def deviation(self) -> float:
        """Relative deviation in percent; absolute when the reference is 0."""
        if abs(self.reference) < 1e-12:
            return (self.model - self.reference) * 100.0
        return (self.model - self.reference) / self.reference * 100.0

    @property
    def passed(self) -> bool:
        return abs(self.deviation) <= self.tol_pct

    def to_dict(self) -> dict:
        return {"name": self.name, "model": self.model,
                "reference": self.reference, "unit": self.unit,
                "deviation_pct": self.deviation, "tol_pct": self.tol_pct,
                "passed": self.passed, "source": self.source,
                "note": self.note}


# ---------------------------------------------------------------------------
def check_reactivity() -> List[Check]:
    out = []
    for T, ref in BOSCH_HALE_TABLE.items():
        out.append(Check(f"⟨σv⟩ D-T, T = {T:g} кэВ", float(sigma_v(T)), ref,
                         "м³/с", 0.5, "Bosch & Hale 1992, Table VIII"))
    return out


def check_geometry() -> List[Check]:
    m = geo.attach_geometry(get_machine("iter"))
    n = geo.attach_geometry(get_machine("nstx"))
    return [
        Check("ITER плазма көлемі", m.V, 840.0, "м³", 1.0, "ITER Organization"),
        Check("ITER плазма беті", m.S, 680.0, "м²", 1.0, "ITER Organization"),
        Check("ITER аудандық κ_a", m.kappa_a, 1.70, "", 1.5, "IPB98(y,2)"),
        Check("NSTX плазма көлемі", n.V, 13.8, "м³", 5.0, "PPPL NSTX"),
    ]


def check_equilibrium() -> List[Check]:
    m = geo.attach_geometry(get_machine("iter"))
    eq = eqm.solve_matched(m.R0, m.a, m.kappa_x, m.delta_x, m.Ip, m.B0,
                           p_avg=2.59e5, q0=1.0, nR=97, nZ=145)
    return [
        Check("ГШ: q₀ (мақсат 1.0)", eq.q0, 1.00, "", 2.0,
              "сауытты разряд үшін q₀ ≈ 1"),
        Check("ГШ: ⟨p⟩ (мақсат 259 кПа)", eq.p_avg / 1e3, 259.0, "кПа", 2.0,
              "ITER Q=10 қысымы"),
        Check("ГШ: ішкі индуктивтілік l_i(3)", eq.li3, 0.90, "", 15.0,
              "ITER жану фазасы, 0.8-1.0"),
        Check("ГШ: β_N (жылулық)", eq.beta_n, 1.64, "", 6.0,
              "ITER жылулық β_N"),
        Check("ГШ: q₉₅ (X-нүкте түзетуімен)", eq.q95, 3.00, "", 2.0,
              "ITER жоба"),
        Check("ГШ: q₉₅ түзетусіз", eq.q95_raw, 2.71, "", 3.0,
              "бекітілген шекаралы шикі мән",
              "түзетудің көлемі көрініп тұруы үшін"),
        Check("ГШ: плазма көлемі", eq.V, 849.5, "м³", 1.0,
              "тегіс шекара (X-нүкте кесілмеген)"),
    ]


def check_bootstrap() -> List[Check]:
    """Bootstrap fraction at ITER's Q=10 profiles."""
    m = geo.attach_geometry(get_machine("iter"))
    rho = np.linspace(0.0, 1.0, 65)
    ne = 1.01 * (1.36 * np.exp(-1.3 * rho ** 2 / 2.0) / 1.0)
    ne = ne / np.trapezoid(ne * 2 * rho, rho) * 1.01
    Te = 4.5 + 20.5 * (1.0 - rho ** 2) ** 2.2
    Ti = 4.2 + 16.8 * (1.0 - rho ** 2) ** 2.2
    q = 1.0 + 2.0 * rho ** 2
    bs = cur.bootstrap(rho, ne, Te, Ti, q, R0=m.R0, a=m.a, B0=m.B0,
                       Ip=m.Ip, z_eff=1.65, f_ion=0.87, A_cs=m.A_cs)
    iter_eps = m.eps
    eta_n = cur.neoclassical_resistivity(np.array([0.35]), np.array([1.2]),
                                         np.array([12.0]), 1.65, m.R0,
                                         m.eps, np.array([1.5]))[0]
    eta_s = cur.spitzer_resistivity(12.0, 1.65)
    return [
        Check("Бутстрап үлесі f_BS (ITER)", bs.f_bs, 0.20, "", 40.0,
              "ITER сценарийлерінде 0.15-0.25",
              "Sauter коэффициенттері; профильге сезімтал"),
        Check("f_BS / (0.7 √ε β_p) сәйкестігі",
              bs.f_bs / (0.7 * np.sqrt(iter_eps) * 0.59), 1.00, "", 20.0,
              "аналитикалық бағалаумен ішкі сәйкестік",
              "екі тәуелсіз өрнек бір-бірімен келісуі керек"),
        Check("Неоклассикалық η / Спитцер", eta_n / eta_s, 2.5, "", 40.0,
              "реактор аспектінде 2-3 есе", ""),
        Check("CS флюкс сыйымдылығы (ITER)",
              cur.cs_flux_capacity(m.R0, m.a, m.Ip), 277.0, "Вб", 25.0,
              "ITER соленоидының ~277 Вб қоры"),
    ]


def check_sol() -> List[Check]:
    m = geo.attach_geometry(get_machine("iter"))
    B_pol = geo.b_poloidal(m.Ip, m.L_pol)
    att = sol_mod.two_point(90.0, R0=m.R0, a=m.a, B0=m.B0, B_pol=B_pol,
                            q95=3.0, n_sep20=0.30, f_rad_seed=0.0)
    det = sol_mod.two_point(90.0, R0=m.R0, a=m.a, B0=m.B0, B_pol=B_pol,
                            q95=3.0, n_sep20=0.30, f_rad_seed=0.75)
    return [
        Check("Eich λ_q (ITER)", att.lambda_q_mm, 0.63, "мм", 20.0,
              "Eich #14: 0.63 B_pol^-1.19; ITER-де ~0.6 мм"),
        Check("Байланыс ұзындығы L∥", sol_mod.connection_length(m.R0, 3.0),
              58.0, "м", 10.0, "π R q₉₅"),
        Check("Себусіз ITER нысанасы шыдамайды (q_t)",
              1.0 if att.q_target_MWm2 > 10.0 else 0.0, 1.0, "", 1.0,
              "себусіз жүктеме 10 МВт/м² шегінен әлдеқайда жоғары",
              "1 = модель себуді міндетті деп табады, солай болуы керек"),
        Check("10 МВт/м² үшін қажет себу үлесі",
              sol_mod.seeding_for_target(
                  90.0, q_limit=10.0, R0=m.R0, a=m.a, B0=m.B0, B_pol=B_pol,
                  q95=3.0, n_sep20=0.30),
              0.75, "", 35.0, "ITER: экзаусттың ~70-80%-ы сәулеленуі керек",
              "модель себуді өзі талап етеді"),
        Check("λ_int = λ_q + 1.64 S (ITER)", att.lambda_int_mm, 2.25, "мм",
              15.0, "Eich екі енді фиті; ITER-де S ≈ 1 мм"),
    ]


def check_disruption() -> List[Check]:
    m = geo.attach_geometry(get_machine("iter"))
    un = disr.simulate(cause="test", Ip=15.0, R0=m.R0, a=m.a, kappa=m.kappa_x,
                       B0=m.B0, W_thermal=325.0, li=0.9, ne20=1.0,
                       z_eff=1.65, S_wall=m.S, mitigated=False)
    mi = disr.simulate(cause="test", Ip=15.0, R0=m.R0, a=m.a, kappa=m.kappa_x,
                       B0=m.B0, W_thermal=325.0, li=0.9, ne20=1.0,
                       z_eff=1.65, S_wall=m.S, mitigated=True)
    return [
        Check("Полоидалды магнит энергиясы", un.W_magnetic, 1300.0, "МДж",
              15.0, "ITER плазмасының ~1.3 ГДж қоры"),
        Check("Ток өшуінің уақыты τ_CQ", un.tau_cq_ms, 80.0, "мс", 70.0,
              "ITER: 50-150 мс"),
        Check("Қашқын күшеюі / I_p", un.avalanche_gain_exp / 15.0, 2.5, "",
              100.0, "Rosenbluth-Putvinski: ln(күшею) ~ 2.5 I_p[МА]",
              "реті дұрыс; коэффициент болжамдарға сезімтал"),
        Check("E_ind / E_c (бәсеңдетусіз)", un.E_induced / un.E_critical,
              200.0, "", 400.0, "қашқындар туу үшін >> 1 болуы керек"),
        Check("Бәсеңдету қашқындарды басады",
              0.0 if mi.I_runaway < 1.0 else 1.0, 0.0, "", 1.0,
              "MGI/SPI E_c-ні E_ind-тан жоғары көтеруі керек",
              "0 = сәтті басылды"),
    ]


def check_fuelcycle() -> List[Check]:
    m = geo.attach_geometry(get_machine("iter"))
    b = fc.blanket(500.0, m.S)
    p = fc.plant_balance(500.0, 50.0, superconducting=True, R0=m.R0)
    d = fc.blanket(2000.0, geo.attach_geometry(get_machine("demo")).S)
    return [
        Check("Нейтрон қабырға жүктемесі (ITER)", b.wall_load, 0.57,
              "МВт/м²", 20.0, "ITER: ~0.5-0.6 МВт/м²"),
        Check("Тритий жану жылдамдығы (500 МВт)", b.tritium_burn_rate,
              0.0766, "кг/тәу", 5.0, "P/E_fus × m_T"),
        Check("EU-DEMO қабырға жүктемесі", d.wall_load, 1.05, "МВт/м²",
              30.0, "DEMO: ~1 МВт/м²"),
        Check("EU-DEMO таза электр қуаты", fc.plant_balance(
            2000.0, 100.0, superconducting=True, R0=9.07).P_net_electric,
            500.0, "МВт", 30.0, "EU-DEMO мақсаты ~500 МВт нетто"),
        Check("EU-DEMO Q_eng", fc.plant_balance(
            2000.0, 100.0, superconducting=True, R0=9.07).Q_engineering,
            3.0, "", 40.0, "реактор үшін Q_eng > 3 керек"),
    ]


def check_pedestal() -> List[Check]:
    """Pedestal model: the ITER point it is calibrated to, and the scaling
    it is not."""
    from . import pedestal as ped
    m = geo.attach_geometry(get_machine("iter"))
    base = dict(R0=m.R0, a=m.a, kappa_a=m.kappa_a, B0=m.B0, Ip=m.Ip,
                q95=3.0, n_ped20=0.75, f_ion=0.87, L_pol=m.L_pol)
    p = ped.solve(**base)
    return [
        Check("Педестал: ITER T_ped", p.T_ped, 4.50, "кэВ", 2.0,
              "ITER педестал болжамдары", "калибрлеу нүктесі"),
        Check("Педестал: ені Δψ_N", p.width_psi, 0.040, "", 15.0,
              "EPED1: ITER-де ~0.04"),
        Check("Педестал: β_p,ped", p.beta_p_ped, 0.25, "", 15.0,
              "ITER педесталының полоидалды бетасы"),
        Check("Педестал: p_ped ~ Ip^n", ped.scaling_exponent("Ip", **base),
              2.00, "", 3.0, "EPED-тің негізгі нәтижесі",
              "КАЛИБРЛЕНБЕГЕН — екі шектеудің қиылысынан шығады"),
        Check("Педестал: p_ped ~ B^n", ped.scaling_exponent("B0", **base),
              4.00, "", 3.0, "сол екі шектеуден"),
    ]


def check_kz1() -> List[Check]:
    """The package's own scaling: does it do what it claims?"""
    iter_m = geo.attach_geometry(get_machine("iter"))
    x_iter = ConfinementInputs(Ip=15.0, B0=5.3, P_loss=89.0, n_bar20=1.06,
                               R0=6.2, a=2.0, kappa_a=iter_m.kappa_a,
                               a_mass=2.5)
    ratio_iter = tau_energy("kz1", x_iter) / tau_energy("ipb98y2", x_iter)
    kz_anchor = tau_energy("kz1", ST_ANCHOR)
    return [
        Check("KZ-1 ITER-де IPB98(y,2)-ге тең", ratio_iter, 1.00, "", 2.0,
              "кәдімгі аспектіде дәл сәйкес болуы керек"),
        Check("KZ-1 NSTX анкерін қайта шығарады", kz_anchor * 1e3,
              ST_ANCHOR_TAU * 1e3, "мс", 1.0, "құрылысы бойынша"),
        Check("KZ-1 араластыру салмағы, ITER", KZ1.weight(2.0 / 6.2), 0.01,
              "", 30.0, "ε=0.32-де ST тармағы қосылмауы керек"),
        Check("KZ-1 араластыру салмағы, NSTX", KZ1.weight(0.67 / 0.85), 1.00,
              "", 2.0, "ε=0.79-да толық ST тармағы"),
    ]


def check_integrated() -> List[Check]:
    """The whole chain, run as one discharge, against ITER's Q=10 point.

    This is the demanding test: no quantity below is imposed.  The pedestal
    comes from peeling-ballooning, the core from stiff transport inside it,
    the equilibrium from Grad-Shafranov, the composition from the ash and
    impurity balance -- and H98 is an OUTPUT, so it is a prediction that
    ITER reaches Q = 10 at roughly the confinement quality it assumes,
    rather than an input that guarantees it.
    """
    from .solver import Simulator, SolverConfig
    sim = Simulator(SolverConfig(machine="iter", equilibrium=True,
                                 eq_interval=8.0, disruption_enabled=False,
                                 pedestal_model="eped", n_rho=49))
    acc, n = {}, 0
    while sim.t < 180.0:
        sim._scenario_actuators()
        sim.step()
        if sim.t > 130.0:
            for k, v in sim.scalars().items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc[k] = acc.get(k, 0.0) + v
            n += 1
    d = {k: v / max(n, 1) for k, v in acc.items()}
    src = "интеграцияланған 1.5-өлшемді есеп"
    return [
        Check("1.5D: синтез қуаты", d["Pfus"], 500.0, "МВт", 8.0, src),
        Check("1.5D: Q", d["Q"], 10.0, "", 8.0, src),
        Check("1.5D: педестал T_ped", d["Tped"], 4.5, "кэВ", 10.0,
              "ITER педесталының болжамдары"),
        Check("1.5D: ⟨T_e⟩", d["Te"], 8.9, "кэВ", 8.0, src),
        Check("1.5D: ⟨T_i⟩", d["Ti"], 8.1, "кэВ", 8.0, src),
        Check("1.5D: осьтегі T_e0", d["Te0"], 25.0, "кэВ", 12.0, src),
        Check("1.5D: жылу энергиясы", d["Wth"], 325.0, "МДж", 8.0, src),
        Check("1.5D: τ_E", d["tauE"], 3.70, "с", 10.0, src),
        Check("1.5D: H98 (БОЛЖАМ, енгізілмеген)", d["H_factor"], 1.00, "",
              10.0, "ITER Q=10 болжамы",
              "енгізілген емес — модельдің өз нәтижесі"),
        Check("1.5D: β_N", d["betaN"], 1.77, "", 10.0, src),
        Check("1.5D: q₉₅", d["q95"], 3.00, "", 3.0, src),
        Check("1.5D: гелий күлі", d["fHePc"], 4.10, "%", 15.0, src),
        Check("1.5D: Z_eff", d["Zeff"], 1.65, "", 3.0, src),
        Check("1.5D: бутстрап үлесі", d["fBS"], 0.20, "", 40.0,
              "ITER сценарийлерінде 0.15-0.25"),
        Check("1.5D: Гринвальд үлесі", d["fG"], 0.845, "", 5.0, src),
    ]


# ---------------------------------------------------------------------------
#: What this package does NOT do.  Kept next to the checks on purpose.
NOT_VALIDATED = [
    "Ядролық тасымалдың критикалық градиенті R/L_T = 6.5 деп фиттелген. "
    "Гирокинетика оны 4-8 аралығында береді, сондықтан мән физикалық "
    "аралықта, бірақ бұл — есептелген емес, келтірілген сан. Ол — "
    "интеграцияланған есептегі жалғыз фиттелген ядролық тұрақты.",
    "Педесталдың баллон табалдырығы да бір нүктеге (ITER) келтірілген; "
    "оның орнына p_ped ~ Ip² масштабтауы тексеріледі.",
    "Град–Шафранов шешушісі бекітілген шекаралы: X-нүкте мен еркін "
    "шекара жоқ, сондықтан q₉₅ жүйелі түрде ~8% төмен.",
    "МГД тұрақтылығы есептелмейді: NTM, RWM, peeling-ballooning, ELM "
    "циклі — ешқайсысы жоқ.",
    "Турбуленттік тасымал модельдері эмпирикалық: гирокинетика жоқ.",
    "Нейтроника Монте-Карло емес, TBR — параметр.",
    "KZ-1-дің ST тармағы бір анкерлік нүктеге бекітілген, регрессия емес.",
    "Екі нүктелі модель детачмент ауысуын қайта шығармайды: ол себудің "
    "міндетті екенін және шамамен қанша керегін көрсетеді (10 МВт/м² үшін "
    "~90%), бірақ T_t бірнеше эВ-қа дейін құлайтын көлемдік рекомбинация "
    "мен импульс жоғалтуы модельденбеген.",
]


def all_checks() -> List[Check]:
    out: List[Check] = []
    for fn in (check_reactivity, check_geometry, check_equilibrium,
               check_bootstrap, check_sol, check_disruption,
               check_fuelcycle, check_kz1, check_pedestal,
               check_integrated):
        try:
            out.extend(fn())
        except Exception as exc:                      # pragma: no cover
            out.append(Check(f"{fn.__name__} қатесі", 0.0, 1.0, "", 0.0,
                             "тексеру орындалмады", str(exc)))
    return out


def report() -> dict:
    checks = all_checks()
    passed = sum(c.passed for c in checks)
    return {
        "checks": [c.to_dict() for c in checks],
        "n_total": len(checks),
        "n_passed": passed,
        "n_failed": len(checks) - passed,
        "mean_abs_deviation_pct": float(
            np.mean([abs(c.deviation) for c in checks])),
        "not_validated": NOT_VALIDATED,
    }


def main() -> int:
    r = report()
    print("\n=== КОМПОНЕНТТІК ВАЛИДАЦИЯ " + "=" * 52)
    print(f"{'Тексеру':46s}{'Модель':>13s}{'Анықтама':>13s}{'Ауытқу':>9s}  ")
    print("-" * 84)
    for c in r["checks"]:
        mark = " " if c["passed"] else "!"
        model = (f"{c['model']:.4g}")
        ref = (f"{c['reference']:.4g}")
        print(f"{c['name'][:45]:46s}{model:>13s}{ref:>13s}"
              f"{c['deviation_pct']:>8.2f}% {mark}")
    print("-" * 84)
    print(f"Өтті: {r['n_passed']}/{r['n_total']}   "
          f"орташа |ауытқу| = {r['mean_abs_deviation_pct']:.2f}%")
    print("\n=== ВАЛИДАЦИЯЛАНБАҒАН " + "=" * 57)
    for line in r["not_validated"]:
        print("  · " + line)
    print()
    return 0 if r["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
