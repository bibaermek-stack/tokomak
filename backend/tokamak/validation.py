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
        Check("ГШ: q₉₅ (бекітілген шекара)", eq.q95, 3.00, "", 5.0,
              "ITER жоба",
              "БЕЛГІЛІ ШЕКТЕУ: X-нүктесіз бекітілген шекара q₉₅-ті "
              "+3.7% асыра береді. Бұл түзетілмейді — ол көрініп тұруы "
              "керек, әрі оның педесталға әсері pedestal.ALPHA_CRIT "
              "түсіндірмесінде сан түрінде жазылған."),
        Check("ГШ: q₉₅-ке түзету коэффициенті жоқ", eqm.Q95_XPOINT, 1.00,
              "", 0.01, "1.109 фудж-коэффициенті алынып тасталды",
              "ол шын мәнінде сплайн тегістеу қатесін жасырып тұрған"),
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
    # evaluated at the q95 the equilibrium solver actually produces, which
    # is the value the calibration saw; at ITER's true q95 = 3.00 the same
    # call returns 5.16 keV, and the difference IS the equilibrium residual
    q95_model, q95_iter = 3.1115, 3.00
    base = dict(R0=m.R0, a=m.a, kappa_a=m.kappa_a, B0=m.B0, Ip=m.Ip,
                q95=q95_model, n_ped20=0.75, f_ion=0.87, L_pol=m.L_pol)
    p = ped.solve(**base)
    # what the ballooning limit would have to be at the true q95 to give
    # the same pedestal: p_ped ~ q95^-4 and T_ped ~ alpha_crit^2, so this
    # is ALPHA_CRIT / (q95_model/q95_iter)^2 and it must come back to the
    # pre-recalibration value.  If it does not, the recalibration absorbed
    # something other than the q95 residual and needs explaining.
    alpha_at_true_q95 = ped.ALPHA_CRIT / (q95_model / q95_iter) ** 2
    return [
        Check("Педестал: ITER T_ped", p.T_ped, 4.50, "кэВ", 5.0,
              "ITER педестал болжамдары",
              "тізбек ішінде калибрленген; модельдің өз q₉₅-інде"),
        Check("Педестал: α_crit шын q₉₅-те", alpha_at_true_q95, 13.40, "",
              2.0, "қайта калибрлеу q₉₅ қалдығына тең болуы керек",
              "ЕСЕП: 14.45 / (3.1115/3.00)² — бұл 13.40-қа қайтуы тиіс, "
              "яғни түзету q₉₅ қатесінен басқа ештеңені сіңірмеген"),
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


def check_mhd() -> List[Check]:
    """MHD stability: each limit against the published number it comes from.

    These are checks on the *implementations*, not on the constants: a
    Troyon limit that returns 4 li is only interesting if the profile it is
    handed produces the li that ITER has, and a Kadomtsev mixing radius is
    only interesting if it conserves helical flux on a real q profile.  So
    most of the references below are geometric or analytic consequences
    rather than measured values -- the measured ones are marked as such.
    """
    from . import mhd
    m = geo.attach_geometry(get_machine("iter"))

    # a representative ITER-like q profile: q0 = 0.85, q95 = 2.97, so that
    # both the q = 1 and the q = 2 surface exist and sit where they should
    rho = np.linspace(0.0, 1.0, 201)
    q = 0.85 + 2.35 * rho ** 2

    # -- analytic ballooning alpha ------------------------------------------
    # p = p0 (1 - rho^2) gives dp/dr = -2 p0 rho / a, so
    # alpha = 2 mu0 R q^2 / B^2 * 2 p0 rho / a, evaluated at rho = 0.5.
    p0 = 5.0e5
    p = p0 * (1.0 - rho ** 2)
    alpha = mhd.ballooning_alpha(rho, p, q, m.R0, m.a, m.B0)
    i = int(np.argmin(abs(rho - 0.5)))
    mu0 = 4.0e-7 * np.pi
    q_half = 0.85 + 2.35 * 0.25
    alpha_exact = (2.0 * mu0 * m.R0 * q_half ** 2 / m.B0 ** 2
                   * 2.0 * p0 * 0.5 / m.a)

    r1, r_mix = mhd.kadomtsev_mixing_radius(rho, q)

    # -- helical flux conservation, the defining property of the crash ------
    rr = rho[rho <= r_mix]
    helical = float(np.trapezoid((1.0 / np.interp(rr, rho, q) - 1.0) * rr, rr))

    # -- NTM at the ITER operating point ------------------------------------
    r_21 = mhd.find_rational_surface(rho, q, 2, 1)
    L_q = r_21 * m.a / max(float(np.interp(r_21, rho, mhd.magnetic_shear(rho, q))),
                           1e-3)
    w_d = mhd.marginal_island_width(r_21 * m.a, L_q)
    # the pressure scale length has to come from the same profile as the
    # ballooning alpha above: L_p is what sets the bootstrap drive, and
    # assuming a round number for it silently moves the NTM threshold.
    dpdr = float(np.gradient(p, rho * m.a)[
        int(np.argmin(abs(rho - r_21)))])
    L_p = abs(float(np.interp(r_21, rho, p)) / dpdr)

    def _ntm(beta_p: float, eccd: float = 0.0) -> mhd.NTM:
        return mhd.NTM(m=2, n=1, r_s=r_21 * m.a, rho_s=r_21, beta_p=beta_p,
                       lq_over_lp=L_q / L_p, eps=r_21 * m.a / m.R0,
                       w_d=w_d, tau_r=10.0, eccd_drive=eccd)

    # beta_p at which the 2/1 island stops healing: bisected, not assumed
    lo, hi = 0.2, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _ntm(mid).saturated_width() > 0.0:
            hi = mid
        else:
            lo = mid
    beta_p_onset = 0.5 * (lo + hi)
    w_sat = _ntm(1.0).saturated_width()
    w_seed = _ntm(1.0).seed_threshold()
    w_sat_cd = _ntm(1.0, eccd=1.0).saturated_width()

    # -- ELMs ----------------------------------------------------------------
    elm_nat = mhd.elm_state(W_th=325.0, P_sep=100.0, ped_width_psi=0.04,
                            mitigated=False)
    elm_mit = mhd.elm_state(W_th=325.0, P_sep=100.0, ped_width_psi=0.04,
                            mitigated=True)

    src_geo = "аналитикалық салдар"
    return [
        Check("МГД: баллондық α аналитикалық түрде", float(alpha[i]),
              alpha_exact, "", 2.0, src_geo,
              "p = p₀(1−ρ²) үшін тұйық түрде есептеледі"),
        Check("МГД: Тройон шегі β_N ≤ 4 lᵢ", mhd.troyon_limit(0.9), 3.60, "",
              1.0, "Troyon 1984, lᵢ-масштабты практикалық түрі"),
        Check("МГД: идеал қабырға бетаны көтеру", 
              mhd.wall_stabilised_limit(3.6) / 3.6, 1.30, "", 5.0,
              "өткізгіш қабырға шегін 30–50% көтереді"),
        Check("МГД: RWM C_β шектен тыс β_N-де", 
              mhd.rwm_margin(4.68, 3.6, 4.68, rotation_stabilised=False),
              1.00, "", 1.0, "анықтама бойынша идеал қабырғада C_β = 1"),
        Check("МГД: q=1 беті", r1, float(np.sqrt(0.15 / 2.35)), "", 2.0,
              src_geo, "√((1−q₀)/c) болуы керек"),
        Check("МГД: Кадомцев r_mix / r₁", r_mix / max(r1, 1e-6), 1.40, "",
              10.0, "Кадомцев 1975: типтік мән ≈ 1.4"),
        Check("МГД: құлаудан кейінгі спираль ағыны", helical, 0.0, "", 2.0,
              "Кадомцев құлауының анықтаушы шарты",
              "∫(1/q−1) r dr = 0 болуы керек — абсолют ауытқу"),
        Check("МГД: 2/1 беті q = 2-де", r_21, float(np.sqrt(1.15 / 2.35)),
              "", 2.0, src_geo),
        Check("МГД: маргиналды арал ені w_d", w_d * 100.0, 2.20, "см", 40.0,
              "ITER болжамдары: бірнеше см"),
        Check("МГД: 2/1 қозу табалдырығы β_p", beta_p_onset, 0.80, "", 30.0,
              "2/1 NTM тәжірибеде β_N ≈ 1.5–2 маңында қозады",
              "бисекциямен табылған — енгізілмеген"),
        Check("МГД: 2/1 қаныққан ені, β_p = 1.0", w_sat * 100.0, 10.0, "см",
              40.0, "ITER 2/1 NTM болжамдары: 5–15 см"),
        Check("МГД: тұқым табалдырығы / w_d", w_seed / max(w_d, 1e-9), 1.30,
              "", 30.0,
              "тұқым арал маргиналды еннен үлкен болуы керек"),
        Check("МГД: ECCD толық басу (j_cd = j_bs)", w_sat_cd * 100.0, 0.0,
              "см", 0.5, "бутстрап тогының орнын толық басу модаға "
              "қозуға мүмкіндік қалдырмайды"),
        Check("МГД: I типті ELM жоғалтуы", elm_nat.energy_loss, 19.5, "МДж",
              15.0, "ITER-де басылмаған ELM ~20 МДж"),
        Check("МГД: ELM жиілігі, басылмаған", elm_nat.frequency, 1.54, "Гц",
              20.0, "ITER-де ~1 Гц шамасында"),
        Check("МГД: басу жиілікті көтеру еселігі",
              elm_mit.frequency / max(elm_nat.frequency, 1e-6), 6.00, "",
              10.0, "ITER ELM бақылауы: жиілікті 5–10 есе көтеру"),
        Check("МГД: басу орташа қуатты өзгертпейді",
              elm_mit.power_to_target / max(elm_nat.power_to_target, 1e-6),
              1.00, "", 1.0,
              "басу тек өлшемді жиілікке айырбастайды — қуат сол күйінде"),
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
        Check("1.5D: q₉₅", d["q95"], 3.00, "", 5.0, src,
              "БЕЛГІЛІ ҚАЛДЫҚ: бекітілген шекара X-нүктесіз, +4% — осы "
              "қалдық pedestal.ALPHA_CRIT-ке сіңірілген, бірақ мұнда "
              "көрініп тұр"),
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
    "Педесталдың баллон табалдырығы бір нүктеге (ITER-дің Q=10 режимі) "
    "толық тізбек ішінде келтірілген, сондықтан ол тізбектің q₉₅ "
    "қалдығын сіңіреді. Сіңірілген шама өлшенген әрі жазылған "
    "(pedestal.ALPHA_CRIT), q₉₅-тің өзі бөлек тексеру ретінде қалады, "
    "ал масштабтау (p_ped ~ Ip²) калибрленбейді — өлшенеді.",
    "Град–Шафранов шешушісі бекітілген шекаралы: X-нүкте мен еркін "
    "шекара жоқ, сондықтан q₉₅ ~3.7% жоғары шығады. Бұл түзетілмейді: "
    "бұрын дәл осы орынға қойылған 1.109 коэффициенті сплайн тегістеу "
    "қатесін екі жыл бойы көрінбейтін етіп жасырған еді.",
    "МГД тұрақтылығы редукцияланған модельдермен есептеледі: Тройон/RWM "
    "шектері, s–α баллондық шекара, модификацияланған Резерфорд теңдеуі "
    "(NTM), Кадомцев қайта қосылуы (пилообразный) және ELM қуат балансы "
    "бар. Идеал МГД меншікті мәндер есептеуіші (DCON/GATO типті), "
    "Мерсье мен резистивті интерченж критерийлері, тороидалды мода "
    "байланысы және peeling-ballooning-тің n-спектрі ЖОҚ — "
    "NTM коэффициенттері (a_bs, a_gg) әдебиеттегі типтік мәндер, "
    "фиттелген емес.",
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
               check_fuelcycle, check_kz1, check_pedestal, check_mhd,
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
