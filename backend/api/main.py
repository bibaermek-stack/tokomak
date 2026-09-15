"""HTTP API for the tokamak package.

Endpoints
    GET  /machines                 registered machines
    GET  /machines/{key}           one machine, with its derived geometry
    GET  /models                   every selectable model, with provenance
    POST /equilibrium              Grad-Shafranov solve
    POST /simulate                 run a discharge, return the time trace
    POST /compare                  the same scenario across several machines
    POST /divertor                 two-point SOL solve
    POST /disruption               post-disruption chain
    POST /plant                    fuel cycle and plant power balance
    GET  /validate                 component validation report
    GET  /health
    GET  /ai/health                advisor availability (no model name)
    GET  /ai/imas                  IMAS catalog search
    POST /ai/chat                  advisor: advice + confirm-only proposals
    GET  /ai/files                 sandboxed advisor files
    GET  /ai/files/{name}          download one advisor file

Model selection is the organising idea: ``/models`` lists what can be
chosen, every request names the models it wants, and every response echoes
them back, so a result always carries the assumptions that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tokamak import disruption as disr
from tokamak import equilibrium as eqm
from tokamak import fuelcycle as fc
from tokamak import mhd
from tokamak import sol as sol_mod
from tokamak import transport as tr
from tokamak import validation as val
from tokamak.geometry import attach_geometry, b_poloidal
from tokamak.machines import custom_machine, get_machine, list_machines
from tokamak.scalings import (CONFINEMENT, ConfinementInputs, LH_THRESHOLD,
                              ThresholdInputs, model_catalogue,
                              p_lh_threshold, tau_energy)
from tokamak.solver import Simulator, SolverConfig

app = FastAPI(
    title="Tokamak physics API",
    version="1.0",
    description=__doc__,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)


def jsonable(obj):
    """Convert numpy scalars and arrays to plain Python for serialisation."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
class MachineOverride(BaseModel):
    """Any machine parameter can be overridden; the rest comes from ``base``."""
    base: str = "iter"
    label: Optional[str] = None
    R0: Optional[float] = None
    a: Optional[float] = None
    kappa_x: Optional[float] = None
    delta_x: Optional[float] = None
    kappa_95: Optional[float] = None
    delta_95: Optional[float] = None
    B0: Optional[float] = None
    Ip: Optional[float] = None
    p_nbi: Optional[float] = None
    p_icrf: Optional[float] = None
    p_ecrf: Optional[float] = None
    a_mass: Optional[float] = None
    fuel: Optional[str] = None


def _resolve_machine(key: str, override: Optional[MachineOverride]):
    try:
        if override is not None:
            kw = {k: v for k, v in override.model_dump().items()
                  if v is not None and k != "base"}
            m = custom_machine(base=override.base, **kw)
        else:
            m = get_machine(key)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from None
    return attach_geometry(m)


class SimulateRequest(BaseModel):
    machine: str = "iter"
    override: Optional[MachineOverride] = None
    confinement_model: str = Field("ipb98y2", description="/models ішінен")
    lh_model: str = "martin08"
    transport_model: str = "scaling_anchored"
    h_factor: float = 1.0
    t_end: float = 120.0
    record_every: float = 1.0
    n_rho: int = 65
    equilibrium: bool = True
    eq_interval: float = 6.0
    seeding_fraction: Optional[float] = None
    disruption_enabled: bool = True
    pedestal_model: str = Field("eped", description="eped | anchor")
    alpha_crit: Optional[float] = Field(
        None, description="педесталдың баллон шегі; None — үнсіз келісім")
    mhd_enabled: bool = True
    sawteeth: bool = True
    ntm_enabled: bool = True
    elm_cycle: bool = False
    eccd_ntm: float = Field(0.0, ge=0.0, le=2.0,
                            description="j_cd/j_bs 2/1 бетінде")


class EquilibriumRequest(BaseModel):
    machine: str = "iter"
    override: Optional[MachineOverride] = None
    Ip: Optional[float] = None
    p_avg: Optional[float] = Field(None, description="мақсатты ⟨p⟩ [Па]")
    q0: Optional[float] = Field(None, description="мақсатты q₀")
    n_r: int = 97
    n_z: int = 145


class StabilityRequest(BaseModel):
    """Stability of one operating point, without running a discharge.

    The point can be given directly (beta_N, li, q profile) or, more
    usefully, taken from a short simulation of the named machine -- the
    profiles matter, since the ballooning and tearing drives are both
    gradient quantities and a made-up q profile answers a made-up question.
    """
    machine: str = "iter"
    override: Optional[MachineOverride] = None
    t_end: float = Field(60.0, ge=1.0, le=600.0,
                         description="нүктені алу үшін есептеу ұзақтығы")
    n_rho: int = 49
    equilibrium: bool = True
    eccd_ntm: float = Field(0.0, ge=0.0, le=2.0)
    mitigated_elms: bool = True
    seed_island_cm: float = Field(
        0.0, ge=0.0, le=50.0,
        description="2/1 бетіне енгізілетін тұқым арал; 0 — енгізілмейді")
    beta_n_scan: Optional[List[float]] = Field(
        None, description="осы β_N мәндері бойынша шектерді сканерлеу")


class DivertorRequest(BaseModel):
    machine: str = "iter"
    P_sep: float = 90.0
    n_sep20: float = 0.30
    q95: float = 3.0
    f_rad_seed: float = 0.0
    q_limit: Optional[float] = Field(None, description="берілсе, қажет себуді табады")


class DisruptionRequest(BaseModel):
    machine: str = "iter"
    Ip: Optional[float] = None
    W_thermal: float = 325.0
    li: float = 0.9
    ne20: float = 1.0
    z_eff: float = 1.65
    mitigated: bool = False


class PlantRequest(BaseModel):
    machine: str = "iter"
    P_fusion: float = 500.0
    P_aux: float = 50.0
    thermal_efficiency: float = 0.35
    tbr_design: float = 1.15


class CompareRequest(BaseModel):
    machines: List[str] = ["iter", "t15md", "nstx", "ktm"]
    confinement_model: str = "kz1"
    lh_model: str = "martin08"
    density_fraction: float = 0.5


class ChatRequest(BaseModel):
    message: str = ""
    mode: str = "live"
    snapshot: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[str]] = None


# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ai/health")
def ai_health() -> dict:
    from ai.operator import health as advisor_health
    return advisor_health()


@app.get("/ai/imas")
def ai_imas(q: str = "", limit: int = 12) -> dict:
    from imas.catalog import search as imas_search
    hits = imas_search(q, limit=limit)
    return {"query": q, "hits": hits, "count": len(hits)}


@app.post("/ai/chat")
def ai_chat(req: ChatRequest) -> dict:
    from ai.operator import chat as advisor_chat, health as advisor_health
    msg = (req.message or "").strip()
    atts = [a for a in (req.attachments or []) if isinstance(a, str)]
    if not msg and not atts:
        raise HTTPException(400, "empty message")
    if len(msg) > 4000:
        raise HTTPException(400, "message too long")
    if not msg:
        msg = "осы файлды қара"
    if not advisor_health()["ai_available"]:
        return JSONResponse(status_code=503, content={
            "reply": "Кеңесші әлі қосылмаған.",
            "proposals": [],
            "citations": [],
            "mode": "live",
            "ai_available": False,
            "simulate": None,
        })
    return jsonable(advisor_chat(msg, "live", req.snapshot, req.history, atts))


@app.post("/ai/upload")
async def ai_upload(file: UploadFile = File(...)) -> dict:
    from ai.files import save_upload
    data = await file.read()
    out = save_upload(file.filename or "upload.bin", data)
    if not out.get("ok"):
        raise HTTPException(400, out.get("error") or "upload failed")
    return out


@app.get("/ai/files")
def ai_files() -> dict:
    from ai.files import list_files
    return {"files": list_files()}


@app.get("/ai/files/{name}")
def ai_file(name: str):
    from ai.files import resolve
    path = resolve(name)
    if path is None or not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path, filename=path.name)


@app.get("/machines")
def machines() -> dict:
    out = []
    for m in list_machines().values():
        out.append(attach_geometry(m).to_dict())
    return jsonable({"machines": out, "count": len(out)})


@app.get("/machines/{key}")
def machine(key: str) -> dict:
    return jsonable(_resolve_machine(key, None).to_dict())


@app.get("/models")
def models() -> dict:
    cat = model_catalogue()
    cat["transport"] = [m.to_dict() for m in tr.TRANSPORT_MODELS.values()]
    cat["notes"] = {
        "kz1": "осы жобаның өз моделі — аспект қатынасы бойынша көпір",
        "kz_lh": "осы жобаның өз моделі — төмен тығыздық тармағы қосылған",
    }
    return jsonable(cat)


@app.post("/equilibrium")
def equilibrium(req: EquilibriumRequest) -> dict:
    m = _resolve_machine(req.machine, req.override)
    try:
        eq = eqm.solve_matched(m.R0, m.a, m.kappa_x, m.delta_x,
                               req.Ip if req.Ip is not None else m.Ip, m.B0,
                               p_avg=req.p_avg, q0=req.q0,
                               nR=req.n_r, nZ=req.n_z)
    except Exception as exc:
        raise HTTPException(500, f"equilibrium solve failed: {exc}") from None
    return jsonable({
        "machine": m.label,
        "summary": eq.summary(),
        "profiles": {"rho": eq.rho.tolist(), "q": eq.q.tolist(),
                     "psi_n": eq.psi_n_grid.tolist(),
                     "V": eq.V_of_rho.tolist()},
        "caveat": "бекітілген шекаралы есеп: X-нүкте жоқ, q₉₅ ~8% төмен",
    })


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict:
    if req.confinement_model not in CONFINEMENT:
        raise HTTPException(400, f"unknown confinement model "
                                 f"{req.confinement_model!r}")
    if req.lh_model not in LH_THRESHOLD:
        raise HTTPException(400, f"unknown L-H model {req.lh_model!r}")
    if req.transport_model not in tr.TRANSPORT_MODELS:
        raise HTTPException(400, f"unknown transport model "
                                 f"{req.transport_model!r}")
    m = _resolve_machine(req.machine, req.override)
    cfg = SolverConfig(
        machine=req.machine, confinement_model=req.confinement_model,
        lh_model=req.lh_model, transport_model=req.transport_model,
        h_factor=req.h_factor, n_rho=req.n_rho,
        equilibrium=req.equilibrium, eq_interval=req.eq_interval,
        seeding_fraction=req.seeding_fraction,
        disruption_enabled=req.disruption_enabled,
        pedestal_model=req.pedestal_model, alpha_crit=req.alpha_crit,
        mhd_enabled=req.mhd_enabled, sawteeth=req.sawteeth,
        ntm_enabled=req.ntm_enabled, elm_cycle=req.elm_cycle,
        eccd_ntm=req.eccd_ntm)
    sim = Simulator(cfg, machine=m)
    trace = sim.run_scenario(t_end=req.t_end, record_every=req.record_every)
    out = {
        "machine": m.to_dict(),
        "models": cfg.to_dict(),
        "trace": trace,
        "final": sim.scalars(),
        "profiles": sim.profiles(),
        "plant": sim.plant(),
        "alarms": sim.alarms,
        "caveat": val.NOT_VALIDATED[0],
    }
    if sim.stab is not None:
        out["stability"] = sim.stab.to_dict()
    if sim.disruption is not None:
        out["disruption"] = sim.disruption.to_dict()
    return jsonable(out)


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    rows = []
    for key in req.machines:
        m = _resolve_machine(key, None)
        x = ConfinementInputs(Ip=m.Ip, B0=m.B0, P_loss=max(m.p_aux, 1.0),
                              n_bar20=req.density_fraction * m.n_greenwald,
                              R0=m.R0, a=m.a, kappa_a=m.kappa_a,
                              a_mass=m.a_mass)
        th = ThresholdInputs(n_bar20=req.density_fraction * m.n_greenwald,
                             B0=m.B0, S=m.S, R0=m.R0, a=m.a,
                             a_mass=m.a_mass)
        taus = {k: tau_energy(k, x) for k in CONFINEMENT}
        rows.append({
            "key": m.key, "label": m.label, "country": m.country,
            "org": m.org, "first_plasma": m.first_plasma, "fuel": m.fuel,
            "R0": m.R0, "a": m.a, "aspect": m.aspect, "eps": m.eps,
            "kappa_x": m.kappa_x, "B0": m.B0, "Ip": m.Ip, "V": m.V,
            "S": m.S, "kappa_a": m.kappa_a, "n_greenwald": m.n_greenwald,
            "p_aux": m.p_aux, "pulse_s": m.pulse_s, "purpose": m.purpose,
            "superconducting": m.superconducting,
            "tau_e": taus,
            "tau_selected": taus[req.confinement_model],
            "validity": {k: CONFINEMENT[k].in_range(m.eps)
                         for k in CONFINEMENT},
            "p_lh": p_lh_threshold(req.lh_model, th),
        })
    return jsonable({"rows": rows, "confinement_model": req.confinement_model,
                     "lh_model": req.lh_model,
                     "density_fraction": req.density_fraction})


@app.post("/stability")
def stability(req: StabilityRequest) -> dict:
    """МГД тұрақтылығы: Тройон/RWM шектері, баллондық шекара, NTM, пилообразный, ELM."""
    m = _resolve_machine(req.machine, req.override)
    cfg = SolverConfig(machine=req.machine, n_rho=req.n_rho,
                       equilibrium=req.equilibrium, eq_interval=8.0,
                       disruption_enabled=False, eccd_ntm=req.eccd_ntm,
                       mhd_enabled=True)
    sim = Simulator(cfg, machine=m)
    while sim.t < req.t_end:
        sim._scenario_actuators()
        sim.step()
    if sim.stab is None:
        raise HTTPException(500, "тұрақтылық есептелмеді")

    out = {
        "machine": m.to_dict(),
        "point": {k: sim.scalars()[k] for k in
                  ("t", "Ip", "betaN", "betaP", "li", "q0", "q95", "Wth",
                   "Psep", "tauE", "Te")},
        "stability": sim.stab.to_dict(),
        "profiles": {"rho": sim.grid.rho.tolist(),
                     "q": sim.q_prof.tolist(),
                     "shear": sim.stab.shear.tolist(),
                     "alpha": sim.stab.alpha.tolist(),
                     "alpha_crit": sim.stab.alpha_crit.tolist()},
        "alarms": sim.alarms,
    }

    # what a seed island of the requested size would do: below the seed
    # threshold it heals, above it the mode runs away to its saturated width
    if req.seed_island_cm > 0:
        seeded = []
        for ntm in sim.stab.ntms:
            w0 = req.seed_island_cm / 100.0
            seeded.append({"mode": ntm.key, "seed_m": w0,
                           "seed_threshold_m": ntm.seed_threshold(),
                           "dwdt_at_seed": ntm.dwdt(w0),
                           "grows": bool(ntm.dwdt(w0) > 0.0),
                           "saturated_m": ntm.saturated_width()})
        out["seeded"] = seeded

    if req.beta_n_scan:
        scan = []
        li = sim.li
        no_wall = mhd.troyon_limit(li)
        ideal = mhd.wall_stabilised_limit(no_wall)
        for bn in req.beta_n_scan:
            scan.append({
                "beta_n": bn,
                "troyon_fraction": bn / max(no_wall, 1e-6),
                "c_beta": mhd.rwm_margin(bn, no_wall, ideal),
                "no_wall_limit": no_wall,
                "ideal_wall_limit": ideal})
        out["beta_n_scan"] = scan
    return jsonable(out)


@app.post("/divertor")
def divertor(req: DivertorRequest) -> dict:
    m = _resolve_machine(req.machine, None)
    B_pol = b_poloidal(m.Ip, m.L_pol)
    kw = dict(R0=m.R0, a=m.a, B0=m.B0, B_pol=B_pol, q95=req.q95,
              n_sep20=req.n_sep20, A_mass=m.a_mass)
    st = sol_mod.two_point(req.P_sep, f_rad_seed=req.f_rad_seed, **kw)
    out = {"machine": m.label, "state": st.to_dict()}
    if req.q_limit is not None:
        out["seeding_required"] = sol_mod.seeding_for_target(
            req.P_sep, q_limit=req.q_limit, **kw)
    return jsonable(out)


@app.post("/disruption")
def disruption(req: DisruptionRequest) -> dict:
    m = _resolve_machine(req.machine, None)
    r = disr.simulate(cause="API", Ip=req.Ip if req.Ip is not None else m.Ip,
                      R0=m.R0, a=m.a, kappa=m.kappa_x, B0=m.B0,
                      W_thermal=req.W_thermal, li=req.li, ne20=req.ne20,
                      z_eff=req.z_eff, S_wall=m.S, mitigated=req.mitigated)
    return jsonable({"machine": m.label, "result": r.to_dict(),
                     "vertical_margin": disr.vertical_stability_margin(
                         m.kappa_x, req.li)})


@app.post("/plant")
def plant(req: PlantRequest) -> dict:
    m = _resolve_machine(req.machine, None)
    b = fc.blanket(req.P_fusion, m.S, tbr_design=req.tbr_design)
    p = fc.plant_balance(req.P_fusion, req.P_aux,
                         thermal_efficiency=req.thermal_efficiency,
                         superconducting=m.superconducting, R0=m.R0)
    return jsonable({"machine": m.label, "blanket": b.to_dict(),
                     "plant": p.to_dict(),
                     "damage": fc.neutron_damage(b.wall_load)})


@app.get("/validate")
def validate() -> dict:
    return jsonable(val.report())


_REPO = Path(__file__).resolve().parents[2]
if (_REPO / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(_REPO), html=True), name="site")
