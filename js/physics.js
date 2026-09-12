/* ==========================================================================
   physics.js — 0-D tokamak transport / burn model
   --------------------------------------------------------------------------
   Volume-averaged energy + particle balance with reconstructed radial
   profiles.  Everything the HUD displays is derived from this state, so the
   graphs and the plasma render stay physically consistent with each other.

   The machine description is a preset (ITER by default); the geometry is
   integrated from the plasma boundary rather than assumed elliptical, so
   the same code runs for a spherical torus or a mid-size device.

   Conventions
     <n>   volume average          (1/V) int n dV
     <T>   density-weighted average  int n T dV / int n dV
   The second convention is the one under which W = (3/2)(<n_e><T_e> +
   <n_i><T_i>) V is exact for peaked profiles, and it is the convention the
   published machine parameters below are quoted in.

   References for the parametrisations used here:
     Bosch & Hale, Nucl. Fusion 32 (1992) 611        — D-T reactivity
     ITER Physics Basis, Nucl. Fusion 39 (1999) 2175 — IPB98(y,2), Q=10 point
     Uckan & Sheffield, ITER Physics Design Guide (1990) — q95 shape factor
     Martin et al., J. Phys. Conf. Ser. 123 (2008) 012033 — L-H threshold
     Greenwald, Plasma Phys. Control. Fusion 44 (2002) R27 — density limit
     Stix, Plasma Phys. 14 (1972) 367                — fast-ion slowing down
     Wesson, Tokamaks 3rd ed. §14                    — neoclassical resistivity
     Albajar et al., Nucl. Fusion 41 (2001) 665      — synchrotron losses
     Puetterich et al., Nucl. Fusion 50 (2010) 025012 — tungsten cooling rate
   ========================================================================== */
(function (global) {
  'use strict';

  /* ---------- physical constants ---------------------------------------- */
  const KEV_J     = 1.602176634e-16;   // 1 keV in joules
  const E_FUS     = 2.8183e-12;        // 17.59 MeV per D-T reaction  [J]
  const E_ALPHA   = 3520;              // alpha birth energy          [keV]
  const E_NEUT    = 14070;             // neutron energy              [keV]
  const E_ALPHA_F = E_ALPHA / (E_ALPHA + E_NEUT);   // 0.2001
  const MU0       = 4 * Math.PI * 1e-7;
  const MW        = 1e6;
  /* (3/2) k_B in MJ per (1e20 m^-3 . keV . m^3) — the energy-content unit */
  const W_UNIT    = 1.5 * KEV_J * 1e20 / MW;        // 2.4033e-2

  /* ======================================================================= */
  /*  Machine presets.                                                       */
  /*  kappaX / deltaX describe the separatrix (what the renderer draws);     */
  /*  kappa95 / delta95 are the 95% flux surface, which is what the q95 and  */
  /*  L-H parametrisations were fitted against.                              */
  /* ======================================================================= */
  const MACHINES = {
    iter: {
      key: 'iter', label: 'ITER',
      R0: 6.20, a: 2.00, kappaX: 1.85, deltaX: 0.45, kappa95: 1.70, delta95: 0.33,
      BtNom: 5.30, IpNom: 15.00, Amass: 2.50, nTF: 18, nPF: 6, nCS: 6,
      Sdiv: 3.32, diverted: true, dt: true
    },
    /* T-15MD — Kurchatov Institute, first plasma 2021 */
    t15md: {
      key: 't15md', label: 'T-15MD',
      R0: 1.48, a: 0.67, kappaX: 1.80, deltaX: 0.40, kappa95: 1.70, delta95: 0.30,
      BtNom: 2.00, IpNom: 2.00, Amass: 2.00, nTF: 16, nPF: 8, nCS: 1,
      Sdiv: 0.9, diverted: true, dt: false
    },
    /* NSTX — spherical torus, PPPL, 1999-2012 */
    nstx: {
      key: 'nstx', label: 'NSTX',
      R0: 0.85, a: 0.67, kappaX: 2.20, deltaX: 0.60, kappa95: 2.00, delta95: 0.45,
      BtNom: 0.45, IpNom: 1.00, Amass: 2.00, nTF: 12, nPF: 8, nCS: 1,
      Sdiv: 0.6, diverted: true, dt: false
    },
    /* KTM — material-testing tokamak, NNC RK, Kurchatov (Kazakhstan) */
    ktm: {
      key: 'ktm', label: 'KTM',
      R0: 0.86, a: 0.43, kappaX: 1.70, deltaX: 0.40, kappa95: 1.60, delta95: 0.30,
      BtNom: 1.00, IpNom: 0.75, Amass: 2.00, nTF: 12, nPF: 6, nCS: 1,
      Sdiv: 0.3, diverted: true, dt: false
    }
  };

  /* --------------------------------------------------------------------- */
  /*  Plasma boundary geometry.                                            */
  /*  R(t) = R0 + a cos(t + asin(delta) sin t),  Z(t) = kappa a sin t.      */
  /*  Volume from Pappus/Green (V = |int pi R^2 dZ|), surface from the      */
  /*  revolved perimeter.  A diverted plasma loses the private-flux region  */
  /*  below the X-point, which the smooth parametrisation still counts —    */
  /*  X_PT trims it and reproduces ITER's published 840 m^3 / 680 m^2.      */
  /* --------------------------------------------------------------------- */
  const X_PT = 0.988;

  function geometry(m) {
    const N = 4096, ds = Math.asin(m.deltaX);
    let V = 0, S = 0, Lp = 0;
    let R = m.R0 + m.a, Z = 0;
    for (let i = 1; i <= N; i++) {
      const th = 2 * Math.PI * i / N;
      const R2 = m.R0 + m.a * Math.cos(th + ds * Math.sin(th));
      const Z2 = m.kappaX * m.a * Math.sin(th);
      const seg = Math.hypot(R2 - R, Z2 - Z);
      V += Math.PI * (R * R + R2 * R2) / 2 * (Z2 - Z);
      S += Math.PI * (R + R2) * seg;
      Lp += seg;
      R = R2; Z = Z2;
    }
    const trim = m.diverted ? X_PT : 1;
    const g = { V: Math.abs(V) * trim, S: S * trim, Lp: Lp };
    g.eps = m.a / m.R0;
    /* areal elongation — the elongation IPB98(y,2) is written in terms of */
    g.kappaA = g.V / (2 * Math.PI * Math.PI * m.R0 * m.a * m.a);
    return g;
  }

  /* ---------- Bosch-Hale D-T fusion reactivity <sigma v> ----------------- */
  const BH = {
    Bg: 34.3827, mrc2: 1124656,
    C1: 1.17302e-9,  C2: 1.51361e-2, C3: 7.51886e-2, C4: 4.60643e-3,
    C5: 1.35000e-2,  C6: -1.06750e-4, C7: 1.36600e-5
  };
  function reactivityDT(T) {                 // T [keV] -> <sigma v> [m^3/s]
    if (T < 0.2) return 0;
    if (T > 100) T = 100;
    const t3 = T * T * T;
    const theta = T / (1 - (T * (BH.C2 + T * (BH.C4 + T * BH.C6))) /
                           (1 + T * (BH.C3 + T * (BH.C5 + T * BH.C7))));
    const xi = Math.pow(BH.Bg * BH.Bg / (4 * theta), 1 / 3);
    const sv = BH.C1 * theta * Math.sqrt(xi / (BH.mrc2 * t3)) * Math.exp(-3 * xi);
    return sv * 1e-6;                        // cm^3/s -> m^3/s
  }

  /* ---------- impurity inventory ----------------------------------------
     Fractions relative to n_e, matching what ITER's Q=10 point assumes:
     beryllium eroded off the first wall, neon seeded to radiate in the
     divertor, and a trace of tungsten sputtered off the divertor targets.
     Above ~4 keV Be and Ne are fully stripped and only contribute
     bremsstrahlung; tungsten never is, so it is the only species that
     line-radiates from the core.  <Z> for tungsten at 8-25 keV is ~60,
     not 74 (Puetterich 2010).                                             */
  const IMP = [
    { name: 'Be', Z: 4,  A: 9,   frac: 0.02000 },
    { name: 'Ne', Z: 10, A: 20,  frac: 0.00214 },
    { name: 'W',  Z: 60, A: 184, frac: 3.80e-5 }
  ];
  /* Tungsten cooling rate [W m^3].  Nearly flat across the core
     temperature range and climbing towards the cooler pedestal, following
     the ADAS curve of Puetterich (2010).  The plateau is what keeps a
     tungsten-seeded burn thermally stable; a steeper low-T branch drives
     the radiative collapse the disruption logic below watches for. */
  function coolW(Te) {
    if (Te <= 0.2) return 0;
    return Te >= 3 ? 4.3e-32 : 4.3e-32 * Math.pow(3 / Te, 0.55);
  }

  /* Ion-to-electron energy confinement ratio, tau_i / tau_e.  Below 1
     because ITER's baseline is predicted to be ion-temperature-gradient
     dominated (chi_i / chi_e ~ 2); the total loss still follows IPB98(y,2). */
  const CHI_RATIO = 0.45;

  /* Helium ash confinement, tau_He* / tau_E.  Literature for a pumped
     divertor spans 5-10; 6 reproduces ITER's reference n_He/n_e = 4.1%. */
  const TAU_HE = 5.85;

  /* ---------- small helpers --------------------------------------------- */
  const clamp = (x, a, b) => x < a ? a : (x > b ? b : x);
  const lerp  = (a, b, t) => a + (b - a) * t;
  function rnd(seed) {                        // deterministic-ish jitter
    let s = seed;
    return () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  }

  /* Radial grid used for every profile integral. */
  const NR = 65;
  const RHO = new Float64Array(NR);
  for (let i = 0; i < NR; i++) RHO[i] = i / (NR - 1);

  /* Pedestal + core profile shape, normalised to f(0) = 1.  Inside the
     pedestal top it is the usual (1-x^2)^alpha core sitting on a pedestal
     of height `ped`; outside it falls linearly to the separatrix value. */
  function profShape(rho, alpha, ped, rped, fsep) {
    if (rho <= rped) {
      const x = rho / rped;
      return ped + (1 - ped) * Math.pow(Math.max(0, 1 - x * x), alpha);
    }
    return ped + (fsep - ped) * (rho - rped) / (1 - rped);
  }

  /* Volume average of a profile sampled on RHO: <f> = int f 2 rho drho. */
  function volAvg(f) {
    let s = 0;
    for (let i = 0; i < NR - 1; i++) {
      s += 0.5 * (f[i] * 2 * RHO[i] + f[i + 1] * 2 * RHO[i + 1]) *
           (RHO[i + 1] - RHO[i]);
    }
    return s;
  }

  /* Slowing-down stored energy per unit birth power, from the Stix
     steady-state distribution f(E) ~ E^1/2 / (E^3/2 + Ec^3/2):
        W = P tau_s G(x),   x = E0/Ec,
        G(x) = [x - I(x)] / (2x),   I(x) = int_0^x du/(1+u^3/2).
     I(inf) = 4 pi / (3 sqrt(3)) = 2.4184, with a 2/sqrt(x) tail. */
  const I_INF = 4 * Math.PI / (3 * Math.sqrt(3));
  function slowingI(x) {
    if (x <= 0) return 0;
    if (x > 4) return I_INF - 2 / Math.sqrt(x);
    let I = 0; const n = 64, h = x / n;
    for (let i = 0; i < n; i++) {
      const u0 = i * h, u1 = u0 + h;
      I += 0.5 * h * (1 / (1 + Math.pow(u0, 1.5)) + 1 / (1 + Math.pow(u1, 1.5)));
    }
    return I;
  }
  function slowingG(x) { return x <= 0 ? 0 : (x - slowingI(x)) / (2 * x); }
  /* Fraction of a fast-ion population's energy delivered to the background
     IONS rather than to the electrons:
        F_i = (1/E0) int_0^E0 Ec^3/2 / (E^3/2 + Ec^3/2) dE = I(x)/x.
     Above the critical energy a fast ion drags mainly on electrons, so at
     ITER's temperatures the 3.5 MeV alphas heat the electrons ~85:15 —
     which is what makes T_e run above T_i in a burning plasma. */
  function ionFraction(x) { return x <= 0 ? 1 : slowingI(x) / x; }

  /* ======================================================================= */
  class Tokamak {
    constructor(machine) {
      const src = MACHINES[machine] || MACHINES.iter;
      const M = {};
      for (const k in src) M[k] = src[k];
      const g = geometry(M);
      M.eps = g.eps; M.V = g.V; M.S = g.S; M.Lp = g.Lp; M.kappaA = g.kappaA;
      /* the renderer and the HUD still speak of a single elongation */
      M.kappa = M.kappaX; M.delta = M.deltaX;
      this.M = M;
      this.reset();
    }

    reset() {
      const M = this.M;
      /* --- evolving state ------------------------------------------------ */
      this.t      = 0;        // shot time                          [s]
      this.ne     = 0.010;    // volume-avg electron density        [1e20 m^-3]
      this.Te     = 0.10;     // density-weighted electron temp     [keV]
      this.Ti     = 0.08;     // density-weighted ion temperature   [keV]
      this.Ip     = 0.20;     // plasma current                     [MA]
      this.Bt     = M.BtNom;  // toroidal field                     [T]
      this.nHe    = 0.0;      // helium ash density                 [1e20 m^-3]
      this.pVac   = 1.0e-5;   // neutral pressure in the vessel     [Pa]
      this.Wmag   = 0;        // poloidal magnetic energy           [MJ]

      /* --- actuators (user controllable) --------------------------------- */
      this.PnbiSet = 0;       // neutral beam injection             [MW]
      this.PicrSet = 0;       // ion cyclotron RF heating           [MW]
      this.PecrSet = 0;       // electron cyclotron RF heating      [MW]
      this.gasSet  = 0;       // gas puff rate                      [0..1]
      this.IpSet   = 0.2;     // requested current                  [MA]
      this._fRamp  = 0.46;    // burn-phase Greenwald fraction ramp
      this._peakIp = 0;       // highest current reached this shot
      this.autoPilot = true;  // run the scripted discharge sequence

      /* --- realised (ramped) actuator values ----------------------------- */
      this.Pnbi = 0; this.Picr = 0; this.Pecr = 0; this.gas = 0;

      /* --- regime / event state ------------------------------------------ */
      this.modeKey   = 'idle';
      this.mode      = 'ДАЙЫН';
      this.hMode     = false;
      this.H98       = 0.50;
      this.elmPhase  = 0;  this.elmFlash = 0;  this.elmCount = 0;
      this.sawPhase  = 0;  this.sawFlash = 0;  this.sawCount = 0;
      this.disruptT  = -1; this.disrupted = false;
      this.ignition  = 0;      // smoothed 0..1 "burning plasma" indicator
      this.alarms    = [];
      this.shotNo    = 84217 + ((Math.random() * 40) | 0);
      this.rotPhase  = 0;      // toroidal rotation phase for the renderer
      this.mhdPhase  = 0;
      this.rngf      = rnd(this.shotNo * 7919 + 13);

      /* --- derived (filled by step()) ------------------------------------ */
      this.Pfus = 0; this.Palpha = 0; this.Paux = 0; this.Pohm = 0;
      this.Prad = 0; this.Pbrem = 0; this.Pline = 0; this.Psync = 0;
      this.Ploss = 0; this.Pnet = 0; this.Psep = 0;
      this.tauE = 0.01; this.tauE98 = 0.01; this.Q = 0; this.beta = 0;
      this.betaN = 0; this.betaTh = 0; this.betaP = 0;
      this.q95 = 20; this.q0 = 3; this.nG = 1; this.fG = 0;
      this.Zeff = 1.6; this.fDT = 1; this.nBar = 0;
      this.neutronRate = 0; this.neutronFlux = 0;
      this.W = 0; this.Wth = 0; this.Wfast = 0; this.We = 0; this.Wi = 0;
      this.qDiv = 0; this.qDivSteady = 0; this.li = 0.9;
      this.Vloop = 0; this.fluxTor = 0; this.fluxPol = 0; this.Pthresh = 1e9;
      this.fBS = 0; this.Ibs = 0; this.eta = 0;
      this.Te0 = 0; this.Ti0 = 0; this.ne0 = 0; this.tauEq = 0.1; this.z2a = 0.4;
      this.nProf = new Float64Array(NR);
      this.TeProf = new Float64Array(NR);
      this.TiProf = new Float64Array(NR);
      this.pProf = new Float64Array(NR);
      this.sfProf = new Float64Array(NR);   // fusion power density [MW/m^3]
      this.qProf = new Float64Array(NR);    // safety factor profile
      this.coilI = new Float64Array(M.nTF); // TF coil currents      [kA]
      this.pfI   = new Float64Array(M.nPF); // PF coil currents      [kA]
      this._gN   = new Float64Array(NR);
      this._gTe  = new Float64Array(NR);
      this._gTi  = new Float64Array(NR);
      this.mirnov = 0;                      // magnetic pickup coil  [a.u.]
      this.dAlpha = 0;                      // D-alpha photodiode    [a.u.]
      /* fill the profiles and every derived quantity once, so the HUD has a
         consistent state to draw before the first step() lands */
      this.composition();
      this.buildProfiles();
      this.integrateProfiles();
      this.fastIons();
      this.updateDerived(0);
    }

    /* -------------------------------------------------------------------- */
    /*  Profile reconstruction.  A pedestal appears in H-mode and the core
        stiffens; the density stays nearly flat while the temperature is
        strongly peaked, which is what sets T_e0 / <T_e> ~ 2.8 at the ITER
        operating point.                                                    */
    /* -------------------------------------------------------------------- */
    buildProfiles() {
      const H = this.hMode;
      const an   = H ? 0.88 : 0.55;    // density peaking exponent
      const aT   = H ? 4.70 : 1.70;    // temperature peaking exponent
      const pedN = H ? 0.53 : 0.05;    // pedestal-top fraction, density
      const pedT = H ? 0.185 : 0.04;   // pedestal-top fraction, temperature
      const rped = 0.94;
      const sawSup = 1 - 0.22 * this.sawFlash;   // core flattening after a crash

      const gN = this._gN, gTe = this._gTe, gTi = this._gTi;
      for (let i = 0; i < NR; i++) {
        const r = RHO[i];
        const core = r < 0.35 ? sawSup : 1;
        gN[i]  = profShape(r, an, pedN, rped, 0.35 * pedN);
        gTe[i] = profShape(r, aT, pedT, rped, 0.10 * pedT) * core;
        gTi[i] = profShape(r, aT * 0.82, pedT, rped, 0.10 * pedT) * core;
      }
      /* <n> is a volume average; <T> is density-weighted, so the temperature
         normaliser is <g_n g_T> / <g_n>. */
      const nAvg = volAvg(gN);
      const nTe = new Float64Array(NR), nTi = new Float64Array(NR);
      for (let i = 0; i < NR; i++) { nTe[i] = gN[i] * gTe[i]; nTi[i] = gN[i] * gTi[i]; }
      const kTe = volAvg(nTe) / nAvg, kTi = volAvg(nTi) / nAvg;

      for (let i = 0; i < NR; i++) {
        this.nProf[i]  = this.ne * gN[i] / nAvg;
        this.TeProf[i] = this.Te * gTe[i] / kTe;
        this.TiProf[i] = this.Ti * gTi[i] / kTi;
        this.pProf[i]  = 1.602 * this.nProf[i] *
                         (this.TeProf[i] + this.fIon * this.TiProf[i]);  // ~kPa
      }
      this.ne0 = this.nProf[0]; this.Te0 = this.TeProf[0]; this.Ti0 = this.TiProf[0];
      /* line-averaged density along the midplane chord — the density
         IPB98(y,2) and the Martin threshold are written in terms of */
      let s = 0;
      for (let i = 0; i < NR - 1; i++) s += 0.5 * (gN[i] + gN[i + 1]) * (RHO[i + 1] - RHO[i]);
      this.nBar = this.ne * s / nAvg;
    }

    /* Composition: fuel dilution and Z_eff from the ash + impurity mix. */
    composition() {
      const fHe = clamp(this.nHe / Math.max(this.ne, 1e-6), 0, 0.3);
      let sumZ = 2 * fHe, sumZ2 = 4 * fHe, sumIon = fHe;
      let z2a = fHe * 4 / 4;                        // sum (n_j/n_e) Z_j^2 / A_j
      for (const s of IMP) {
        sumZ  += s.Z * s.frac;
        sumZ2 += s.Z * s.Z * s.frac;
        sumIon += s.frac;
        z2a   += s.frac * s.Z * s.Z / s.A;
      }
      const fDT = Math.max(0, 1 - sumZ);            // charge neutrality
      this.fDT  = fDT;
      this.fHe  = fHe;
      this.fIon = fDT + sumIon;                     // n_i / n_e
      this.Zeff = clamp(fDT + sumZ2, 1.0, 6.0);
      /* sum_j (n_j/n_e) Z_j^2 / A_j — the combination that sets both the
         electron-ion equipartition rate and the fast-ion critical energy */
      this.z2a  = z2a + fDT / this.M.Amass;
    }

    /* Radially integrated fusion power and radiated power. */
    integrateProfiles() {
      const M = this.M;
      const dVdr = 2 * M.V;                  // dV = 2*rho*drho * V
      const fDT = this.fDT;
      /* wall reflectivity for the cyclotron harmonics */
      const refl = 0.7;

      let Pfus = 0, Pbrem = 0, Psync = 0, Pline = 0;
      for (let i = 0; i < NR - 1; i++) {
        const w = 0.5 * (RHO[i] + RHO[i + 1]);
        const dr = RHO[i + 1] - RHO[i];
        const n20 = 0.5 * (this.nProf[i] + this.nProf[i + 1]);
        const Ti  = 0.5 * (this.TiProf[i] + this.TiProf[i + 1]);
        const Te  = 0.5 * (this.TeProf[i] + this.TeProf[i + 1]);

        /* D-T fusion: n_D = n_T = n_DT/2, distinct species so no 1/2 */
        const nDT = n20 * 1e20 * fDT;
        const pd  = 0.25 * nDT * nDT * reactivityDT(Ti) * E_FUS;          // W/m^3

        /* bremsstrahlung, with the relativistic correction (Rider 1995) */
        const br = 5.35e-3 * this.Zeff * n20 * n20 * Math.sqrt(Math.max(Te, 0.01))
                   * (1 + 2.0 * Te / 511);                                // MW/m^3

        /* synchrotron: Trubnikov scaling shape (n^1/2 T^5/2 B^5/2 a^-1/2,
           reduced by wall reflection), with the absolute level set to the
           Albajar (2001) ITER result of ~15 MW at the Q=10 point.  A full
           radiation-transport treatment is out of scope for a 0-D model. */
        const sy = 2.60e-6 * Math.sqrt(n20) * Math.pow(Math.max(Te, 0.01), 2.5)
                   * Math.pow(this.Bt, 2.5) * Math.sqrt((1 - refl) / M.a);  // MW/m^3

        /* impurity line radiation.  Be and Ne are fully stripped in the
           core, so only tungsten contributes there. */
        let ln = 0;
        for (const s of IMP) {
          if (s.name !== 'W') continue;
          ln += n20 * 1e20 * (n20 * 1e20 * s.frac) * coolW(Te) / MW;       // MW/m^3
        }

        Pfus  += pd / MW * w * dVdr * dr;
        Pbrem += br * w * dVdr * dr;
        Psync += sy * w * dVdr * dr;
        Pline += ln * w * dVdr * dr;
        this.sfProf[i] = pd / MW;
      }
      this.sfProf[NR - 1] = 0;
      this.Pfus   = Pfus;
      this.Pbrem  = Pbrem;
      this.Psync  = Psync;
      this.Pline  = Pline;
      this.Prad   = Pbrem + Psync + Pline;
      this.Palpha = Pfus * E_ALPHA_F;
    }

    /* Fast-ion stored energy from the alpha, beam and RF minority
       populations (Stix slowing-down distribution).  This is ~7% of the
       plasma energy at the ITER Q=10 point and is what separates the
       thermal beta from the beta the MHD limits actually see. */
    fastIons() {
      const lnL = 17;
      /* Spitzer slowing-down time on electrons, per species (A, Z) */
      const tauS = (A, Z, Te, n20) =>
        0.19826 * A / (Z * Z * lnL) * Math.pow(Math.max(Te, 0.1), 1.5) /
        Math.max(n20, 0.02);
      /* critical energy: below it the fast ion slows mainly on the ions */
      const zoa = this.z2a;
      const Ecrit = (A, Te) => 14.8 * A * Math.max(Te, 0.1) *
                               Math.pow(Math.max(zoa, 0.05), 2 / 3);

      /* Alphas are born where the fusion rate is, deep in the hot core,
         and tau_s goes as T_e^3/2 — so a volume-averaged slowing-down time
         understates their stored energy badly.  Integrate over the profile
         instead, weighting by the local alpha source. */
      let Wa = 0, srcTot = 0, ionA = 0;
      for (let i = 0; i < NR - 1; i++) {
        const w = 0.5 * (RHO[i] + RHO[i + 1]);
        const dr = RHO[i + 1] - RHO[i];
        const src = this.sfProf[i] * E_ALPHA_F * w * 2 * this.M.V * dr;  // MW
        if (src <= 0) continue;
        const Te = 0.5 * (this.TeProf[i] + this.TeProf[i + 1]);
        const n  = 0.5 * (this.nProf[i] + this.nProf[i + 1]);
        const x  = E_ALPHA / Ecrit(4, Te);
        Wa    += src * tauS(4, 2, Te, n) * slowingG(x);
        ionA  += src * ionFraction(x);
        srcTot += src;
      }

      /* Beam and RF minority populations are deposited broadly, so the
         volume-averaged temperature is adequate for them. */
      const Te = Math.max(this.Te, 0.1), n20 = Math.max(this.ne, 0.02);
      const xN = 1000 / Ecrit(2, Te), xI = 500 / Ecrit(1, Te);
      const Wn = this.Pnbi * tauS(2, 1, Te, n20) * slowingG(xN);
      const Wi = this.Picr * tauS(1, 1, Te, n20) * slowingG(xI);

      this.fIonSplit = {
        alpha: srcTot > 0 ? ionA / srcTot : 0.15,
        nbi: ionFraction(xN),
        icr: ionFraction(xI)
      };
      this.Wfast = Wa + Wn + Wi;
    }

    /* IPB98(y,2) ELMy H-mode energy confinement scaling.  Written in terms
       of the line-averaged density and the areal elongation, as fitted. */
    tauScaling(Ploss) {
      const M = this.M;
      const P = Math.max(Ploss, 0.5);
      const n19 = Math.max(this.nBar * 10, 0.1);
      return 0.0562 *
        Math.pow(Math.max(this.Ip, 0.05), 0.93) *
        Math.pow(Math.max(this.Bt, 0.05), 0.15) *
        Math.pow(P, -0.69) *
        Math.pow(n19, 0.41) *
        Math.pow(M.Amass, 0.19) *
        Math.pow(M.R0, 1.97) *
        Math.pow(M.eps, 0.58) *
        Math.pow(M.kappaA, 0.78);
    }

    /* Martin 2008 L-H transition power threshold (line-averaged density). */
    lhThreshold() {
      const M = this.M;
      return 0.0488 * Math.pow(Math.max(this.nBar, 0.02), 0.717) *
             Math.pow(Math.max(this.Bt, 0.1), 0.803) *
             Math.pow(M.S, 0.941) * (2 / M.Amass);
    }

    /* Neoclassical parallel resistivity: Spitzer, corrected for the
       trapped-particle fraction (Wesson §14.10). */
    resistivity() {
      /* the current channel sits well inside the density-weighted average
         radius, so the resistivity is evaluated on the rho = 0.35 surface
         rather than at <T_e> — a factor of ~2.5 in eta at ITER conditions */
      const iRef = Math.round(0.35 * (NR - 1));
      const Te = Math.max(this.TeProf[iRef] || this.Te, 0.05);
      const spitzer = 2.8e-8 * this.Zeff / Math.pow(Te, 1.5);
      /* trapped fraction at the current-carrying mid-radius */
      const e = 0.5 * this.M.eps;
      const ft = clamp(1.46 * Math.sqrt(e) - 0.46 * Math.pow(e, 1.5), 0, 0.85);
      this.eta = spitzer / Math.pow(1 - ft, 2);
      return this.eta;
    }

    /* -------------------------------------------------------------------- */
    /*  Scripted discharge programme.                                        */
    /* -------------------------------------------------------------------- */
    autoSequence(t, dt) {
      const M = this.M;
      const seg = (t0, t1, v0, v1) =>
        t <= t0 ? v0 : t >= t1 ? v1 : lerp(v0, v1, (t - t0) / (t1 - t0));
      /* the reference heating mix: 33 MW of NBI + 17 MW of ICRF = 50 MW */
      const Pnb = 33 * M.IpNom / 15, Pic = 17 * M.IpNom / 15;

      this.IpSet   = seg(1, 26, 0.4, M.IpNom);
      this.PecrSet = (t > 1.5 && t < 8) ? 12 : (t > 8 ? seg(8, 14, 12, 0) : 0);
      this.PnbiSet = t < 14 ? 0 : seg(14, 22, 0, Pnb);
      this.PicrSet = t < 18 ? 0 : seg(18, 24, 0, Pic);

      /* Density programme.  The L-H power threshold rises with density, so
         the ramp is held at a low Greenwald fraction until H-mode is
         established, then pushed up towards the burn-phase target.        */
      let fTarget = seg(2, 20, 0.30, 0.46);
      if (this.hMode) {
        this._fRamp = Math.min(0.857, (this._fRamp || 0.46) + 0.028 * dt);
        fTarget = Math.max(fTarget, this._fRamp);
      }
      if (t > 120) {                       // controlled ramp-down and re-arm
        this.IpSet   = seg(120, 145, M.IpNom, 0.3);
        this.PnbiSet = seg(120, 132, Pnb, 0);
        this.PicrSet = seg(120, 130, Pic, 0);
        this.PecrSet = 0;
        fTarget      = seg(120, 142, 0.857, 0.10);
      }

      /* feed-forward + proportional gas valve control */
      const nTarget = fTarget * Math.max(this.nG, 0.005);
      /* steady state is n = tau_P (S_gas + S_nbi), so the feed-forward term
         has to subtract the fuelling the beams already provide */
      const tauP = 2.0 * Math.max(this.tauE, 0.05);
      const Snbi = this.Pnbi * 0.0016;
      const ff = (nTarget / tauP - Snbi) / (0.30 * this.M.V / 840);
      this.gasSet = clamp(ff + 6.0 * (nTarget - this.ne), 0, 1);

      if (t > 152) { this.reset(); this.autoPilot = true; }
    }

    /* -------------------------------------------------------------------- */
    /*  Main integration step.  dt is the wall-clock frame time scaled by the
        simulation rate; it is sub-stepped for numerical stability.          */
    /* -------------------------------------------------------------------- */
    step(dt) {
      const N = 4;
      const h = clamp(dt, 0, 0.12) / N;
      for (let s = 0; s < N; s++) this.substep(h);
      this.updateDerived(dt);
    }

    substep(dt) {
      if (dt <= 0) return;
      const M = this.M;
      this.t += dt;
      if (this.autoPilot) this.autoSequence(this.t, dt);

      /* --- actuator slew limits ------------------------------------------ */
      const slew = (cur, set, rate) => cur + clamp(set - cur, -rate * dt, rate * dt);
      this.Pnbi = slew(this.Pnbi, this.PnbiSet, 40);
      this.Picr = slew(this.Picr, this.PicrSet, 40);
      this.Pecr = slew(this.Pecr, this.PecrSet, 60);
      this.gas  = slew(this.gas,  this.gasSet,  1.2);

      /* --- current ramp; the poloidal field system is rate limited -------- */
      const dIp = clamp(this.IpSet - this.Ip, -0.9 * dt, 0.55 * dt);
      this.Ip = Math.max(0, this.Ip + dIp);
      this.li = clamp(0.65 + 0.45 * Math.exp(-this.t / 30), 0.6, 1.25);
      this.Wmag = 0.5 * (M.R0 * MU0 * (Math.log(8 / M.eps) + this.li / 2 - 2)) *
                  Math.pow(this.Ip * 1e6, 2) / MW;

      /* --- composition, profiles and power sources ------------------------ */
      this.composition();
      this.buildProfiles();
      this.integrateProfiles();
      this.fastIons();

      const Rp = this.resistivity() * 2 * Math.PI * M.R0 /
                 (Math.PI * M.a * M.a * M.kappaA);          // plasma resistance
      /* only the inductive fraction of the current dissipates ohmically */
      const Iind = Math.max(this.Ip * (1 - this.fBS), 0);
      this.Pohm  = Math.min(Rp * Math.pow(Iind * 1e6, 2) / MW, 60);
      this.Vloop = Rp * Iind * 1e6;

      this.Paux = this.Pnbi + this.Picr + this.Pecr;
      const Pheat = this.Paux + this.Pohm + this.Palpha;
      this.Ploss = Math.max(Pheat - this.Prad, 0.5);
      this.Psep  = this.Ploss;

      /* --- L-H transition with hysteresis -------------------------------- */
      this.Pthresh = this.lhThreshold();
      if (!this.hMode && Pheat > this.Pthresh * 1.05 &&
          this.Ip > 0.4 * M.IpNom && this.t > 6) {
        this.hMode = true; this.elmPhase = 0;
        this.pushAlarm('L-H АУЫСУЫ', 'ok');
      } else if (this.hMode && Pheat < this.Pthresh * 0.75) {
        this.hMode = false;
        this.pushAlarm('H-L КЕРІ АУЫСУЫ', 'warn');
      }
      const targetH = this.hMode ? 1.00 : 0.52;
      this.H98 += (targetH - this.H98) * clamp(dt / 0.35, 0, 1);

      /* --- confinement --------------------------------------------------- */
      this.tauE98 = this.tauScaling(this.Ploss);
      let tau = this.tauE98 * this.H98;
      /* IPB98(y,2) is an ELMy H-mode scaling, so the time-averaged ELM and
         sawtooth cost is already inside it.  The discrete crashes below are
         therefore kept at the mitigated ITER amplitude (<1% of W per ELM)
         and no further transient penalty is applied here — subtracting one
         would count the same energy twice and leave the burn 20% cold. */
      if (this.disrupted) tau *= 0.02;
      this.tauE = Math.max(tau, 1e-3);

      /* --- energy balance (separate electron / ion channels) -------------- */
      const cE = W_UNIT * M.V * this.ne;               // MJ per keV of Te
      const cI = W_UNIT * M.V * Math.max(this.ne * this.fIon, 1e-4);

      /* Electron-ion equipartition (NRL Formulary nu_eps^{e|i}).  tau_eq
         goes as T_e^3/2, so in the hot core it is several times longer than
         at <T_e> — integrating over the profile instead of evaluating at
         the average is what lets T_e sit above T_i in a burning plasma. */
      let Pei = 0, tauEqSum = 0, wSum = 0;
      for (let i = 0; i < NR - 1; i++) {
        const w = 0.5 * (RHO[i] + RHO[i + 1]);
        const dr = RHO[i + 1] - RHO[i];
        const nR  = 0.5 * (this.nProf[i] + this.nProf[i + 1]);
        const TeR = 0.5 * (this.TeProf[i] + this.TeProf[i + 1]);
        const TiR = 0.5 * (this.TiProf[i] + this.TiProf[i + 1]);
        const teq = Math.max(0.02, 5.813e-3 * Math.pow(Math.max(TeR, 0.05), 1.5) /
                    (Math.max(nR, 0.02) * Math.max(this.z2a, 0.05)));
        Pei += W_UNIT * M.V * nR * (TiR - TeR) / teq * w * 2 * dr;
        tauEqSum += teq * w * 2 * dr; wSum += w * 2 * dr;
      }
      this.tauEq = tauEqSum / wSum;                 // ion -> electron transfer

      /* electron / ion split of each fast population, from the same
         slowing-down distribution that sets the fast-ion stored energy */
      const fa = this.fIonSplit || { alpha: 0.15, nbi: 0.25, icr: 0.25 };
      const Pa_i  = this.Palpha * fa.alpha, Pa_e  = this.Palpha - Pa_i;
      const Pnb_i = this.Pnbi   * fa.nbi,   Pnb_e = this.Pnbi   - Pnb_i;
      const Pic_i = this.Picr   * fa.icr,   Pic_e = this.Picr   - Pic_i;

      const We = cE * this.Te, Wi = cI * this.Ti;
      /* IPB98(y,2) constrains the TOTAL loss, not the split between the
         channels.  ITER's baseline is ITG-dominated, so the ion heat
         diffusivity runs above the electron one; CHI_RATIO = tau_i / tau_e
         distributes the same total loss accordingly and is what keeps T_i
         below T_e once the alphas start heating the electrons. */
      const tauEl = this.tauE * (We + Wi / CHI_RATIO) / Math.max(We + Wi, 1e-9);
      const tauIon = CHI_RATIO * tauEl;
      const dWe = Pa_e + Pnb_e + Pic_e + this.Pecr + this.Pohm + Pei
                  - this.Prad - We / tauEl;
      const dWi = Pa_i + Pnb_i + Pic_i - Pei - Wi / tauIon;

      this.Te = Math.max(0.02, this.Te + dWe / cE * dt);
      this.Ti = Math.max(0.02, this.Ti + dWi / cI * dt);
      this.We = cE * this.Te; this.Wi = cI * this.Ti;
      this.Wth = this.We + this.Wi;
      this.W   = this.Wth;              // HUD trace: thermal stored energy

      /* --- particle balance ---------------------------------------------- */
      const tauP = 2.0 * this.tauE;
      const Sgas = this.gas * 0.30 * (M.V / 840);         // fuelling [1e20/s]
      const Snbi = this.Pnbi * 0.0016;
      this.ne = Math.max(0.005, this.ne + (Sgas + Snbi - this.ne / tauP) * dt);

      /* helium ash from the fusion rate, pumped with tau_He* = 6 tau_E */
      const Sfus = this.Pfus * MW / E_FUS / M.V / 1e20;   // [1e20 m^-3 s^-1]
      this.nHe = Math.max(0, this.nHe + (Sfus - this.nHe / (TAU_HE * this.tauE)) * dt);

      /* neutral pressure: puff in, torus cryopumps out */
      const pTarget = 1.2e-5 + this.gas * 4.5e-4 + (1 - clamp(this.ne * 2, 0, 1)) * 1e-4;
      this.pVac += (pTarget - this.pVac) * clamp(dt / 0.4, 0, 1);

      /* --- safety factor, beta, limits ------------------------------------ */
      /* Uckan/Sheffield q95, using the 95% surface shape and the finite
         aspect-ratio correction.  Reproduces q95 = 3.0 for ITER at 15 MA. */
      const k95 = M.kappa95, d95 = M.delta95;
      const shape = (1 + k95 * k95 * (1 + 2 * d95 * d95 - 1.2 * Math.pow(d95, 3))) / 2;
      const fEps  = (1.17 - 0.65 * M.eps) / Math.pow(1 - M.eps * M.eps, 2);
      this.q95 = this.Ip > 0.05
        ? 5 * M.a * M.a * this.Bt / (M.R0 * this.Ip) * shape * fEps : 99;
      this.q0  = clamp(this.q95 / (2.6 + 1.4 * this.li), 0.55, 30);
      this.nG  = this.Ip / (Math.PI * M.a * M.a);         // Greenwald  [1e20]
      this.fG  = this.ne / Math.max(this.nG, 1e-6);

      /* beta[%] = 2 mu0 <p> / B^2.  The MHD limits see the total pressure,
         so the fast-ion contribution belongs in it. */
      const pTh = W_UNIT * (this.ne * this.Te + this.ne * this.fIon * this.Ti); // MJ/m^3
      const pTot = pTh + (2 / 3) * this.Wfast / M.V;
      this.betaTh = 100 * 2 * MU0 * (pTh * MW / 1.5) / (this.Bt * this.Bt);
      this.beta   = 100 * 2 * MU0 * (pTot * MW / 1.5) / (this.Bt * this.Bt);
      this.betaN  = this.Ip > 0.05 ? this.beta * M.a * this.Bt / this.Ip : 0;
      /* poloidal beta and the bootstrap fraction it drives */
      const Bp = MU0 * this.Ip * 1e6 / M.Lp;
      this.betaP = 2 * MU0 * (pTot * MW / 1.5) / (Bp * Bp);
      this.fBS = clamp(0.40 * Math.sqrt(M.eps) * this.betaP, 0, 0.9);
      this.Ibs = this.fBS * this.Ip;

      /* --- MHD activity: sawteeth and ELMs -------------------------------- */
      this.mhdPhase += dt * (8 + 30 * this.beta);
      if (this.q0 < 1.0 && this.Te > 1.5) {
        this.sawPhase += dt / (0.35 + 1.9 * clamp(this.tauE, 0, 1.6));
        if (this.sawPhase >= 1) {
          this.sawPhase = 0; this.sawFlash = 1; this.sawCount++;
          /* the crash mainly flattens the core (sawSup in buildProfiles);
             the net energy it expels is a couple of percent, not a tenth */
          this.Te *= 0.985; this.Ti *= 0.988;
        }
      }
      this.sawFlash = Math.max(0, this.sawFlash - dt * 5);

      if (this.hMode && !this.disrupted) {
        /* type-I ELM frequency scales with the power crossing the separatrix */
        const fELM = clamp(0.35 + 0.020 * this.Ploss, 0.35, 2.6);   // Hz
        this.elmPhase += dt * fELM;
        if (this.elmPhase >= 1) {
          this.elmPhase = 0; this.elmFlash = 1; this.elmCount++;
          /* 0.4-0.9% of W per ELM — the mitigated regime the ELM control
             coils are designed to hold, and the basis of the Q=10 point.
             Most of the type-I ELM cost is already inside IPB98(y,2).     */
          const frac = 0.004 + 0.005 * this.rngf();
          this.Te *= (1 - frac); this.Ti *= (1 - frac);
          this.ne *= (1 - frac * 0.7);
        }
      }
      this.elmFlash = Math.max(0, this.elmFlash - dt * 9);

      /* --- disruption logic ---------------------------------------------- */
      /* Only police the operational limits once there is a real current
         channel — during breakdown the plasma is far from any of them. */
      if (!this.disrupted && this.t > 5 && this.Ip > 0.13 * M.IpNom) {
        let cause = null;
        if (this.fG > 1.25) cause = 'ТЫҒЫЗДЫҚ ШЕГІ — ГРИНВАЛЬД АСЫП КЕТТІ';
        else if (this.q95 < 2.0) cause = 'q95 ТӨМЕН — КИНК ТҰРАҚСЫЗДЫҒЫ';
        else if (this.betaN > 4.2) cause = 'БЕТА ШЕГІ — ИДЕАЛ МГД';
        else if (this.Prad > (this.Paux + this.Pohm + this.Palpha) * 1.35 && this.Te > 1)
          cause = 'СӘУЛЕЛЕНУ КОЛЛАПСЫ';
        if (cause) this.triggerDisruption(cause);
      }
      if (this.disrupted) {
        /* thermal quench then current quench — stay down until operator reset */
        const dtq = this.t - this.disruptT;
        if (dtq < 0.003) { this.Te *= 0.2; this.Ti *= 0.2; }
        this.Ip = Math.max(0, this.Ip - dt * 45 * M.IpNom / 15);
        this.Te = Math.max(0.02, this.Te - dt * 8);
        this.Ti = Math.max(0.02, this.Ti - dt * 8);
        this.PnbiSet = 0; this.PicrSet = 0; this.PecrSet = 0; this.gasSet = 0;
      }

      /* --- rotation phase used by the renderer ---------------------------- */
      const vTor = 0.35 + 0.9 * clamp(this.Ti / 20, 0, 1.4) + 0.02 * this.Pnbi;
      this.rotPhase += dt * vTor;
    }

    /* -------------------------------------------------------------------- */
    updateDerived(dt) {
      const M = this.M;
      this.Q = this.Paux > 0.3 ? this.Pfus / this.Paux : (this.Pfus > 1 ? 999 : 0);
      this.Pnet = this.Pfus * 0.33 - (this.Paux / 0.4);   // electrical balance
      /* one 14.07 MeV neutron per D-T reaction */
      this.neutronRate = this.Pfus * MW / E_FUS;
      this.neutronFlux = this.neutronRate / M.S;
      /* ~65% of the exhaust is radiated in the divertor before it reaches
         the targets; the rest is spread over the wetted area */
      this.qDivSteady = this.Psep * (1 - 0.65) / M.Sdiv;
      this.qDiv = this.qDivSteady * (1 + 5.0 * this.elmFlash);

      this.fluxTor = Math.PI * M.a * M.a * this.Bt * M.kappaA;     // [Wb]
      this.fluxPol = MU0 * this.Ip * 1e6 * M.R0 *
                     (Math.log(8 / M.eps) + this.li / 2 - 2);      // [Wb]

      /* q-profile for the cross-section widget */
      for (let i = 0; i < NR; i++) {
        const r = Math.max(RHO[i], 1e-3);
        this.qProf[i] = this.q0 + (this.q95 - this.q0) * Math.pow(r, 2.0);
      }

      /* coil currents: TF set by Bt, PF by the shaping/vertical control */
      const Itf = this.Bt / M.BtNom * 68.0;                        // [kA]
      for (let i = 0; i < M.nTF; i++) {
        this.coilI[i] = Itf * (1 + 0.0016 * Math.sin(this.t * 3.1 + i * 1.7))
                            * (1 - 0.02 * this.elmFlash);
      }
      const pfBase = [-0.42, 0.31, 0.18, -0.24, -0.55, 0.47];
      for (let i = 0; i < M.nPF; i++) {
        this.pfI[i] = pfBase[i % pfBase.length] * (12 + this.Ip * 2.6) *
                      (1 + 0.05 * Math.sin(this.t * 1.7 + i) + 0.25 * this.elmFlash);
      }

      /* diagnostics: Mirnov coil and D-alpha photodiode */
      const mhd = this.hMode ? 0.25 : 0.8;
      this.mirnov = (Math.sin(this.mhdPhase) * 0.4 + Math.sin(this.mhdPhase * 2.7) * 0.2) *
                    mhd * clamp(this.beta * 0.6, 0, 2) +
                    (this.rngf() - 0.5) * 0.15 + this.elmFlash * 3.0 +
                    this.sawFlash * 1.6 + (this.disrupted ? (this.rngf() - 0.5) * 8 : 0);
      this.dAlpha = 0.15 + this.gas * 0.5 + this.elmFlash * 4.2 +
                    (this.rngf() - 0.5) * 0.06 + (1 - clamp(this.Te, 0, 1)) * 0.4;

      /* burning-plasma indicator, smoothed for the renderer */
      const ig = clamp((this.Palpha / Math.max(this.Ploss, 1) - 0.12) / 0.5, 0, 1);
      this.ignition += (ig - this.ignition) * clamp((dt || 0.016) * 2.5, 0, 1);

      /* operating mode label */
      /* modeKey stays stable for styling / logic; mode is the display text */
      this._peakIp = Math.max(this._peakIp || 0, this.Ip);
      const frac = this.Ip / M.IpNom;
      if (this.disrupted)                       this.modeKey = 'disrupt';
      else if (frac < 0.033)                    this.modeKey = 'breakdown';
      else if (this.Ip < this._peakIp - 0.033 * M.IpNom) this.modeKey = 'rampdown';
      else if (frac < 0.93)                     this.modeKey = 'rampup';
      else if (this.Q > 4)                      this.modeKey = 'burn';
      else if (this.hMode)                      this.modeKey = 'hmode';
      else                                      this.modeKey = 'lmode';
      this.mode = Tokamak.MODE_LABEL[this.modeKey];
    }

    /* Cheap state capture, so a known-good phase of the discharge can be
       restored instantly instead of re-integrating tens of seconds.  Only
       scalars are stored: the profiles and every derived quantity are
       rebuilt from them on restore. */
    snapshot() {
      const s = {};
      for (const k in this) {
        const v = this[k];
        const t = typeof v;
        if (t === 'number' || t === 'boolean' || t === 'string') s[k] = v;
      }
      s.alarms = this.alarms.map(a => ({ text: a.text, level: a.level, t: a.t }));
      return s;
    }

    restore(s) {
      if (!s) return;
      for (const k in s) if (k !== 'alarms') this[k] = s[k];
      this.alarms = s.alarms.map(a => ({ text: a.text, level: a.level, t: a.t }));
      this.composition();
      this.buildProfiles();
      this.integrateProfiles();
      this.fastIons();
      this.updateDerived(0);
    }

    /* -------------------------------------------------------------------- */
    pushAlarm(text, level) {
      this.alarms.unshift({ text, level, t: this.t });
      if (this.alarms.length > 7) this.alarms.pop();
    }

    triggerDisruption(cause) {
      if (this.disrupted) return;
      this.disrupted = true;
      this.disruptT = this.t;
      this.hMode = false;
      this.pushAlarm('ДИЗРУПЦИЯ: ' + cause, 'crit');
    }

    /* user actions ------------------------------------------------------- */
    injectPellet() {
      this.ne = Math.min(this.ne + 0.22, 2.5);
      this.Te *= 0.93; this.Ti *= 0.94;
      this.pushAlarm('ПЕЛЛЕТ ЕНГІЗІЛДІ — 5.0 мм D-T', 'ok');
    }
    forceELM() {
      this.elmPhase = 1.01;
      this.pushAlarm('ELM БАСҚАРУ ИМПУЛЬСІ', 'ok');
    }
    mitigate() {
      if (!this.disrupted) { this.pushAlarm('DMV ДАЙЫН — ДИЗРУПЦИЯ ЖОҚ', 'warn'); return; }
      this.Prad += 200;
      this.pushAlarm('МАССАЛЫҚ ГАЗ ЕНГІЗУ — БӘСЕҢДЕТІЛДІ', 'ok');
    }
  }

  /* Display labels for each operating regime (Kazakh). */
  Tokamak.MODE_LABEL = {
    idle:      'ДАЙЫН',
    breakdown: 'ІСКЕ ҚОСУ',
    rampup:    'ТОК ӨСУІ',
    lmode:     'L-РЕЖИМ',
    hmode:     'H-РЕЖИМ',
    burn:      'ЖАНУ',
    rampdown:  'ТОК ТӨМЕНДЕУІ',
    disrupt:   'ДИЗРУПЦИЯ'
  };

  Tokamak.NR = NR;
  Tokamak.RHO = RHO;
  Tokamak.reactivityDT = reactivityDT;
  Tokamak.MACHINES = MACHINES;
  Tokamak.geometry = geometry;
  global.Tokamak = Tokamak;
  if (typeof module !== 'undefined' && module.exports) module.exports = Tokamak;
})(typeof window !== 'undefined' ? window : globalThis);
