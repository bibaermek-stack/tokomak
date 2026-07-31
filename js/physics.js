/* ==========================================================================
   physics.js — 0-D tokamak transport / burn model  (ITER-class machine)
   --------------------------------------------------------------------------
   Volume-averaged energy + particle balance with reconstructed radial
   profiles.  Everything the HUD displays is derived from this state, so the
   graphs and the plasma render stay physically consistent with each other.

   References for the parametrisations used below:
     Bosch & Hale, Nucl. Fusion 32 (1992) 611     — D-T reactivity
     ITER Physics Basis, Nucl. Fusion 39 (1999)   — IPB98(y,2) confinement
     Martin et al., J. Phys. Conf. Ser. 123 (2008)— L-H power threshold
     Greenwald, Plasma Phys. Control. Fusion 44   — density limit
   ========================================================================== */
(function (global) {
  'use strict';

  /* ---------- physical constants ---------------------------------------- */
  const KEV_J     = 1.602176634e-16;   // 1 keV in joules
  const E_FUS     = 2.8183e-12;        // 17.59 MeV per D-T reaction  [J]
  const E_ALPHA_F = 0.2013;            // fraction carried by the 3.5 MeV alpha
  const MW        = 1e6;

  /* ---------- machine description (ITER baseline) ------------------------ */
  const M = {
    R0:      6.20,    // major radius                              [m]
    a:       2.00,    // minor radius                              [m]
    kappa:   1.85,    // elongation
    delta:   0.45,    // triangularity
    BtNom:   5.30,    // nominal toroidal field on axis            [T]
    IpNom:  15.00,    // flat-top plasma current                   [MA]
    Amass:   2.50,    // average ion mass (50/50 D-T)              [amu]
    nTF:    18,       // toroidal field coils
    nPF:     6,       // poloidal field coils
    nCS:     6        // central solenoid modules
  };
  M.eps  = M.a / M.R0;
  M.V    = 2 * Math.PI * Math.PI * M.R0 * M.a * M.a * M.kappa;   // ~831 m^3
  M.S    = 4 * Math.PI * Math.PI * M.R0 * M.a *
           Math.sqrt((1 + M.kappa * M.kappa) / 2);               // ~700 m^2
  M.Sdiv = 3.4;       // effective wetted divertor area            [m^2]

  /* ---------- Bosch-Hale D-T fusion reactivity <sigma v> ----------------- */
  const BH = {
    Bg: 34.3827, mrc2: 1124656,
    C1: 1.17302e-9,  C2: 1.51361e-2, C3: 7.51886e-2, C4: 4.60643e-3,
    C5: 1.35000e-2,  C6: -1.06750e-4, C7: 1.36600e-5
  };
  function reactivityDT(T) {                 // T [keV] -> <sigma v> [m^3/s]
    if (T < 0.2) return 0;
    if (T > 100) T = 100;
    const t2 = T * T, t3 = t2 * T;
    const theta = T / (1 - (T * (BH.C2 + T * (BH.C4 + T * BH.C6))) /
                           (1 + T * (BH.C3 + T * (BH.C5 + T * BH.C7))));
    const xi = Math.pow(BH.Bg * BH.Bg / (4 * theta), 1 / 3);
    const sv = BH.C1 * theta * Math.sqrt(xi / (BH.mrc2 * t3)) * Math.exp(-3 * xi);
    return sv * 1e-6;                        // cm^3/s -> m^3/s
  }

  /* ---------- small helpers --------------------------------------------- */
  const clamp = (x, a, b) => x < a ? a : (x > b ? b : x);
  const lerp  = (a, b, t) => a + (b - a) * t;
  function rnd(seed) {                        // deterministic-ish jitter
    let s = seed;
    return () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
  }

  /* Radial grid used for every profile integral. */
  const NR = 33;
  const RHO = new Float64Array(NR);
  for (let i = 0; i < NR; i++) RHO[i] = i / (NR - 1);

  /* ======================================================================= */
  class Tokamak {
    constructor() {
      this.M = M;
      this.reset();
    }

    reset() {
      /* --- evolving state ------------------------------------------------ */
      this.t      = 0;        // shot time                          [s]
      this.ne     = 0.010;    // volume-avg electron density        [1e20 m^-3]
      this.Te     = 0.10;     // volume-avg electron temperature    [keV]
      this.Ti     = 0.08;     // volume-avg ion temperature         [keV]
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
      this.betaN = 0; this.q95 = 20; this.q0 = 3; this.nG = 1; this.fG = 0;
      this.Zeff = 1.6; this.neutronRate = 0; this.neutronFlux = 0;
      this.W = 0; this.We = 0; this.Wi = 0; this.qDiv = 0; this.li = 0.9;
      this.Vloop = 0; this.fluxTor = 0; this.fluxPol = 0; this.Pthresh = 1e9;
      this.nProf = new Float64Array(NR);
      this.TeProf = new Float64Array(NR);
      this.TiProf = new Float64Array(NR);
      this.pProf = new Float64Array(NR);
      this.sfProf = new Float64Array(NR);   // fusion power density [MW/m^3]
      this.qProf = new Float64Array(NR);    // safety factor profile
      this.coilI = new Float64Array(M.nTF); // TF coil currents      [kA]
      this.pfI   = new Float64Array(M.nPF); // PF coil currents      [kA]
      this.mirnov = 0;                      // magnetic pickup coil  [a.u.]
      this.dAlpha = 0;                      // D-alpha photodiode    [a.u.]
      this.updateDerived(0);
    }

    /* -------------------------------------------------------------------- */
    /*  Profile reconstruction.  Peaking factors stiffen in H-mode and a
        pedestal appears at the last ~5% of the minor radius.                */
    /* -------------------------------------------------------------------- */
    buildProfiles() {
      const an = this.hMode ? 0.35 : 0.55;   // density peaking exponent
      const aT = this.hMode ? 1.35 : 1.85;   // temperature peaking exponent
      const ped = this.hMode ? 0.32 : 0.03;  // pedestal height fraction
      const rped = 0.94;

      /* Volume average of (1-rho^2)^alpha over dV = 2 rho drho is 1/(alpha+1) */
      const shape = (rho, alpha) => Math.pow(Math.max(0, 1 - rho * rho), alpha);
      let normN = 0, normTe = 0, normTi = 0;
      const fN = new Float64Array(NR), fTe = new Float64Array(NR), fTi = new Float64Array(NR);

      for (let i = 0; i < NR; i++) {
        const r = RHO[i];
        const pedN  = ped * 0.9 * (r > rped ? (1 - r) / (1 - rped) : 1);
        const pedT  = ped * (r > rped ? (1 - r) / (1 - rped) : 1);
        fN[i]  = (1 - ped * 0.9) * shape(r, an) + pedN;
        fTe[i] = (1 - ped)       * shape(r, aT) + pedT;
        fTi[i] = (1 - ped)       * shape(r, aT * 0.95) + pedT;
      }
      /* volume-weighted normalisation: <f> = int f 2 rho drho */
      for (let i = 0; i < NR - 1; i++) {
        const r0 = RHO[i], r1 = RHO[i + 1], dr = r1 - r0;
        normN  += 0.5 * (fN[i] * 2 * r0 + fN[i + 1] * 2 * r1) * dr;
        normTe += 0.5 * (fTe[i] * 2 * r0 + fTe[i + 1] * 2 * r1) * dr;
        normTi += 0.5 * (fTi[i] * 2 * r0 + fTi[i + 1] * 2 * r1) * dr;
      }
      const sawSup = 1 - 0.22 * this.sawFlash;   // core flattening after a crash
      for (let i = 0; i < NR; i++) {
        const core = RHO[i] < 0.35 ? sawSup : 1;
        this.nProf[i]  = this.ne * fN[i] / normN;
        this.TeProf[i] = this.Te * fTe[i] / normTe * core;
        this.TiProf[i] = this.Ti * fTi[i] / normTi * core;
        this.pProf[i]  = 1.602 * (this.nProf[i] * this.TeProf[i] +
                                  this.nProf[i] * this.TiProf[i]); // ~kPa
      }
    }

    /* Radially integrated fusion power and bremsstrahlung. */
    integrateProfiles() {
      const dVdr = 2 * M.V;                  // dV = 2*rho*drho * V
      const fHe  = clamp(this.nHe / Math.max(this.ne, 1e-6), 0, 0.3);
      const fBe  = 0.02;                     // beryllium/tungsten impurity frac
      /* charge neutrality: fDT + 2 fHe + 4 fBe = 1  (fractions of n_e) */
      const fDT  = Math.max(0, 1 - 2 * fHe - 4 * fBe);
      /* Zeff = sum(n_i Z_i^2)/n_e */
      this.Zeff  = clamp(fDT + 4 * fHe + 16 * fBe, 1.0, 4.5);

      let Pfus = 0, Pbrem = 0, Psync = 0;
      for (let i = 0; i < NR - 1; i++) {
        const w = 0.5 * (RHO[i] + RHO[i + 1]);
        const dr = RHO[i + 1] - RHO[i];
        const n20 = 0.5 * (this.nProf[i] + this.nProf[i + 1]);
        const Ti  = 0.5 * (this.TiProf[i] + this.TiProf[i + 1]);
        const Te  = 0.5 * (this.TeProf[i] + this.TeProf[i + 1]);
        const nDT = n20 * 1e20 * fDT;
        const nD = nDT * 0.5, nT = nDT * 0.5;
        const sv = reactivityDT(Ti);
        const pd = nD * nT * sv * E_FUS;                    // W/m^3
        const br = 5.35e-3 * this.Zeff * n20 * n20 * Math.sqrt(Math.max(Te, 0.01)); // MW/m^3
        /* Trubnikov-like synchrotron scaling, normalised to ~10 MW at
           n=1e20, Te=10 keV, B=5.3 T after wall reflection losses          */
        const sy = 1.35e-6 * Math.sqrt(n20) * Math.pow(Math.max(Te, 0.01), 2.5)
                   * this.Bt * this.Bt;                                            // MW/m^3
        Pfus  += pd / MW * w * dVdr * dr;
        Pbrem += br * w * dVdr * dr;
        Psync += sy * w * dVdr * dr;
        this.sfProf[i] = pd / MW;
      }
      this.sfProf[NR - 1] = 0;
      this.Pfus  = Pfus;
      this.Pbrem = Pbrem;
      this.Psync = Psync;
      /* Edge/divertor line radiation: strong radiator fraction, grows with
         impurity seeding and with the He ash content. */
      this.Pline = (0.25 + 1.6 * fHe + 8 * fBe) * (this.Pbrem + 4) *
                   (this.hMode ? 1.15 : 0.85);
      this.Prad  = this.Pbrem + this.Psync + this.Pline;
      this.Palpha = this.Pfus * E_ALPHA_F;
      this.fDT = fDT;
    }

    /* IPB98(y,2) ELMy H-mode energy confinement scaling. */
    tauScaling(Ploss) {
      const P = Math.max(Ploss, 0.5);
      const n19 = Math.max(this.ne * 10, 0.1);
      const kA = M.V / (2 * Math.PI * Math.PI * M.R0 * M.a * M.a);
      return 0.0562 *
        Math.pow(Math.max(this.Ip, 0.05), 0.93) *
        Math.pow(Math.max(this.Bt, 0.05), 0.15) *
        Math.pow(P, -0.69) *
        Math.pow(n19, 0.41) *
        Math.pow(M.Amass, 0.19) *
        Math.pow(M.R0, 1.97) *
        Math.pow(M.eps, 0.58) *
        Math.pow(kA, 0.78);
    }

    /* Martin 2008 L-H transition power threshold. */
    /* Martin 2008: n is the line-averaged density in 1e20 m^-3 */
    lhThreshold() {
      return 0.0488 * Math.pow(Math.max(this.ne, 0.02), 0.717) *
             Math.pow(Math.max(this.Bt, 0.1), 0.803) *
             Math.pow(M.S, 0.941) * (2 / M.Amass);
    }

    /* -------------------------------------------------------------------- */
    /*  Scripted discharge programme.                                        */
    /* -------------------------------------------------------------------- */
    autoSequence(t, dt) {
      const seg = (t0, t1, v0, v1) =>
        t <= t0 ? v0 : t >= t1 ? v1 : lerp(v0, v1, (t - t0) / (t1 - t0));

      this.IpSet   = seg(1, 26, 0.4, M.IpNom);
      this.PecrSet = (t > 1.5 && t < 8) ? 12 : (t > 8 ? seg(8, 12, 12, 6) : 0);
      this.PnbiSet = t < 14 ? 0 : seg(14, 22, 0, 33);
      this.PicrSet = t < 18 ? 0 : seg(18, 24, 0, 20);

      /* Density programme.  The L-H power threshold rises with density, so
         the ramp is held at a low Greenwald fraction until H-mode is
         established, then pushed up towards the burn-phase target.        */
      let fTarget = seg(2, 20, 0.30, 0.46);
      if (this.hMode) {
        this._fRamp = Math.min(0.85, (this._fRamp || 0.46) + 0.028 * dt);
        fTarget = Math.max(fTarget, this._fRamp);
      }
      if (t > 120) {                       // controlled ramp-down and re-arm
        this.IpSet   = seg(120, 145, M.IpNom, 0.3);
        this.PnbiSet = seg(120, 132, 33, 0);
        this.PicrSet = seg(120, 130, 20, 0);
        this.PecrSet = 0;
        fTarget      = seg(120, 142, 0.85, 0.10);
      }

      /* feed-forward + proportional gas valve control */
      const nTarget = fTarget * Math.max(this.nG, 0.005);
      const ff = nTarget / (0.30 * 2.0 * Math.max(this.tauE, 0.05));
      this.gasSet = clamp(ff + 3.0 * (nTarget - this.ne), 0, 1);

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
      this.Wmag = 0.5 * (M.R0 * 1.256e-6 * (Math.log(8 / M.eps) + this.li / 2 - 2)) *
                  Math.pow(this.Ip * 1e6, 2) / MW;

      /* --- profiles and power sources ------------------------------------ */
      this.buildProfiles();
      this.integrateProfiles();

      const Spitzer = 2.8e-8 * this.Zeff / Math.pow(Math.max(this.Te, 0.05), 1.5);
      const Rp = Spitzer * 2 * Math.PI * M.R0 /
                 (Math.PI * M.a * M.a * M.kappa);           // plasma resistance
      this.Pohm = Math.min(Rp * Math.pow(this.Ip * 1e6, 2) / MW, 60);
      this.Vloop = Rp * this.Ip * 1e6;

      this.Paux = this.Pnbi + this.Picr + this.Pecr;
      const Pheat = this.Paux + this.Pohm + this.Palpha;
      this.Ploss = Math.max(Pheat - this.Prad, 0.5);
      this.Psep  = this.Ploss;

      /* --- L-H transition with hysteresis -------------------------------- */
      this.Pthresh = this.lhThreshold();
      if (!this.hMode && Pheat > this.Pthresh * 1.05 && this.Ip > 6 && this.t > 6) {
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
      /* IPB98(y,2) is already an ELMy H-mode scaling, so the average ELM
         cost is baked in.  Only a small transient dip is applied here so
         the trace shows the event; the discrete crash below does the rest. */
      tau *= (1 - 0.12 * this.elmFlash);
      if (this.disrupted) tau *= 0.02;
      this.tauE = Math.max(tau, 1e-3);

      /* --- energy balance (separate electron / ion channels) -------------- */
      const nE = this.ne * 1e20, nI = nE * (this.fDT + this.nHe / Math.max(this.ne, 1e-9));
      const cE = 1.5 * nE * KEV_J * M.V / MW;     // MJ per keV of Te
      const cI = 1.5 * Math.max(nI, 1e17) * KEV_J * M.V / MW;

      /* electron-ion equipartition time (Braginskii, ~s) */
      const tauEq = Math.max(0.02, 0.032 * Math.pow(Math.max(this.Te, 0.05), 1.5) /
                    Math.max(this.ne, 0.02) / this.Zeff);
      const Pei = cE * (this.Ti - this.Te) / tauEq;   // ion -> electron transfer

      const Pa_e = this.Palpha * 0.58, Pa_i = this.Palpha * 0.42;
      const Pnb_e = this.Pnbi * 0.35, Pnb_i = this.Pnbi * 0.65;
      const Pic_e = this.Picr * 0.30, Pic_i = this.Picr * 0.70;

      const We = cE * this.Te, Wi = cI * this.Ti;
      const dWe = Pa_e + Pnb_e + Pic_e + this.Pecr + this.Pohm + Pei
                  - this.Prad - We / this.tauE;
      const dWi = Pa_i + Pnb_i + Pic_i - Pei - Wi / this.tauE;

      this.Te = Math.max(0.02, this.Te + dWe / cE * dt);
      this.Ti = Math.max(0.02, this.Ti + dWi / cI * dt);
      this.We = cE * this.Te; this.Wi = cI * this.Ti;
      this.W  = this.We + this.Wi;

      /* --- particle balance ---------------------------------------------- */
      const tauP = 2.0 * this.tauE;
      const Sgas = this.gas * 0.30;                       // fuelling  [1e20/s]
      const Snbi = this.Pnbi * 0.0016;
      this.ne = Math.max(0.005, this.ne + (Sgas + Snbi - this.ne / tauP) * dt);

      /* helium ash from the fusion rate, pumped with tau_He ~ 5 tau_E */
      const Sfus = this.Pfus * MW / E_FUS / M.V / 1e20;   // [1e20 m^-3 s^-1]
      this.nHe = Math.max(0, this.nHe + (Sfus - this.nHe / (5 * this.tauE)) * dt);

      /* neutral pressure: puff in, torus cryopumps out */
      const pTarget = 1.2e-5 + this.gas * 4.5e-4 + (1 - clamp(this.ne * 2, 0, 1)) * 1e-4;
      this.pVac += (pTarget - this.pVac) * clamp(dt / 0.4, 0, 1);

      /* --- safety factor, limits ----------------------------------------- */
      const shape = (1 + this.M.kappa * this.M.kappa *
                    (1 + 2 * M.delta * M.delta - 1.2 * Math.pow(M.delta, 3))) / 2;
      this.q95 = this.Ip > 0.05
        ? 5 * M.a * M.a * this.Bt / (M.R0 * this.Ip) * shape : 99;
      this.q0  = clamp(this.q95 / (2.6 + 1.4 * this.li), 0.55, 30);
      this.nG  = this.Ip / (Math.PI * M.a * M.a);         // Greenwald  [1e20]
      this.fG  = this.ne / Math.max(this.nG, 1e-6);
      /* beta[%] = 2 mu0 <p> / B^2, with n in 1e20 m^-3 and T in keV */
      this.beta = 4.027 * (this.ne * (this.Te + this.Ti)) / (this.Bt * this.Bt);
      this.betaN = this.Ip > 0.05
        ? this.beta * M.a * this.Bt / this.Ip : 0;

      /* --- MHD activity: sawteeth and ELMs -------------------------------- */
      this.mhdPhase += dt * (8 + 30 * this.beta);
      if (this.q0 < 1.0 && this.Te > 1.5) {
        this.sawPhase += dt / (0.35 + 1.9 * clamp(this.tauE, 0, 1.6));
        if (this.sawPhase >= 1) {
          this.sawPhase = 0; this.sawFlash = 1; this.sawCount++;
          this.Te *= 0.935; this.Ti *= 0.945;
        }
      }
      this.sawFlash = Math.max(0, this.sawFlash - dt * 5);

      if (this.hMode && !this.disrupted) {
        /* type-I ELM frequency scales with the power crossing the separatrix */
        const fELM = clamp(0.35 + 0.020 * this.Ploss, 0.35, 2.6);   // Hz
        this.elmPhase += dt * fELM;
        if (this.elmPhase >= 1) {
          this.elmPhase = 0; this.elmFlash = 1; this.elmCount++;
          /* ~1.5-2.5% of W per ELM — the mitigated regime the ELM control
             coils are designed to hold, and the basis of the Q=10 point   */
          const frac = 0.012 + 0.011 * this.rngf();
          this.Te *= (1 - frac); this.Ti *= (1 - frac);
          this.ne *= (1 - frac * 0.7);
        }
      }
      this.elmFlash = Math.max(0, this.elmFlash - dt * 9);

      /* --- disruption logic ---------------------------------------------- */
      /* Only police the operational limits once there is a real current
         channel — during breakdown the plasma is far from any of them. */
      if (!this.disrupted && this.t > 5 && this.Ip > 2.0) {
        let cause = null;
        if (this.fG > 1.25) cause = 'ТЫҒЫЗДЫҚ ШЕГІ — ГРИНВАЛЬД АСЫП КЕТТІ';
        else if (this.q95 < 2.0 && this.Ip > 2) cause = 'q95 ТӨМЕН — КИНК ТҰРАҚСЫЗДЫҒЫ';
        else if (this.betaN > 4.2) cause = 'БЕТА ШЕГІ — ИДЕАЛ МГД';
        else if (this.Prad > (this.Paux + this.Pohm + this.Palpha) * 1.35 && this.Te > 1)
          cause = 'СӘУЛЕЛЕНУ КОЛЛАПСЫ';
        if (cause) this.triggerDisruption(cause);
      }
      if (this.disrupted) {
        /* thermal quench then current quench */
        const dtq = this.t - this.disruptT;
        if (dtq < 0.003) { this.Te *= 0.2; this.Ti *= 0.2; }
        this.Ip = Math.max(0, this.Ip - dt * 45);
        this.Te = Math.max(0.02, this.Te - dt * 8);
        this.Ti = Math.max(0.02, this.Ti - dt * 8);
        if (this.Ip < 0.05 && dtq > 2.5) {
          this.disrupted = false; this.hMode = false;
          this.reset(); this.autoPilot = true;
        }
      }

      /* --- rotation phase used by the renderer ---------------------------- */
      const vTor = 0.35 + 0.9 * clamp(this.Ti / 20, 0, 1.4) + 0.02 * this.Pnbi;
      this.rotPhase += dt * vTor;
    }

    /* -------------------------------------------------------------------- */
    updateDerived(dt) {
      this.Q = this.Paux > 0.3 ? this.Pfus / this.Paux : (this.Pfus > 1 ? 999 : 0);
      this.Pnet = this.Pfus * 0.33 - (this.Paux / 0.4);   // electrical balance
      this.neutronRate = this.Pfus * MW * (1 - E_ALPHA_F) / (14.06e6 * 1.602e-19);
      this.neutronFlux = this.neutronRate / M.S;
      this.qDiv = (this.Psep * (1 - 0.65) / M.Sdiv) * (1 + 5.0 * this.elmFlash);

      this.fluxTor = Math.PI * M.a * M.a * this.Bt * M.kappa;      // [Wb]
      this.fluxPol = 1.256e-6 * this.Ip * 1e6 * M.R0 *
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
        this.pfI[i] = pfBase[i] * (12 + this.Ip * 2.6) *
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
      if (this.disrupted)                    this.modeKey = 'disrupt';
      else if (this.Ip < 0.5)                this.modeKey = 'breakdown';
      else if (this.Ip < this._peakIp - 0.5) this.modeKey = 'rampdown';
      else if (this.Ip < M.IpNom * 0.93)     this.modeKey = 'rampup';
      else if (this.Q > 4)                   this.modeKey = 'burn';
      else if (this.hMode)                   this.modeKey = 'hmode';
      else                                   this.modeKey = 'lmode';
      this.mode = Tokamak.MODE_LABEL[this.modeKey];
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
  global.Tokamak = Tokamak;
})(window);
