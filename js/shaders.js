/* ==========================================================================
   shaders.js — GLSL ES 3.00 sources for the tokamak interior render
   --------------------------------------------------------------------------
   Pass 1  SCENE      analytic vessel intersection + volumetric plasma march
   Pass 2  BRIGHT     threshold / karis-average downsample
   Pass 3  BLUR       separable gaussian (run per mip, H then V)
   Pass 4  COMPOSITE  bloom combine, ACES tonemap, heat shimmer, grain
   ========================================================================== */
(function (global) {
  'use strict';

  /* ---------------------------------------------------------------- common */
  const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main(){
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

  /* Shared noise / hash utilities. */
  const NOISE = `
float hash11(float p){ p = fract(p*0.1031); p *= p+33.33; p *= p+p; return fract(p); }
float hash13(vec3 p3){
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec3 x){
  vec3 i = floor(x), f = fract(x);
  f = f*f*(3.0-2.0*f);
  float n000=hash13(i+vec3(0,0,0)), n100=hash13(i+vec3(1,0,0));
  float n010=hash13(i+vec3(0,1,0)), n110=hash13(i+vec3(1,1,0));
  float n001=hash13(i+vec3(0,0,1)), n101=hash13(i+vec3(1,0,1));
  float n011=hash13(i+vec3(0,1,1)), n111=hash13(i+vec3(1,1,1));
  return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y),
             mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);
}
float fbm3(vec3 p){
  float a = 0.5, s = 0.0;
  for(int i=0;i<3;i++){ s += a*vnoise(p); p *= 2.03; a *= 0.5; }
  return s;
}
float fbm4(vec3 p){
  float a = 0.5, s = 0.0;
  for(int i=0;i<4;i++){ s += a*vnoise(p); p *= 2.07; a *= 0.5; }
  return s;
}`;

  /* ================================================================= SCENE */
  const SCENE = `#version 300 es
precision highp float;
precision highp int;

in  vec2 vUv;
layout(location = 0) out vec4 fragColor;

uniform vec2  uRes;
uniform float uTime;
uniform vec3  uCamPos;
uniform mat3  uCamMat;
uniform float uFovHalf;     // half-angle of the fisheye mapping [rad]

/* --- plasma state, all normalised 0..1 unless noted -------------------- */
uniform float uRot;         // toroidal rotation phase
uniform float uEmis;        // overall emissivity
uniform float uTemp;        // 0..1 normalised temperature (colour balance)
uniform float uDens;        // 0..1 normalised density (opacity)
uniform float uInstab;      // turbulence amplitude
uniform float uElm;         // ELM burst 0..1
uniform float uSaw;         // sawtooth crash 0..1
uniform float uIgnite;      // burning-plasma fraction
uniform float uDisrupt;     // disruption 0..1
uniform float uNbi;         // neutral beam power 0..1
uniform float uRf;          // RF heating 0..1
uniform float uElong;       // plasma elongation
uniform float uHmode;       // 0 = L-mode, 1 = H-mode (pedestal sharpness)
uniform int   uSteps;       // volumetric march steps
uniform int   uLights;      // ring-light samples
uniform float uFieldLines;  // magnetic field line overlay 0..1
uniform float uParticles;   // charged particle trace overlay 0..1

${NOISE}

/* ---- machine geometry in scene units ---------------------------------- */
const float R0    = 2.02;    // plasma major radius
const float AMIN  = 0.94;    // plasma minor radius
const float DELTA = 0.42;    // triangularity
const float RIN   = 1.03;    // central column (inner wall) radius
const float ROUT  = 3.66;    // outer first-wall radius
const float RPMAX = 3.30;    // bounding radius of all emitting plasma
const float YTOP  = 2.42;
const float YBOT  = -2.30;
const float PI    = 3.14159265359;
const float TAU   = 6.28318530718;

/* =======================================================================
   Surfaces of revolution — analytic intersection
   ======================================================================= */

/* radius of the central column as a function of height (slight barrel) */
float colRadius(float y){
  return RIN * (1.0 + 0.045*smoothstep(0.2, 2.3, y) + 0.018*smoothstep(-2.2,-0.9,-y));
}

/* Quadratic ray/cylinder. near=true -> smallest positive root. */
float cylHit(vec3 ro, vec3 rd, float R, bool near){
  float a = dot(rd.xz, rd.xz);
  if(a < 1e-6) return -1.0;
  float b = 2.0*dot(ro.xz, rd.xz);
  float c = dot(ro.xz, ro.xz) - R*R;
  float d = b*b - 4.0*a*c;
  if(d < 0.0) return -1.0;
  d = sqrt(d);
  float t0 = (-b - d)/(2.0*a), t1 = (-b + d)/(2.0*a);
  if(near) return t0 > 1e-3 ? t0 : (t1 > 1e-3 ? t1 : -1.0);
  return t1 > 1e-3 ? t1 : -1.0;
}

/* Newton refinement against the height-varying column radius. */
float columnHit(vec3 ro, vec3 rd){
  float t = cylHit(ro, rd, RIN, true);
  if(t < 0.0) return -1.0;
  for(int i=0;i<3;i++){
    vec3 p = ro + rd*t;
    float f = length(p.xz) - colRadius(p.y);
    float dr = dot(normalize(p.xz), rd.xz) - (colRadius(p.y+0.01)-colRadius(p.y-0.01))/0.02*rd.y;
    if(abs(dr) < 1e-4) break;
    t -= f/dr;
  }
  vec3 p = ro + rd*t;
  if(t < 1e-3 || p.y > YTOP || p.y < YBOT) return -1.0;
  return t;
}

float planeHit(vec3 ro, vec3 rd, float y){
  if(abs(rd.y) < 1e-6) return -1.0;
  float t = (y - ro.y)/rd.y;
  if(t < 1e-3) return -1.0;
  vec3 p = ro + rd*t;
  float r = length(p.xz);
  if(r < colRadius(y)*0.98 || r > ROUT*1.02) return -1.0;
  return t;
}

/* =======================================================================
   Plasma — flux-surface coordinate, emission and extinction
   ======================================================================= */

/* Normalised flux coordinate rho for a point p, with elongation and a
   D-shaped (triangular) cross-section approximated by an angular warp.  */
float fluxRho(vec3 p, out float theta){
  float r  = length(p.xz);
  float u  = r - R0;
  float v  = p.y / max(uElong, 0.3);
  theta    = atan(v, u);
  /* D-shape: outboard bulge, inboard flattening, corners pulled inward */
  float shp = 1.0 + 0.135*DELTA*cos(theta) - 0.115*DELTA*cos(2.0*theta);
  return length(vec2(u, v)) / (AMIN * shp);
}

/* Distance to the two divertor legs in the poloidal plane (r,y).
   The X-point sits below the core; the legs run down to the targets. */
float divertorLegs(float r, float y){
  vec2 X  = vec2(R0 - 0.30, -1.28);            // X-point
  vec2 T1 = vec2(R0 - 0.86, YBOT + 0.30);      // inner strike point
  vec2 T2 = vec2(R0 + 0.52, YBOT + 0.24);      // outer strike point
  vec2 q  = vec2(r, y);
  vec2 e1 = T1 - X, w1 = q - X;
  float h1 = clamp(dot(w1,e1)/dot(e1,e1), 0.0, 1.0);
  float d1 = length(w1 - e1*h1);
  vec2 e2 = T2 - X, w2 = q - X;
  float h2 = clamp(dot(w2,e2)/dot(e2,e2), 0.0, 1.0);
  float d2 = length(w2 - e2*h2);
  /* the outer leg carries most of the power */
  return min(d1*1.35, d2*1.0);
}

/* Emission (rgb, W/sr-ish arbitrary) and extinction (a) at a point. */
vec4 plasma(vec3 p){
  float r   = length(p.xz);
  float phi = atan(p.z, p.x);
  float th;
  float rho = fluxRho(p, th);

  if(rho > 2.4 && p.y > -1.0) return vec4(0.0);

  /* --- turbulence: drift-wave filaments, ballooning on the outboard side */
  float outb = smoothstep(-0.2, 0.9, cos(th));                 // low-field side
  vec3 tp = vec3(phi*2.6 - uRot*1.4, th*2.2 + uRot*0.35, rho*4.0 - uTime*0.20);
  float turb = fbm4(tp) - 0.5;
  vec3 tp2 = vec3(phi*9.0 - uRot*2.1, th*6.5, uTime*0.9);
  float fil = (fbm3(tp2) - 0.5) * outb;

  float amp = uInstab * (0.055 + 0.14*uDisrupt);
  float rhoT = rho * (1.0 + amp*turb + amp*0.9*fil*smoothstep(0.55,1.05,rho));

  /* --- radial emissivity structure -------------------------------------
     The core is hot and optically thin (weak visible emission, blue).
     Visible light comes mostly from the cool edge where recombination and
     line radiation dominate -> the characteristic magenta shell.          */
  float pedW = mix(0.105, 0.062, uHmode);
  float edge = exp(-pow((rhoT - 0.972)/pedW, 2.0)) * 1.7;
  float core = exp(-rhoT*rhoT*1.65) * (0.10 + 0.22*uIgnite + 0.45*uSaw);
  float sol  = exp(-pow((rhoT - 1.10)/0.105, 2.0)) * 0.20;
  /* The scrape-off layer is thin: beyond it the vessel is optically empty,
     otherwise the plasma nearest the lens fogs the whole frame.           */
  float bound = 1.0 - smoothstep(1.06, 1.30, rhoT);
  edge *= bound; sol *= bound; core *= bound;
  /* Poloidal weighting: the boundary radiates hardest near the X-point at
     the bottom of the vessel and is faintest at the top.                  */
  edge *= 0.30 + 0.95*(0.5 - 0.5*sin(th));

  /* --- divertor / X-point region ---------------------------------------- */
  float dleg = divertorLegs(r, p.y);
  float div  = exp(-pow(dleg/0.135, 2.0)) * (1.0 + 3.2*uElm);
  float strike = exp(-pow((p.y - (YBOT+0.26))/0.16, 2.0)) *
                 exp(-pow((abs(r - R0) - 0.62)/0.42, 2.0)) * 1.5;

  /* --- toroidal ripple from the discrete TF coils (18 fold) -------------- */
  float ripple = 1.0 + 0.035*cos(phi*18.0 + 0.4) * smoothstep(0.6, 1.2, rho);

  float kill = 1.0 - 0.55*uDisrupt*step(0.5, hash11(floor(uTime*22.0)+floor(phi*3.0)));
  edge *= kill; sol *= kill; core *= kill;

  /* --- colour ------------------------------------------------------------
     Hot core   -> blue / white     (bremsstrahlung + high ionisation stages)
     Cool edge  -> magenta / violet (Balmer + impurity line emission)
     Divertor   -> near white-pink  (dense recombining plasma)              */
  vec3 cCore = mix(vec3(0.35,0.52,1.00), vec3(0.72,0.82,1.00), uTemp);
  vec3 cEdge = mix(vec3(0.85,0.16,0.92), vec3(1.00,0.32,0.86), uTemp);
  vec3 cDiv  = vec3(1.00,0.62,0.92);

  /* vertical tint: the upper chamber reads bluer, the lower more magenta   */
  float vt = smoothstep(-1.4, 1.9, p.y);
  cEdge = mix(cEdge, vec3(0.50,0.42,1.00), vt*0.55);

  vec3 col = cCore*core*1.25 + cEdge*(edge + sol)*ripple;
  col += cDiv * (div*4.50 + strike*1.80);

  /* --- ELM filaments: bright transient blobs ejected across the separatrix */
  if(uElm > 0.001){
    float elmF = fbm3(vec3(phi*7.0, th*3.0, uTime*6.0));
    float elmShell = exp(-pow((rhoT - 1.03)/0.15, 2.0)) * outb;
    col += vec3(1.0,0.55,0.85) * uElm * elmShell * (0.5 + elmF) * 3.2;
  }

  /* --- neutral beam footprint: tangential injection, two sources --------- */
  if(uNbi > 0.01){
    for(int b=0;b<2;b++){
      float off = (b==0) ? 0.0 : 2.2;
      /* beam axis: tangential chord, slightly above the midplane */
      vec3 bo = vec3(cos(off)*3.2, 0.16, sin(off)*3.2);
      vec3 bd = normalize(vec3(-cos(off+1.05), -0.02, -sin(off+1.05)));
      vec3 w  = p - bo;
      float h = clamp(dot(w,bd), 0.0, 4.4);
      float d = length(w - bd*h);
      float atten = exp(-h*0.55);                 // beam deposition profile
      col += vec3(0.55,0.85,1.00) * uNbi * exp(-pow(d/0.13,2.0)) * atten * 1.5;
    }
  }

  /* --- RF heating: resonance layer on the low-field side ----------------- */
  if(uRf > 0.01){
    float res = exp(-pow((r - (R0 - 0.42))/0.085, 2.0)) *
                exp(-pow(p.y/1.05, 2.0)) * step(rho, 1.0);
    col += vec3(0.45,1.00,0.95)*uRf*res*0.55*(0.7+0.3*sin(uTime*30.0));
  }

  /* --- helical field lines / fast particle orbits ------------------------ */
  if(uFieldLines > 0.01){
    float m = 3.0, n = 2.0;                       // q=3/2 rational surface
    float hel = cos(m*th - n*phi + uRot*0.9);
    float band = exp(-pow((rho - 0.72)/0.05, 2.0));
    col += vec3(0.40,0.95,1.00)*uFieldLines*band*smoothstep(0.90,1.0,hel)*2.2;
  }
  if(uParticles > 0.01){
    float sp = hash11(floor(phi*22.0) + floor(th*9.0)*17.0);
    float trail = fract(phi*3.5 + uTime*(0.6+sp*1.8) + sp*10.0);
    float band = exp(-pow((rho - (0.30+sp*0.65))/0.035, 2.0));
    col += vec3(0.85,0.95,1.00)*uParticles*band*
           pow(max(0.0,1.0-trail*6.0),3.0)*1.6;
  }

  /* The plasma is optically thin in the visible: extinction stays low so
     the first wall and the central column remain readable through it.     */
  float ext = (edge + sol + core*0.5) * (0.03 + 0.09*uDens);
  return vec4(col * uEmis * 0.26, ext);
}

/* =======================================================================
   Surface shading
   ======================================================================= */

/* Cook-Torrance-lite point light. */
vec3 lightPoint(vec3 p, vec3 n, vec3 v, vec3 lp, vec3 lc, float rough, float metal, vec3 alb){
  vec3 ld = lp - p;
  float d2 = dot(ld, ld);
  ld = normalize(ld);
  float ndl = max(dot(n, ld), 0.0);
  if(ndl <= 0.0) return vec3(0.0);
  vec3 h = normalize(ld + v);
  float ndh = max(dot(n, h), 0.0);
  float ndv = max(dot(n, v), 1e-3);
  float a  = max(rough*rough, 0.004);
  float a2 = a*a;
  float dnm = ndh*ndh*(a2-1.0)+1.0;
  float D = a2/(PI*dnm*dnm);
  float k = a*0.5;
  float G = (ndl/(ndl*(1.0-k)+k)) * (ndv/(ndv*(1.0-k)+k));
  float F = 0.04 + 0.96*pow(1.0 - max(dot(h,v),0.0), 5.0);
  vec3 spec = vec3(D*G*F/(4.0*ndv*ndl + 1e-4));
  vec3 diff = alb*(1.0-metal)/PI;
  return (diff + spec) * lc * ndl / (0.55 + d2*0.75);
}

/* The plasma acts as three coaxial ring lights: the main emissive shell,
   the bright divertor ring, and the cooler upper region. */
vec3 ringLighting(vec3 p, vec3 n, vec3 v, float rough, float metal, vec3 alb){
  vec3 acc = vec3(0.0);
  float N = float(uLights);
  vec3 cMid = mix(vec3(0.90,0.18,0.95), vec3(1.00,0.35,0.88), uTemp);
  vec3 cDiv = vec3(1.00,0.58,0.90);
  vec3 cTop = vec3(0.42,0.46,1.00);
  float e = uEmis * (1.0 + 1.6*uElm);
  for(int i=0;i<48;i++){
    if(i >= uLights) break;
    float a = (float(i)+0.5)/N * TAU + uRot*0.03;
    vec2 cs = vec2(cos(a), sin(a));
    float wob = 0.10*sin(a*4.0 + uTime*1.3);
    acc += lightPoint(p,n,v, vec3(cs.x*(R0+0.55), 0.10+wob, cs.y*(R0+0.55)),
                      cMid*e*1.00, rough, metal, alb);
    acc += lightPoint(p,n,v, vec3(cs.x*(R0-0.60), -0.05-wob, cs.y*(R0-0.60)),
                      cMid*e*0.55, rough, metal, alb);
    acc += lightPoint(p,n,v, vec3(cs.x*1.92, YBOT+0.34, cs.y*1.92),
                      cDiv*e*1.45, rough, metal, alb);
    acc += lightPoint(p,n,v, vec3(cs.x*2.25, 1.30, cs.y*2.25),
                      cTop*e*0.50, rough, metal, alb);
  }
  /* acc holds 4 rings x N samples; divide by N so the result is the sum of
     four ring averages and stays independent of the sample count.        */
  return acc * (6.5/N);
}

/* Beryllium/tungsten first-wall panels: tile grid, bolts, cooling channels. */
void wallMaterial(float u, float v, float scaleU, float scaleV,
                  out vec3 alb, out float rough, out float bump, out float ao)
{
  vec2 g = vec2(u*scaleU, v*scaleV);
  vec2 cell = floor(g);
  vec2 f = fract(g);
  float seedT = hash13(vec3(cell, 3.7));

  /* panel gaps */
  float gx = min(f.x, 1.0-f.x), gy = min(f.y, 1.0-f.y);
  float gap = smoothstep(0.0, 0.035, gx) * smoothstep(0.0, 0.030, gy);

  /* bolt heads at the panel corners */
  float bolt = 0.0;
  for(int i=0;i<4;i++){
    vec2 bp = vec2(float(i&1), float((i>>1)&1));
    bp = mix(vec2(0.13,0.13), vec2(0.87,0.87), bp);
    bolt = max(bolt, 1.0 - smoothstep(0.030, 0.048, length(f-bp)));
  }
  /* horizontal cooling channel ribs */
  float rib = 0.5 + 0.5*sin(f.y*TAU*4.0);

  float wear = fbm3(vec3(g*3.0, 1.7));
  alb = mix(vec3(0.090,0.095,0.105), vec3(0.205,0.200,0.185), seedT*0.6 + wear*0.4);
  alb = mix(alb, vec3(0.30,0.27,0.24), bolt*0.7);
  alb *= (0.42 + 0.78*gap);

  rough = mix(0.62, 0.30, seedT*0.5 + wear*0.5);
  rough = mix(rough, 0.22, bolt);
  bump  = gap*0.6 + bolt*0.5 + rib*0.06;
  ao    = mix(0.45, 1.0, gap) * (0.9 + 0.1*rib);
}

/* Perturb a normal by a scalar height field using screen-space-free
   analytic tangents supplied by the caller. */
vec3 bumpNormal(vec3 n, vec3 t1, vec3 t2, float h, float hu, float hv, float k){
  return normalize(n - (hu-h)*k*t1 - (hv-h)*k*t2);
}

/* =======================================================================
   Main
   ======================================================================= */
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes) / uRes.y;

  /* wide-angle (equidistant fisheye) projection to match a port-view lens */
  float rr = length(uv);
  float ang = rr * uFovHalf;
  vec2 dirxy = rr > 1e-5 ? uv/rr : vec2(0.0);
  vec3 rd = uCamMat * vec3(dirxy*sin(ang), -cos(ang));
  vec3 ro = uCamPos;

  /* ---- solid geometry -------------------------------------------------- */
  float tCol = columnHit(ro, rd);
  float tOut = cylHit(ro, rd, ROUT, false);
  if(tOut > 0.0){
    vec3 p = ro + rd*tOut;
    if(p.y > YTOP || p.y < YBOT) tOut = -1.0;
  }
  float tTop = planeHit(ro, rd, YTOP);
  float tBot = planeHit(ro, rd, YBOT);

  int   id = 0;                 // 0 none, 1 column, 2 outer wall, 3 top, 4 bottom
  float tS = 1e9;
  if(tCol > 0.0 && tCol < tS){ tS = tCol; id = 1; }
  if(tOut > 0.0 && tOut < tS){ tS = tOut; id = 2; }
  if(tTop > 0.0 && tTop < tS){ tS = tTop; id = 3; }
  if(tBot > 0.0 && tBot < tS){ tS = tBot; id = 4; }

  vec3 surf = vec3(0.0);
  if(id != 0){
    vec3 p = ro + rd*tS;
    vec3 v = -rd;
    float phi = atan(p.z, p.x);
    float r   = length(p.xz);
    vec3 n, t1, t2;
    vec3 alb; float rough, bump, ao, hu, hv;
    float su, sv;

    if(id == 1){                                   // ---- central column
      n  = normalize(vec3(p.x, 0.0, p.z));
      n  = normalize(n + vec3(0.0, 0.14, 0.0));    // barrel taper
      t1 = normalize(cross(vec3(0,1,0), n)); t2 = cross(n, t1);
      su = 26.0; sv = 3.4;
      float u = (phi+PI)/TAU, vv = (p.y - YBOT)/(YTOP-YBOT);
      wallMaterial(u, vv, su, sv, alb, rough, bump, ao);
      wallMaterial(u+0.0016, vv, su, sv, alb, rough, hu, ao);
      wallMaterial(u, vv+0.0016, su, sv, alb, rough, hv, ao);
      n = bumpNormal(n, t1, t2, bump, hu, hv, 0.9);
      alb *= 1.06;
    } else if(id == 2){                            // ---- outer first wall
      n  = -normalize(vec3(p.x, 0.0, p.z));
      t1 = normalize(cross(vec3(0,1,0), n)); t2 = cross(n, t1);
      su = 44.0; sv = 7.0;
      float u = (phi+PI)/TAU, vv = (p.y - YBOT)/(YTOP-YBOT);
      wallMaterial(u, vv, su, sv, alb, rough, bump, ao);
      wallMaterial(u+0.0010, vv, su, sv, alb, rough, hu, ao);
      wallMaterial(u, vv+0.0010, su, sv, alb, rough, hv, ao);
      n = bumpNormal(n, t1, t2, bump, hu, hv, 0.7);

      /* diagnostic / heating ports: dark recessed rectangles */
      float pu = fract(u*9.0) - 0.5;
      float pv = (p.y - 0.35)/1.05;
      float port = (1.0 - smoothstep(0.16,0.20,abs(pu))) *
                   (1.0 - smoothstep(0.42,0.50,abs(pv)));
      alb = mix(alb, vec3(0.012,0.014,0.018), port);
      rough = mix(rough, 0.85, port);
      ao   = mix(ao, 0.18, port);
      /* viewport glass ring highlight */
      float ring = smoothstep(0.20,0.185,abs(pu))*smoothstep(0.50,0.47,abs(pv));
      alb = mix(alb, vec3(0.20,0.21,0.24), max(0.0, ring-port));
    } else if(id == 3){                            // ---- upper structure
      n  = vec3(0.0,-1.0,0.0);
      t1 = vec3(1,0,0); t2 = vec3(0,0,1);
      su = 30.0; sv = 5.0;
      float u = (phi+PI)/TAU, vv = (r - RIN)/(ROUT-RIN);
      wallMaterial(u, vv, su, sv, alb, rough, bump, ao);
      wallMaterial(u+0.0014, vv, su, sv, alb, rough, hu, ao);
      wallMaterial(u, vv+0.0014, su, sv, alb, rough, hv, ao);
      n = bumpNormal(n, t1, t2, bump, hu, hv, 0.8);
      alb *= 0.85;
      /* upper port plugs, radially arranged */
      float seg = fract(u*18.0)-0.5;
      float plug = (1.0-smoothstep(0.20,0.26,abs(seg))) *
                   smoothstep(0.25,0.35,vv)*(1.0-smoothstep(0.72,0.80,vv));
      alb = mix(alb, vec3(0.02,0.022,0.026), plug*0.8);
      ao  = mix(ao, 0.3, plug);
    } else {                                       // ---- divertor cassettes
      n  = vec3(0.0,1.0,0.0);
      t1 = vec3(1,0,0); t2 = vec3(0,0,1);
      su = 54.0; sv = 3.0;
      float u = (phi+PI)/TAU, vv = (r - RIN)/(ROUT-RIN);
      wallMaterial(u, vv, su, sv, alb, rough, bump, ao);
      wallMaterial(u+0.0009, vv, su, sv, alb, rough, hu, ao);
      wallMaterial(u, vv+0.0009, su, sv, alb, rough, hv, ao);
      n = bumpNormal(n, t1, t2, bump, hu, hv, 0.9);
      /* tungsten monoblock targets are shinier and heat-discoloured */
      float strike = exp(-pow((r-(R0+0.52))/0.24,2.0)) +
                     0.7*exp(-pow((r-(R0-0.86))/0.20,2.0));
      alb   = mix(alb, vec3(0.16,0.13,0.12), strike*0.8);
      rough = mix(rough, 0.16, strike*0.85);
    }

    /* Plasma-facing armour is oxidised and beam-blasted: low metalness so
       the tile albedo pattern reads, rather than a pure mirror.           */
    surf = ringLighting(p, n, v, rough, 0.25, alb) * ao;

    /* ambient plasma bounce (crude global illumination) */
    vec3 amb = mix(vec3(0.30,0.10,0.45), vec3(0.16,0.20,0.55),
                   smoothstep(-1.0,1.5,p.y));
    surf += alb * amb * uEmis * 0.16 * ao;

    /* thermal glow of the divertor targets when the heat flux is high */
    if(id == 4){
      float strike = exp(-pow((r-(R0+0.52))/0.26,2.0)) +
                     0.7*exp(-pow((r-(R0-0.86))/0.22,2.0));
      surf += vec3(1.0,0.42,0.16) * strike * uEmis * (0.10 + 0.55*uElm);
    }
    /* electromagnetic arcing on the wall during transients */
    float arcSeed = hash11(floor(uTime*13.0)*7.7 + floor(phi*11.0));
    if(arcSeed > 0.985 - 0.05*uElm - 0.35*uDisrupt){
      float ay = hash11(arcSeed*53.0)*2.0-1.0;
      float arc = exp(-pow((p.y-ay*1.6)/0.05,2.0)) *
                  exp(-pow((fract(phi*11.0)-0.5)/0.09,2.0));
      surf += vec3(0.75,0.90,1.00)*arc*7.0;
    }
  }

  /* ---- volumetric plasma march -----------------------------------------
     All emitting plasma lies inside the cylinder r < RPMAX, so the ray is
     clipped to that slab before stepping.  This concentrates the samples
     where they matter — the emission shell is only a few centimetres wide
     and would otherwise alias badly.                                      */
  float tmax = min(id != 0 ? tS : 12.0, 12.0);
  float tnear = 0.0;
  {
    float qa = dot(rd.xz, rd.xz);
    float qb = 2.0*dot(ro.xz, rd.xz);
    float qc = dot(ro.xz, ro.xz) - RPMAX*RPMAX;
    float disc = qb*qb - 4.0*qa*qc;
    if(disc <= 0.0 || qa < 1e-6){ tmax = 0.0; }
    else {
      float sd = sqrt(disc);
      tnear = max(0.0, (-qb - sd)/(2.0*qa));
      tmax  = min(tmax, (-qb + sd)/(2.0*qa));
    }
  }

  vec3 acc = vec3(0.0);
  float trans = 1.0;
  float N  = float(uSteps);
  float dt = max(tmax - tnear, 0.0) / N;
  float jitter = hash13(vec3(gl_FragCoord.xy, floor(uTime*60.0)));
  float t = tnear + dt*jitter;

  for(int i=0;i<256;i++){
    if(i >= uSteps || t >= tmax || trans < 0.004) break;
    vec3 p = ro + rd*t;
    vec4 e = plasma(p);
    if(e.a > 1e-5 || e.r+e.g+e.b > 1e-5){
      acc   += e.rgb * trans * dt;
      trans *= exp(-e.a * dt * 0.55);
    }
    t += dt;
  }

  vec3 col = surf * trans + acc;

  /* store plasma opacity in alpha — the composite pass uses it to drive
     the refractive heat shimmer */
  fragColor = vec4(col, clamp(1.0 - trans, 0.0, 1.0));
}`;

  /* ================================================================ BRIGHT */
  const BRIGHT = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uSoft;
void main(){
  vec3 s = vec3(0.0);
  s += texture(uTex, vUv + uTexel*vec2(-1.0,-1.0)).rgb;
  s += texture(uTex, vUv + uTexel*vec2( 1.0,-1.0)).rgb;
  s += texture(uTex, vUv + uTexel*vec2(-1.0, 1.0)).rgb;
  s += texture(uTex, vUv + uTexel*vec2( 1.0, 1.0)).rgb;
  s *= 0.25;
  float l = dot(s, vec3(0.2126, 0.7152, 0.0722));
  float k = clamp((l - uThreshold + uSoft) / (2.0*uSoft + 1e-4), 0.0, 1.0);
  k = k*k*(l > 0.0 ? 1.0 : 0.0) * max(l - uThreshold, 0.0) / max(l, 1e-4);
  fragColor = vec4(s * k, 1.0);
}`;

  /* ================================================================== BLUR */
  const BLUR = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uDir;          // texel-sized step along the blur axis
void main(){
  /* 9-tap gaussian using linear-sampling offsets */
  const float o1 = 1.3846153846, o2 = 3.2307692308;
  const float w0 = 0.2270270270, w1 = 0.3162162162, w2 = 0.0702702703;
  vec3 c = texture(uTex, vUv).rgb * w0;
  c += texture(uTex, vUv + uDir*o1).rgb * w1;
  c += texture(uTex, vUv - uDir*o1).rgb * w1;
  c += texture(uTex, vUv + uDir*o2).rgb * w2;
  c += texture(uTex, vUv - uDir*o2).rgb * w2;
  fragColor = vec4(c, 1.0);
}`;

  /* ============================================================= COMPOSITE */
  const COMPOSITE = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uScene;
uniform sampler2D uBloom0;
uniform sampler2D uBloom1;
uniform sampler2D uBloom2;
uniform vec2  uRes;
uniform float uTime;
uniform float uBloomAmt;
uniform float uExposure;
uniform float uShimmer;
uniform float uElm;
uniform float uDisrupt;
uniform float uGrain;
uniform float uVignette;

${NOISE}

/* ACES filmic tonemap (Narkowicz fit) */
vec3 aces(vec3 x){
  const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main(){
  vec2 uv = vUv;
  vec2 cc = uv - 0.5;
  float r2 = dot(cc, cc);

  /* --- heat shimmer: refraction through the hot boundary layer ----------- */
  float heat = texture(uScene, uv).a;
  vec2 wob = vec2(
    fbm3(vec3(uv*14.0, uTime*1.7)) - 0.5,
    fbm3(vec3(uv*14.0 + 31.7, uTime*1.9)) - 0.5);
  vec2 duv = wob * uShimmer * (0.25 + heat) * (1.0 + 3.0*uElm) * 0.006;
  uv += duv;

  /* --- chromatic aberration toward the frame edge ------------------------ */
  float ca = (0.0016 + 0.010*r2) * (1.0 + 2.0*uDisrupt);
  vec3 col;
  col.r = texture(uScene, uv - cc*ca).r;
  col.g = texture(uScene, uv).g;
  col.b = texture(uScene, uv + cc*ca).b;

  /* --- bloom ------------------------------------------------------------- */
  vec3 b = texture(uBloom0, uv).rgb * 1.00
         + texture(uBloom1, uv).rgb * 0.65
         + texture(uBloom2, uv).rgb * 0.42;
  col += b * uBloomAmt;

  /* --- exposure + tonemap ------------------------------------------------ */
  col *= uExposure * (1.0 + 0.35*uElm);
  col = aces(col);

  /* --- lens vignette and a faint anamorphic streak ------------------------ */
  float vig = 1.0 - uVignette * smoothstep(0.15, 0.78, r2);
  col *= vig;

  /* --- sensor grain ------------------------------------------------------- */
  float g = hash13(vec3(gl_FragCoord.xy, floor(uTime*60.0))) - 0.5;
  col += g * uGrain * (0.35 + 0.65*(1.0 - dot(col, vec3(0.33))));

  /* --- disruption: video sync loss ---------------------------------------- */
  if(uDisrupt > 0.01){
    float band = step(0.5, hash11(floor(uv.y*90.0) + floor(uTime*24.0)));
    col = mix(col, col.bgr*1.4, band*uDisrupt*0.25);
  }

  col = pow(max(col, 0.0), vec3(1.0/2.2));   // -> sRGB
  fragColor = vec4(col, 1.0);
}`;

  global.SHADERS = { VERT, SCENE, BRIGHT, BLUR, COMPOSITE };
})(window);
