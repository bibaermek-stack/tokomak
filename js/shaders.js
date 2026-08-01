/* ==========================================================================
   shaders.js — GLSL ES 3.00 sources for the tokamak renderer
   --------------------------------------------------------------------------
   Pass 1  SCENE      unified SDF machine + volumetric plasma raymarch
   Pass 2  BRIGHT     threshold / karis-average downsample
   Pass 3  BLUR       separable gaussian (run per mip, H then V)
   Pass 4  COMPOSITE  bloom combine, ACES tonemap, heat shimmer, grain

   The whole reactor — cryostat, toroidal field coils, poloidal field coils,
   central solenoid, vacuum vessel, first wall, divertor and ports — lives in
   a single signed distance field.  That is what lets the camera fly from
   outside the machine, through the cutaway, and come to rest inside the
   plasma ring as one continuous shot.
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

  const NOISE = `
/* pow(x, 2.0) with a possibly-negative base is undefined in GLSL and is
   compiled as exp2(n*log2(x)) even when it is defined — both a correctness
   and a speed problem in the inner loop.  Use plain multiplies. */
float sq(float x){ return x*x; }
float cube(float x){ return x*x*x; }
float pow5(float x){ float t = x*x; return t*t*x; }
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
float fbm2(vec3 p){
  return 0.5*vnoise(p) + 0.25*vnoise(p*2.03);
}
float fbm3(vec3 p){
  float a = 0.5, s = 0.0;
  for(int i=0;i<3;i++){ s += a*vnoise(p); p *= 2.03; a *= 0.5; }
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
uniform float uFovHalf;

/* --- plasma state ------------------------------------------------------ */
uniform float uRot;
uniform float uEmis;
uniform float uTemp;
uniform float uDens;
uniform float uInstab;
uniform float uElm;
uniform float uSaw;
uniform float uIgnite;
uniform float uDisrupt;
uniform float uNbi;
uniform float uRf;
uniform float uElong;
uniform float uHmode;

/* --- presentation ------------------------------------------------------ */
uniform int   uSteps;        // volumetric march steps
uniform int   uSceneSteps;   // SDF march steps
uniform int   uLights;       // ring-light samples
uniform int   uAOSteps;      // ambient occlusion taps (0 = off)
uniform float uDetail;       // 1 = extra plasma noise octave
uniform float uHighlight;    // material id to emphasise (0 = none)
uniform float uFieldLines;
uniform float uParticles;
uniform float uCut;          // 0..1 cutaway wedge opening
uniform float uCutCenter;    // wedge bearing [rad]
uniform float uExtLight;     // 0 = lit only by plasma, 1 = studio lit
uniform float uShowPlasma;
uniform float uShowCoils;
uniform float uHeatMap;      // first-wall heat flux false colour
uniform float uQdiv;         // normalised divertor heat flux

${NOISE}

const float PI  = 3.14159265359;
const float TAU = 6.28318530718;

/* =======================================================================
   Machine geometry.  1 scene unit ~ 2.6 m, so the cryostat spans ~28 m.
   ======================================================================= */
const float R0     = 2.02;    // plasma major radius
const float AMIN   = 0.94;    // plasma minor radius
const float DELTA  = 0.42;    // triangularity
const float RPMAX  = 3.30;    // bounding radius of emitting plasma

const float VV_CX  = 2.35;    // vacuum vessel D-section centre in r
const float VV_HW  = 1.32;    // -> inner wall r = 1.03, outer r = 3.67
const float VV_HH  = 2.36;
const float VV_RND = 0.80;
const float VV_T   = 0.13;

const float CS_R   = 0.60;    // central solenoid
const float CS_HH  = 3.15;

const float TF_CX  = 2.47;    // toroidal field coil D-loop
const float TF_HW  = 1.75;
const float TF_HH  = 3.05;
const float TF_RND = 0.95;
const float TF_T   = 0.26;
const float NTF    = 18.0;

const float CRYO_R  = 5.30;   // cryostat
const float CRYO_HH = 4.20;
const float CRYO_RND= 0.85;
const float CRYO_T  = 0.10;

/* material ids */
const float ID_CRYO = 1.0, ID_TF = 2.0, ID_PF = 3.0, ID_CS = 4.0,
            ID_VVO = 5.0, ID_WALL = 6.0, ID_DIV = 7.0, ID_PORT = 8.0,
            ID_BASE = 9.0;

/* ---------------------------------------------------------- SDF helpers */
float sdBox2(vec2 p, vec2 b){ vec2 d = abs(p)-b; return length(max(d,0.0)) + min(max(d.x,d.y),0.0); }
float sdRB2(vec2 p, vec2 b, float r){ return sdBox2(p, b-r) - r; }
float sdBox3(vec3 p, vec3 b){ vec3 d = abs(p)-b; return length(max(d,0.0)) + min(max(max(d.x,d.y),d.z),0.0); }
vec2  opU(vec2 a, vec2 b){ return a.x < b.x ? a : b; }

/* D-shaped poloidal cross-section, shared by the vessel and the coils. */
float dShape(float r, float y, float cx, float hw, float hh, float rnd, float tri){
  vec2 q = vec2(r - cx, y);
  q.x += tri * (q.y*q.y)/(hh*hh);          /* pull the crown inboard */
  return sdRB2(q, vec2(hw, hh), rnd);
}

/* ------------------------------------------------------------ components */
float sdCryostat(vec3 p, float r){
  vec2 q = vec2(r, abs(p.y));
  float d = length(max(q - vec2(CRYO_R-CRYO_RND, CRYO_HH-CRYO_RND), 0.0)) - CRYO_RND;
  d = max(d, max(r - CRYO_R, abs(p.y) - CRYO_HH));
  return abs(d) - CRYO_T;                  /* shell */
}

float sdCentralSolenoid(vec3 p, float r){
  float d = max(r - CS_R, abs(p.y) - CS_HH);
  /* six stacked modules: shallow grooves between them */
  float band = abs(fract(p.y/1.05 + 0.5) - 0.5) * 1.05;
  d = max(d, -(0.035 - band) * 0.6 - (r - CS_R + 0.05));
  return d;
}

float sdTF(vec3 p, float r, float phi){
  float sect = TAU/NTF;
  float pl = mod(phi + sect*0.5, sect) - sect*0.5;
  float arc = pl * max(r, 0.05);
  float loop = abs(dShape(r, p.y, TF_CX, TF_HW, TF_HH, TF_RND, 0.35)) - TF_T;
  /* coils are wedge shaped: they merge into a solid vault near the axis */
  float w = 0.20 + 0.34 * smoothstep(2.6, 0.5, r);
  return max(loop, abs(arc) - w);
}

float sdPF(vec3 p, float r){
  /* six poloidal field rings outside the toroidal cage */
  float d = 1e9;
  d = min(d, length(vec2(r-1.55, p.y-3.62)) - 0.26);
  d = min(d, length(vec2(r-4.35, p.y-2.62)) - 0.30);
  d = min(d, length(vec2(r-4.92, p.y-0.85)) - 0.26);
  d = min(d, length(vec2(r-4.92, p.y+0.85)) - 0.26);
  d = min(d, length(vec2(r-4.35, p.y+2.62)) - 0.30);
  d = min(d, length(vec2(r-1.80, p.y+3.62)) - 0.26);
  return d;
}

float sdVessel(vec3 p, float r, out float inner){
  float d = dShape(r, p.y, VV_CX, VV_HW, VV_HH, VV_RND, 0.30);
  inner = d;                                /* <0 means inside the chamber */
  return abs(d) - VV_T;
}

float sdDivertor(vec3 p, float r){
  /* cassette ring in the floor of the chamber */
  vec2 q = vec2(r - 2.05, p.y + 2.02);
  float d = sdRB2(q, vec2(0.95, 0.17), 0.07);
  float slope = abs(p.y + 2.10 + (r-2.05)*0.16) - 0.13;
  return max(d, slope - 0.10);
}

float sdPorts(vec3 p, float r, float phi){
  float sect = TAU/9.0;
  float pl = mod(phi + sect*0.5, sect) - sect*0.5;
  float arc = pl * max(r, 0.05);
  /* equatorial port ducts running out to the cryostat */
  float eq = sdBox3(vec3(arc, p.y - 0.15, r - 4.55), vec3(0.46, 0.62, 1.05));
  /* upper port ducts, angled outward */
  float up = sdBox3(vec3(arc, p.y - 3.15, r - 3.55), vec3(0.34, 0.70, 0.72));
  return min(eq, up);
}

float sdBase(vec3 p, float r){
  /* gravity supports and the machine plinth */
  float ring = max(abs(length(vec2(r-4.55, p.y+4.35)) - 0.55) - 0.16, p.y + 3.6);
  float plinth = max(abs(p.y + 4.66) - 0.18, r - 5.75);
  return min(ring, plinth);
}

/* Cutaway wedge: negative inside the removed sector. */
float cutWedge(vec3 p, float phi){
  if(uCut < 0.001) return 1e9;
  float hw = 1.02 * uCut;                   /* ~117 deg quadrant removed */
  float d = abs(mod(phi - uCutCenter + PI + TAU, TAU) - PI) - hw;
  /* only remove material, never the far side of the machine */
  return d * max(length(p.xz), 0.25);
}

/* ------------------------------------------------------------------ map */
vec2 mapScene(vec3 p){
  float r = length(p.xz);
  float phi = atan(p.z, p.x);
  float inner;

  vec2 res = vec2(sdCryostat(p, r), ID_CRYO);
  res = opU(res, vec2(sdBase(p, r), ID_BASE));
  if(uShowCoils > 0.5){
    res = opU(res, vec2(sdTF(p, r, phi), ID_TF));
    res = opU(res, vec2(sdPF(p, r), ID_PF));
  }
  res = opU(res, vec2(sdPorts(p, r, phi), ID_PORT));

  float dv = sdVessel(p, r, inner);
  res = opU(res, vec2(dv, inner < 0.0 ? ID_WALL : ID_VVO));
  res = opU(res, vec2(sdDivertor(p, r), ID_DIV));

  /* The wedge is subtracted from the machine, then the central solenoid is
     unioned back in whole — the reference cutaway keeps it intact, and it
     gives the shot something lit to look at through the opening. */
  float w = cutWedge(p, phi);
  if(w < 1e8) res.x = max(res.x, -w);
  res = opU(res, vec2(sdCentralSolenoid(p, r), ID_CS));
  return res;
}

vec3 calcNormal(vec3 p){
  const vec2 e = vec2(1.0, -1.0) * 0.0016;
  return normalize(
      e.xyy * mapScene(p + e.xyy).x + e.yyx * mapScene(p + e.yyx).x +
      e.yxy * mapScene(p + e.yxy).x + e.xxx * mapScene(p + e.xxx).x);
}

float calcAO(vec3 p, vec3 n){
  if(uAOSteps <= 0) return 1.0;
  float occ = 0.0, sca = 1.0;
  for(int i=0;i<5;i++){
    if(i >= uAOSteps) break;
    float h = 0.02 + 0.14*float(i);
    float d = mapScene(p + n*h).x;
    occ += (h - d) * sca;
    sca *= 0.72;
  }
  return clamp(1.0 - 1.4*occ, 0.05, 1.0);
}

/* =======================================================================
   Plasma
   ======================================================================= */
float fluxRho(vec3 p, out float theta){
  float r  = length(p.xz);
  float u  = r - R0;
  float v  = p.y / max(uElong, 0.3);
  theta    = atan(v, u);
  float shp = 1.0 + 0.135*DELTA*cos(theta) - 0.115*DELTA*cos(2.0*theta);
  return length(vec2(u, v)) / (AMIN * shp);
}

float divertorLegs(float r, float y){
  vec2 X  = vec2(R0 - 0.30, -1.28);
  vec2 T1 = vec2(R0 - 0.86, -2.00);
  vec2 T2 = vec2(R0 + 0.52, -2.06);
  vec2 q  = vec2(r, y);
  vec2 e1 = T1 - X, w1 = q - X;
  float d1 = length(w1 - e1*clamp(dot(w1,e1)/dot(e1,e1), 0.0, 1.0));
  vec2 e2 = T2 - X, w2 = q - X;
  float d2 = length(w2 - e2*clamp(dot(w2,e2)/dot(e2,e2), 0.0, 1.0));
  return min(d1*1.35, d2);
}

/* Conservative lower bound on the distance from p to anything that emits.
   fluxRho measures in a metric that compresses the vertical direction by
   the elongation, so the flux-space distance never overstates the true
   Euclidean one — which is exactly what a sphere trace needs.           */
float plasmaSkip(vec3 p, out float rho, out float th, out float dleg){
  float r = length(p.xz);
  rho  = fluxRho(p, th);
  dleg = divertorLegs(r, p.y);
  return min((rho - 1.34) * AMIN * 0.9, dleg - 0.42);
}

vec4 plasmaAt(vec3 p, float rho, float th, float dleg){
  float r   = length(p.xz);
  float phi = atan(p.z, p.x);

  /* the plasma is cut away with the machine */
  if(uCut > 0.001){
    float hw = 1.02 * uCut;
    if(abs(mod(phi - uCutCenter + PI + TAU, TAU) - PI) < hw) return vec4(0.0);
  }

  float outb = smoothstep(-0.2, 0.9, cos(th));
  /* Drift-wave turbulence as a short sum of incommensurate sines.  Value
     noise looks marginally better but costs sixteen hashes per octave, and
     this runs at every step of the volumetric march — by far the hottest
     loop in the renderer. */
  /* Amplitudes are kept below the pedestal width: perturbing rho by more
     than the shell thickness smears the ring into a haze and the limb
     brightening that defines the whole image disappears. */
  float turb = (sin(phi*3.1 + th*2.2 - uRot*1.4)*0.34
              + sin(phi*5.7 - th*3.9 - uTime*0.55)*0.21
              + sin(th*7.3 - phi*2.1 + uTime*0.90)*0.13) * 0.55;
  float fil = 0.0;
  if(uDetail > 0.5)
    fil = (sin(phi*9.0 - uRot*2.1 + th*6.5)*0.5 +
           sin(th*11.0 + phi*4.0 - uTime*1.7)*0.3) * outb * 0.47;

  float amp = uInstab * (0.055 + 0.14*uDisrupt);
  float rhoT = rho * (1.0 + amp*turb + amp*0.9*fil*smoothstep(0.55,1.05,rho));

  /* A narrow emission shell is what produces limb brightening: the ring is
     bright because a tangential ray travels far inside it, while a ray
     crossing it head-on barely clips it.  Widening this flattens the shot. */
  /* Level of detail for the emission shell.  At low step counts a 5 cm
     shell cannot be resolved along a 6 m chord, so it is broadened and its
     amplitude divided by the same factor — the line integral through it,
     which is what sets the ring brightness, is preserved.               */
  float widen = uDetail > 0.5 ? 1.0 : 1.9;
  float pedW = mix(0.082, 0.048, uHmode) * widen;
  float edge = exp(-sq((rhoT - 0.978)/pedW)) * 1.45 / widen;
  float core = exp(-rhoT*rhoT*1.65) * (0.05 + 0.15*uIgnite + 0.40*uSaw);
  float sol  = exp(-sq((rhoT - 1.09)/0.085)) * 0.18;
  float bound = 1.0 - smoothstep(1.06, 1.30, rhoT);
  edge *= bound; sol *= bound; core *= bound;
  /* brightest near the X-point, faintest at the crown */
  edge *= 0.30 + 0.95*(0.5 - 0.5*sin(th));
  /* the high-field (inboard) side radiates far less than the outboard side */
  edge *= 0.34 + 0.80*smoothstep(-0.95, 0.55, cos(th));

  float div  = exp(-sq(dleg/0.135)) * (1.0 + 3.2*uElm);
  float strike = exp(-sq((p.y + 2.04)/0.16)) *
                 exp(-sq((abs(r - R0) - 0.62)/0.42)) * 1.5;

  float ripple = 1.0 + 0.035*cos(phi*18.0 + 0.4) * smoothstep(0.6, 1.2, rho);
  float kill = 1.0 - 0.55*uDisrupt*step(0.5, hash11(floor(uTime*22.0)+floor(phi*3.0)));
  edge *= kill; sol *= kill; core *= kill;

  vec3 cCore = mix(vec3(0.35,0.52,1.00), vec3(0.72,0.82,1.00), uTemp);
  vec3 cEdge = mix(vec3(0.85,0.16,0.92), vec3(1.00,0.32,0.86), uTemp);
  vec3 cDiv  = vec3(1.00,0.62,0.92);
  float vt = smoothstep(-1.4, 1.9, p.y);
  cEdge = mix(cEdge, vec3(0.50,0.42,1.00), vt*0.55);

  vec3 col = cCore*core*1.25 + cEdge*(edge + sol)*ripple;
  col += cDiv * (div*4.50 + strike*1.80);

  if(uElm > 0.001){
    float elmF = fbm3(vec3(phi*7.0, th*3.0, uTime*6.0));
    float elmShell = exp(-sq((rhoT - 1.03)/0.15)) * outb;
    col += vec3(1.0,0.55,0.85) * uElm * elmShell * (0.5 + elmF) * 3.2;
  }
  if(uNbi > 0.01){
    for(int b=0;b<2;b++){
      float off = (b==0) ? 0.0 : 2.2;
      vec3 bo = vec3(cos(off)*3.2, 0.16, sin(off)*3.2);
      vec3 bd = normalize(vec3(-cos(off+1.05), -0.02, -sin(off+1.05)));
      vec3 w  = p - bo;
      float h = clamp(dot(w,bd), 0.0, 4.4);
      float d = length(w - bd*h);
      col += vec3(0.55,0.85,1.00) * uNbi * exp(-sq(d/0.13)) * exp(-h*0.55) * 1.5;
    }
  }
  if(uRf > 0.01){
    float res = exp(-sq((r - (R0 - 0.42))/0.085)) *
                exp(-sq(p.y/1.05)) * step(rho, 1.0);
    col += vec3(0.45,1.00,0.95)*uRf*res*0.55*(0.7+0.3*sin(uTime*30.0));
  }
  if(uFieldLines > 0.01){
    float hel = cos(3.0*th - 2.0*phi + uRot*0.9);
    float band = exp(-sq((rho - 0.72)/0.05));
    col += vec3(0.40,0.95,1.00)*uFieldLines*band*smoothstep(0.90,1.0,hel)*2.2;
  }
  if(uParticles > 0.01){
    float sp = hash11(floor(phi*22.0) + floor(th*9.0)*17.0);
    float trail = fract(phi*3.5 + uTime*(0.6+sp*1.8) + sp*10.0);
    float band = exp(-sq((rho - (0.30+sp*0.65))/0.035));
    col += vec3(0.85,0.95,1.00)*uParticles*band*
           cube(max(0.0,1.0-trail*6.0))*1.6;
  }

  float ext = (edge + sol + core*0.5) * (0.03 + 0.09*uDens);
  /* anatomy highlight: id 20 is the plasma itself */
  if(uHighlight > 0.5) col *= (abs(uHighlight - 20.0) < 0.1) ? 1.7 : 0.30;
  return vec4(col * uEmis * 0.60, ext);
}

/* =======================================================================
   Materials and lighting
   ======================================================================= */
void wallMaterial(float u, float v, float scaleU, float scaleV,
                  out vec3 alb, out float rough, out float bump, out float ao)
{
  vec2 g = vec2(u*scaleU, v*scaleV);
  vec2 cell = floor(g);
  vec2 f = fract(g);
  float seedT = hash13(vec3(cell, 3.7));
  float gx = min(f.x, 1.0-f.x), gy = min(f.y, 1.0-f.y);
  float gap = smoothstep(0.0, 0.035, gx) * smoothstep(0.0, 0.030, gy);
  float bolt = 0.0;
  for(int i=0;i<4;i++){
    vec2 bp = vec2(float(i&1), float((i>>1)&1));
    bp = mix(vec2(0.13,0.13), vec2(0.87,0.87), bp);
    bolt = max(bolt, 1.0 - smoothstep(0.030, 0.048, length(f-bp)));
  }
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

/* Engineering surface for the machine exterior: panel seams and plate wear */
void panelMaterial(vec3 p, float scale, out float seam, out float grime){
  vec3 g = p * scale;
  vec3 f = abs(fract(g) - 0.5);
  float m = min(min(f.x, f.y), f.z);
  seam = 1.0 - smoothstep(0.0, 0.035, m);
  grime = fbm3(p*1.6);
}

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
  float F = 0.04 + 0.96*pow5(1.0 - max(dot(h,v),0.0));
  vec3 spec = vec3(D*G*F/(4.0*ndv*ndl + 1e-4));
  vec3 diff = alb*(1.0-metal)/PI;
  return (diff + spec) * lc * ndl / (0.55 + d2*0.75);
}

/* Directional light, for the studio rig used on the exterior shot. */
vec3 lightDir(vec3 n, vec3 v, vec3 ld, vec3 lc, float rough, float metal, vec3 alb){
  float ndl = max(dot(n, ld), 0.0);
  if(ndl <= 0.0) return vec3(0.0);
  vec3 h = normalize(ld + v);
  float ndh = max(dot(n, h), 0.0);
  float ndv = max(dot(n, v), 1e-3);
  float a = max(rough*rough, 0.004), a2 = a*a;
  float dnm = ndh*ndh*(a2-1.0)+1.0;
  float D = a2/(PI*dnm*dnm);
  float k = a*0.5;
  float G = (ndl/(ndl*(1.0-k)+k)) * (ndv/(ndv*(1.0-k)+k));
  float F = 0.04 + 0.96*pow5(1.0 - max(dot(h,v),0.0));
  return (alb*(1.0-metal)/PI + vec3(D*G*F/(4.0*ndv*ndl+1e-4))) * lc * ndl;
}

vec3 ringLighting(vec3 p, vec3 n, vec3 v, float rough, float metal, vec3 alb){
  vec3 acc = vec3(0.0);
  /* far from the torus the ring reads as a single soft source, so a coarse
     sampling is indistinguishable and much cheaper */
  int nl = dot(p, p) > 20.0 ? min(uLights, 5) : uLights;
  float N = float(nl);
  /* Rotate the sample set by a per-pixel offset.  Without this a coarse
     ring shows as scalloped banding on the wall; with it the error becomes
     noise, which the bloom and grain absorb completely. */
  float jit = hash13(vec3(gl_FragCoord.xy, 7.3));
  vec3 cMid = mix(vec3(0.90,0.18,0.95), vec3(1.00,0.35,0.88), uTemp);
  vec3 cDiv = vec3(1.00,0.58,0.90);
  vec3 cTop = vec3(0.42,0.46,1.00);
  float e = uEmis * (1.0 + 1.6*uElm) * uShowPlasma;
  for(int i=0;i<48;i++){
    if(i >= nl) break;
    float a = (float(i)+jit)/N * TAU + uRot*0.03;
    vec2 cs = vec2(cos(a), sin(a));
    float wob = 0.10*sin(a*4.0 + uTime*1.3);
    /* one merged mid-plane ring instead of an inboard/outboard pair — at
       these radii the two are indistinguishable and this is the hottest
       per-pixel loop on the surface path */
    acc += lightPoint(p,n,v, vec3(cs.x*(R0+0.18), 0.06+wob, cs.y*(R0+0.18)),
                      cMid*e*1.55, rough, metal, alb);
    acc += lightPoint(p,n,v, vec3(cs.x*1.92, -1.96, cs.y*1.92),
                      cDiv*e*2.10, rough, metal, alb);
    acc += lightPoint(p,n,v, vec3(cs.x*2.25, 1.30, cs.y*2.25),
                      cTop*e*0.22, rough, metal, alb);
  }
  return acc * (1.35/N);
}

/* False-colour overlay for the first wall heat load. */
vec3 heatRamp(float t){
  t = clamp(t, 0.0, 1.0);
  vec3 c = mix(vec3(0.03,0.05,0.30), vec3(0.10,0.55,0.85), smoothstep(0.0,0.35,t));
  c = mix(c, vec3(0.25,0.90,0.55), smoothstep(0.30,0.58,t));
  c = mix(c, vec3(1.00,0.80,0.20), smoothstep(0.55,0.80,t));
  c = mix(c, vec3(1.00,0.25,0.15), smoothstep(0.78,1.00,t));
  return c;
}

/* =======================================================================
   Main
   ======================================================================= */
void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5*uRes) / uRes.y;
  float rr = length(uv);
  float ang = rr * uFovHalf;
  vec2 dirxy = rr > 1e-5 ? uv/rr : vec2(0.0);
  vec3 rd = uCamMat * vec3(dirxy*sin(ang), -cos(ang));
  vec3 ro = uCamPos;

  /* ---------------- SDF march ---------------------------------------
     The whole machine fits inside r < 6.0, |y| < 5.0.  Clipping the ray to
     that box first means rays that miss it cost nothing at all, which is
     most of the frame on the exterior shots.                            */
  float far = 26.0;
  float t = 0.02, tEnd = far;
  {
    const float BR = 6.0, BH = 5.0;
    float t0 = 0.0, t1 = far;
    float qa = dot(rd.xz, rd.xz);
    if(qa > 1e-6){
      float qb = 2.0*dot(ro.xz, rd.xz);
      float qc = dot(ro.xz, ro.xz) - BR*BR;
      float disc = qb*qb - 4.0*qa*qc;
      if(disc <= 0.0){ t0 = 1.0; t1 = 0.0; }
      else {
        float sd = sqrt(disc);
        t0 = max(t0, (-qb - sd)/(2.0*qa));
        t1 = min(t1, (-qb + sd)/(2.0*qa));
      }
    } else if(dot(ro.xz, ro.xz) > BR*BR){ t0 = 1.0; t1 = 0.0; }
    if(abs(rd.y) > 1e-6){
      float ta = (-BH - ro.y)/rd.y, tb = (BH - ro.y)/rd.y;
      t0 = max(t0, min(ta, tb));
      t1 = min(t1, max(ta, tb));
    } else if(abs(ro.y) > BH){ t0 = 1.0; t1 = 0.0; }
    t = max(0.02, t0);
    tEnd = t1;
  }

  float id = 0.0;
  bool hit = false;
  if(t < tEnd){
    for(int i=0;i<256;i++){
      if(i >= uSceneSteps) break;
      vec3 p = ro + rd*t;
      vec2 h = mapScene(p);
      if(h.x < 0.0009*t + 0.0006){ id = h.y; hit = true; break; }
      t += h.x * 0.86;
      if(t > tEnd) break;
    }
  }

  vec3 surf = vec3(0.0);
  float tS = hit ? t : far;

  if(hit){
    vec3 p = ro + rd*tS;
    vec3 n = calcNormal(p);
    vec3 v = -rd;
    float r = length(p.xz);
    float phi = atan(p.z, p.x);
    float ao = calcAO(p, n);

    vec3 alb = vec3(0.10);
    float rough = 0.45, metal = 0.7;
    vec3 emis = vec3(0.0);

    float seam, grime;
    panelMaterial(p, 2.2, seam, grime);

    if(id == ID_WALL){
      /* beryllium first wall — the surface seen from inside the chamber */
      float bump, hu, hv, tao;
      vec3 t1 = normalize(cross(vec3(0,1,0), n)), t2 = cross(n, t1);
      float u = (phi+PI)/TAU;
      float vv = (p.y + VV_HH) / (2.0*VV_HH);
      /* Blanket modules are a roughly constant physical width, so the
         inboard column carries far fewer of them than the outboard wall. */
      float su = mix(15.0, 48.0, smoothstep(1.0, 3.6, r));
      wallMaterial(u, vv, su, 8.0, alb, rough, bump, tao);
      wallMaterial(u+0.0011, vv, su, 8.0, alb, rough, hu, tao);
      wallMaterial(u, vv+0.0011, su, 8.0, alb, rough, hv, tao);
      n = normalize(n - (hu-bump)*0.7*t1 - (hv-bump)*0.7*t2);
      metal = 0.25;
      ao *= tao;
      if(uHeatMap > 0.5){
        float load = uQdiv * (0.35 + 0.65*exp(-sq((p.y+1.6)/1.3)))
                            * (0.6 + 0.4*smoothstep(1.2, 3.4, r));
        alb = heatRamp(load);
        metal = 0.0; rough = 0.75;
        emis += alb * 0.35;
      }
    } else if(id == ID_DIV){
      alb = vec3(0.16,0.14,0.13);
      rough = 0.20; metal = 0.75;
      float strike = exp(-sq((r-(R0+0.52))/0.26)) + 0.7*exp(-sq((r-(R0-0.86))/0.22));
      emis += vec3(1.0,0.42,0.16) * strike * uEmis * uShowPlasma * (0.10 + 0.55*uElm);
      if(uHeatMap > 0.5){ alb = heatRamp(uQdiv*(0.5+0.9*strike)); metal=0.0; emis += alb*0.5; }
    } else if(id == ID_VVO){
      /* vessel exterior: bare stainless steel, ribbed */
      alb = vec3(0.42,0.45,0.50) * (0.85 + 0.3*grime);
      rough = mix(0.32, 0.5, grime);
      metal = 0.9;
      alb *= (1.0 - 0.35*seam);
    } else if(id == ID_TF){
      /* toroidal field coil cases: dark blue-grey steel */
      alb = vec3(0.20,0.25,0.33) * (0.8 + 0.35*grime);
      rough = 0.38; metal = 0.85;
      float wind = 0.5 + 0.5*sin(dShape(r, p.y, TF_CX, TF_HW, TF_HH, TF_RND, 0.35)*140.0);
      alb *= (0.82 + 0.30*wind);
      alb *= (1.0 - 0.4*seam);
    } else if(id == ID_PF){
      /* poloidal field coils, copper conductor in the reference cutaway */
      alb = vec3(0.62,0.32,0.22) * (0.85 + 0.3*grime);
      rough = 0.34; metal = 0.85;
      float turn = 0.5 + 0.5*sin(atan(p.y-0.0, r-4.6)*90.0);
      alb *= (0.85 + 0.28*turn);
    } else if(id == ID_CS){
      alb = vec3(0.50,0.52,0.56) * (0.85 + 0.25*grime);
      rough = 0.30; metal = 0.9;
      float band = smoothstep(0.02, 0.05, abs(fract(p.y/1.05 + 0.5) - 0.5)*1.05);
      alb *= (0.55 + 0.5*band);
    } else if(id == ID_PORT){
      alb = vec3(0.30,0.34,0.40) * (0.85 + 0.3*grime);
      rough = 0.40; metal = 0.85;
      alb *= (1.0 - 0.35*seam);
    } else if(id == ID_BASE){
      alb = vec3(0.16,0.17,0.19);
      rough = 0.62; metal = 0.4;
    } else {
      /* cryostat: pale painted shell, with stiffening ribs */
      alb = vec3(0.27,0.30,0.35) * (0.86 + 0.26*grime);
      rough = 0.46; metal = 0.30;
      float hoop = smoothstep(0.04, 0.11, abs(fract(p.y*0.9) - 0.5));
      float stave = smoothstep(0.05, 0.13, abs(fract(phi*(18.0/TAU)) - 0.5));
      alb *= (0.70 + 0.24*hoop + 0.16*stave);
      alb *= (1.0 - 0.32*seam);
    }

    /* --- anatomy highlight: lift the named part, mute everything else -- */
    if(uHighlight > 0.5){
      if(abs(id - uHighlight) < 0.1){
        float rim = sq(1.0 - max(dot(n, v), 0.0));
        alb *= 1.55;
        emis += vec3(0.30,0.80,1.00) * (0.10 + 0.75*rim);
      } else {
        alb *= 0.30;
      }
    }

    /* --- plasma ring lights (dominant once inside) -------------------- */
    surf = ringLighting(p, n, v, rough, metal, alb) * ao;

    /* --- studio rig, faded out as the camera enters the machine ------- */
    if(uExtLight > 0.001){
      vec3 key  = normalize(vec3(0.45, 0.80, 0.55));
      vec3 fill = normalize(vec3(-0.70, 0.15, 0.35));
      vec3 rim  = normalize(vec3(-0.25, -0.45, -0.85));
      vec3 L = vec3(0.0);
      L += lightDir(n, v, key,  vec3(1.00,0.98,0.95)*2.10, rough, metal, alb);
      L += lightDir(n, v, fill, vec3(0.34,0.52,0.95)*1.05, rough, metal, alb);
      L += lightDir(n, v, rim,  vec3(0.75,0.45,1.00)*1.15, rough, metal, alb);
      /* hemispherical ambient from the dark laboratory */
      vec3 amb = mix(vec3(0.030,0.036,0.058), vec3(0.075,0.095,0.150), n.y*0.5+0.5);
      L += alb * amb * 3.4;
      surf += L * ao * uExtLight;
    }

    /* --- plasma bounce light ----------------------------------------- */
    vec3 pamb = mix(vec3(0.42,0.16,0.55), vec3(0.07,0.09,0.30),
                    smoothstep(-1.8,1.6,p.y));
    surf += alb * pamb * uEmis * uShowPlasma * 0.045 * ao;
    surf += emis;

    /* --- arcing on the first wall during transients ------------------- */
    if(id == ID_WALL){
      float arcSeed = hash11(floor(uTime*13.0)*7.7 + floor(phi*11.0));
      if(arcSeed > 0.985 - 0.05*uElm - 0.35*uDisrupt){
        float ay = hash11(arcSeed*53.0)*2.0-1.0;
        float arc = exp(-sq((p.y-ay*1.6)/0.05)) *
                    exp(-sq((fract(phi*11.0)-0.5)/0.09));
        surf += vec3(0.75,0.90,1.00)*arc*7.0;
      }
    }
  }

  /* ---------------- volumetric plasma ------------------------------- */
  float tmax = min(tS, far);
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
  /* The emitting shell is only a few centimetres thick, so the step is tied
     to that rather than to the chord length; empty space is crossed by
     sphere tracing on plasmaSkip instead of by brute force.              */
  /* Step bounds are derived from the budget, not fixed.  If the march runs
     out of iterations before tmax it stops accumulating extinction, the
     transmittance stays high, and the bright wall behind the plasma bleeds
     through — which reads as a washed-out frame rather than as noise.
     Tying both bounds to span/steps guarantees the ray always completes. */
  float span = tmax - tnear;
  float avg  = span / float(uSteps);
  float dtFine   = clamp(avg * 0.55, 0.020, 0.090);
  float dtCoarse = max(avg * 2.2, 0.12);
  float dt = dtFine;
  float jitter = hash13(vec3(gl_FragCoord.xy, floor(uTime*60.0)));
  float tv = tnear + dt*jitter;
  if(uShowPlasma > 0.5){
    for(int i=0;i<256;i++){
      if(i >= uSteps || tv >= tmax || trans < 0.004) break;
      vec3 p = ro + rd*tv;
      float rho, th, dleg;
      float empty = plasmaSkip(p, rho, th, dleg);
      if(empty > 0.0){ tv += max(dt, empty); continue; }
      /* Dense sampling only across the thin emission shell and the divertor
         legs; the core glow is broad and smooth, so it is integrated with
         long strides.  Riemann sum uses the step actually taken.          */
      float dShell = (abs(rho - 0.99) - 0.15) * AMIN * 0.9;
      float step = clamp(min(dShell, dleg - 0.20) * 0.6, dtFine, dtCoarse);
      vec4 e = plasmaAt(p, rho, th, dleg);
      acc   += e.rgb * trans * step;
      trans *= exp(-e.a * step * 0.55);
      tv += step;
    }
  }

  vec3 col = surf * trans + acc;

  /* ---------------- laboratory haze and floating dust ---------------- */
  if(uExtLight > 0.001 && !hit){
    /* faint volumetric backdrop so the machine sits in a space */
    float g = max(0.0, 1.0 - length(uv)*1.15);
    col += vec3(0.012,0.016,0.032) * g * uExtLight;
  }
  if(uExtLight > 0.001){
    /* motes drifting in the beam of the key light */
    vec2 dp = uv*3.0;
    float m = 0.0;
    for(int i=0;i<3;i++){
      float fi = float(i);
      vec2 q = dp*(1.0+fi*0.6) + vec2(uTime*(0.02+fi*0.012), -uTime*0.014 + fi*3.1);
      vec2 c = floor(q);
      float hs = hash13(vec3(c, fi*7.0));
      if(hs > 0.982){
        vec2 f = fract(q) - 0.5 - 0.3*vec2(hash11(hs*13.0)-0.5, hash11(hs*29.0)-0.5);
        m += exp(-dot(f,f)*90.0) * (0.4 + 0.6*hash11(hs*53.0));
      }
    }
    col += vec3(0.55,0.72,1.0) * m * 0.5 * uExtLight;
  }

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
uniform vec2 uDir;
void main(){
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
uniform float uFade;      // cinematic dip-to-black

${NOISE}

vec3 aces(vec3 x){
  const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main(){
  vec2 uv = vUv;
  vec2 cc = uv - 0.5;
  float r2 = dot(cc, cc);

  float heat = texture(uScene, uv).a;
  vec2 wob = vec2(
    fbm3(vec3(uv*14.0, uTime*1.7)) - 0.5,
    fbm3(vec3(uv*14.0 + 31.7, uTime*1.9)) - 0.5);
  uv += wob * uShimmer * (0.25 + heat) * (1.0 + 3.0*uElm) * 0.006;

  float ca = (0.0016 + 0.010*r2) * (1.0 + 2.0*uDisrupt);
  vec3 col;
  col.r = texture(uScene, uv - cc*ca).r;
  col.g = texture(uScene, uv).g;
  col.b = texture(uScene, uv + cc*ca).b;

  vec3 b = texture(uBloom0, uv).rgb * 1.00
         + texture(uBloom1, uv).rgb * 0.65
         + texture(uBloom2, uv).rgb * 0.42;
  col += b * uBloomAmt;

  col *= uExposure * (1.0 + 0.35*uElm);
  col = aces(col);

  col *= 1.0 - uVignette * smoothstep(0.15, 0.78, r2);

  float g = hash13(vec3(gl_FragCoord.xy, floor(uTime*60.0))) - 0.5;
  col += g * uGrain * (0.35 + 0.65*(1.0 - dot(col, vec3(0.33))));

  if(uDisrupt > 0.01){
    float band = step(0.5, hash11(floor(uv.y*90.0) + floor(uTime*24.0)));
    col = mix(col, col.bgr*1.4, band*uDisrupt*0.25);
  }

  col = pow(max(col, 0.0), vec3(1.0/2.2));
  col *= (1.0 - clamp(uFade, 0.0, 1.0));
  fragColor = vec4(col, 1.0);
}`;

  global.SHADERS = { VERT, SCENE, BRIGHT, BLUR, COMPOSITE };
})(window);
