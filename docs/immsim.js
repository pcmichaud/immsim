// immsim2026 en JavaScript : transposition fidèle de immsim2026.proj() (facteurs calés fixes, admissions de
// l'étranger fixées) et de effective_workers_by_status(). Intrants : docs/intrants.json (prep/export_web.py).
// Convention : tableaux [durée][âge], durées 0 à 10 (10 = 10 ans et plus), âges 0 à 100 (100 = 100 ans et plus).
'use strict';

const NA = 101, ND = 11;

function zeros2() { return Array.from({ length: ND }, () => new Float64Array(NA)); }

// déplacement d'un an d'âge et de durée (bornes absorbantes), comme dans immsim2026.proj()
function shift2(src) {
  const out = zeros2();
  for (let d = 0; d < ND; d++) {
    const nd = Math.min(d + 1, ND - 1);
    for (let a = 0; a < NA; a++) out[nd][Math.min(a + 1, NA - 1)] += src[d][a];
  }
  return out;
}

function shift1(v) {                      // vieillissement d'un an d'un vecteur par âge
  const out = new Float64Array(NA);
  for (let a = 0; a < NA - 1; a++) out[a + 1] = v[a];
  out[NA - 1] += v[NA - 1];
  return out;
}

function sumAges(m) {                     // somme sur les durées -> vecteur par âge
  const out = new Float64Array(NA);
  for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) out[a] += m[d][a];
  return out;
}

function total(m) { let s = 0; for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) s += m[d][a]; return s; }

// facteur f tel que somme(stock * min(taux*f, 1)) / somme(stock) = cible (bissection, comme _calibrate_factor)
function calibrateFactor(stock, rates, target) {
  const tot = total(stock);
  if (tot <= 0 || target <= 0) return 0;
  const eff = f => { let s = 0; for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) s += stock[d][a] * Math.min(rates[a][d] * f, 1); return s / tot; };
  let lo = 0, hi = 1;
  while (eff(hi) < target && hi < 1e6) hi *= 2;
  for (let i = 0; i < 80; i++) { const mid = 0.5 * (lo + hi); if (eff(mid) < target) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

function applyRates(stock, rates, f) {
  const out = zeros2();
  for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) out[d][a] = stock[d][a] * Math.min(rates[a][d] * f, 1);
  return out;
}

function profile({ e0, he, r0, hr }, speed) {
  const w = new Float64Array(ND);
  const he_ = Math.max(he * speed, 1e-9), hr_ = Math.max(hr * speed, 1e-9);
  for (let d = 0; d < ND; d++) {
    const e = 1 - (1 - e0) * Math.max(1 - d / he_, 0);
    const r = 1 - (1 - r0) * Math.max(1 - d / hr_, 0);
    w[d] = e * r;
  }
  return w;
}

// p : { rp0, rp1, an1, etr, entrees, kappa, vitesse }
function simuler(I, p) {
  const T0 = I.annee_depart, T1 = I.annee_fin;
  const rpCap = t => (p.an1 <= 2027 || t >= p.an1) ? (t >= p.an1 ? p.rp1 : p.rp0)
    : p.rp0 + (p.rp1 - p.rp0) * (t - 2027) / (p.an1 - 2027);
  const kappa = p.kappa ?? 1, vitesse = p.vitesse ?? 1;
  const exitF = I.exit_factor * kappa;

  let pop = Float64Array.from(I.start_pop);
  let npr = I.start_npr.map(r => Float64Array.from(r));
  let prN = I.start_pr.map(r => Float64Array.from(r, x => x * 0.5));
  let prE = I.start_pr.map(r => Float64Array.from(r, x => x * 0.5));

  const res = { annees: [], rnp: [], pop: [], passages: [], departs: [], etranger: [], eff: [] };
  const wP = (t) => {
    const sp = t <= 2026 ? 1 : Math.max(1 / vitesse, 1 - (1 - 1 / vitesse) * (t - 2026) / 5);
    const P = I.profils, M = I.mix;
    const wt = profile(P.travail, sp), we = profile(P.etudiants, sp), wa = profile(P.asile, sp);
    const wN = new Float64Array(ND);
    for (let d = 0; d < ND; d++) wN[d] = M.travail * wt[d] + M.etudiants * we[d] + M.asile * wa[d];
    return { wN, wE: profile(P.rp, sp) };
  };
  const effective = (t) => {
    const { wN, wE } = wP(t);
    let s = 0;
    for (let a = 15; a <= 64; a++) {
      let v = pop[a];
      for (let d = 0; d < ND; d++) {
        const nl = npr[d][a] + prN[d][a];
        v += (wN[d] - 1) * nl + (wE[d] - 1) * prE[d][a];
      }
      s += v;
    }
    return s;
  };
  const record = (t, passages, departs, etranger) => {
    res.annees.push(t); res.rnp.push(total(npr)); res.pop.push(pop.reduce((x, y) => x + y, 0));
    res.passages.push(passages); res.departs.push(departs); res.etranger.push(etranger); res.eff.push(effective(t));
  };
  record(T0, 0, 0, 0);

  for (let t = T0 + 1; t <= T1; t++) {
    const mx = I.mx[t], ex = I.ex[t], ix = I.interpx[t], sh = I.share_ix[t];
    const p0 = pop, n0 = npr;
    const p1 = shift1(p0);
    let births = 0; for (let a = 15; a <= 45; a++) births += p0[a];
    births *= I.bx[t]; p1[0] = births;

    const exits = applyRates(n0, I.npr_exit_rates, exitF);
    const cap = rpCap(t);
    const target = Math.max(cap - p.etr, 0) / total(n0);
    const fA = calibrateFactor(n0, I.npr_apply_rates, target);
    const accepted = applyRates(n0, I.npr_apply_rates, fA);
    const remaining = zeros2();
    for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) remaining[d][a] = Math.max(n0[d][a] - exits[d][a] - accepted[d][a], 0);
    const n1 = shift2(remaining);
    for (let a = 0; a < NA; a++) n1[0][a] += p.entrees * I.npr_entry_profile[a];

    const nprAged = shift1(sumAges(n0));
    const departsAged = shift1(sumAges(exits));
    const accTot = total(accepted), exTot = total(exits);
    const prAbroad = Math.max(cap - accTot, 0);

    // RP récents : issus de RNP (prN) et arrivés de l'étranger (prE)
    const srcN = zeros2();
    for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) srcN[d][a] = prN[d][a] + accepted[d][a];
    const zN = shift2(srcN), zE = shift2(prE);
    for (let d = 0; d < ND; d++) for (let a = 0; a < NA; a++) {
      const surv = Math.min(Math.max(1 - mx[a] - ex[a], 0), 1);
      zN[d][a] *= surv; zE[d][a] *= surv;
    }
    for (let a = 0; a < NA; a++) zE[0][a] += prAbroad * sh[a];

    const pNew = new Float64Array(NA);
    for (let a = 0; a < NA; a++) {
      const nonNpr = Math.max(p1[a] - nprAged[a], 0);
      const deaths = a === 0 ? births * mx[0] : p1[a] * mx[a];
      pNew[a] = p1[a] - deaths - nonNpr * ex[a] + nonNpr * ix[a] + p.entrees * I.npr_entry_profile[a]
        - departsAged[a] + prAbroad * sh[a];
    }
    pop = pNew; npr = n1; prN = zN; prE = zE;
    record(t, accTot, exTot, prAbroad);
  }
  const y = res.eff.map((e, i) => e / res.pop[i]);
  res.pib = y.map(v => (v / y[0] - 1) * 100);
  res.part = res.annees.map((t, i) => {
    const c = I.canada.M1[t]; return c ? res.pop[i] / c * 100 : null;
  });
  return res;
}

if (typeof module !== 'undefined') module.exports = { simuler };
