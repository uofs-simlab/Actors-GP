#pragma once

#include <algorithm>
#include <cmath>

// Early GP brackets (asynchronous mode, --gp-async-early-brackets).
//
// A GP-seeded bracket [median*exp(-k sigma), median*exp(+k sigma)] replaces
// the global [Av_lo, Av_hi] only when it is EXPECTED TO COST LESS. Both GP
// ends are verified by solves (worker.cc), so a wrong bracket costs solves,
// never correctness; the only question is cost.
//
// Cost model: the bisection the worker would run if the GP median were the
// exact threshold is simulated step by step (same tolerance rule), counting
// igniting and quenched solves separately. With the early-exit solver a
// quenched solve ends within seconds while an igniting one runs to the end,
// so cost = ignited + rho * quenched with rho the measured quench/ignite
// time ratio (rho = 1 recovers plain solve counting). The lower-end
// verification of a GP bracket is a quench and therefore nearly free; the
// saving comes from the shorter descent of igniting midpoints. Pure
// functions only.

struct early_bracket_t
{
  bool   available = false;
  double lo = 0.0;
  double hi = 0.0;
};

struct bisection_cost_t
{
  int ignited = 0;
  int quenched = 0;
  double cost(const double rho) const { return ignited + rho * quenched; }
};

inline early_bracket_t gp_bracket_from_forecast(const double median,
                                                const double sigma,
                                                const double k,
                                                const double Av_lo,
                                                const double Av_hi)
{
  early_bracket_t out;
  if (!std::isfinite(median) || !std::isfinite(sigma) || median <= 0.0 ||
      sigma < 0.0 || !(k > 0.0) || !(Av_hi > Av_lo))
    return out;
  const double spread = std::exp(k * sigma);
  double       lo     = median / spread;
  double       hi     = median * spread;
  if (!std::isfinite(lo) || !std::isfinite(hi))
    return out;
  lo = std::max(lo, Av_lo);
  hi = std::min(hi, Av_hi);
  if (!(hi > lo))
    return out;
  out.available = true;
  out.lo        = lo;
  out.hi        = hi;
  return out;
}

// The worker's search on [lo, hi] when the threshold is exactly `a_star`
// (a_star within [lo, hi]): the upper end is tested first (ignites), the
// lower end next when `verify_lower` (quenches), then midpoints until the
// bracket is narrower than max(abs_tol, rel_tol * mid), as in run_bisection.
inline bisection_cost_t simulate_bisection(double lo, double hi,
                                           const double a_star,
                                           const double abs_tol,
                                           const double rel_tol,
                                           const bool verify_lower)
{
  bisection_cost_t c;
  if (!(hi > lo) || !(a_star > 0.0))
    return c;
  ++c.ignited; // upper end: hi >= a_star
  if (verify_lower)
    {
      if (lo < a_star) ++c.quenched; else ++c.ignited;
    }
  for (int i = 0; i < 64; ++i)
    {
      const double mid = 0.5 * (lo + hi);
      const double tol = std::max(abs_tol, rel_tol * mid);
      if (!(hi - lo >= tol) || !(tol > 0.0))
        break;
      if (mid >= a_star) { ++c.ignited; hi = mid; }
      else               { ++c.quenched; lo = mid; }
    }
  return c;
}

inline double global_bracket_cost(const double Av_lo, const double Av_hi,
                                  const double estimate, const double abs_tol,
                                  const double rel_tol, const double rho)
{
  return simulate_bisection(Av_lo, Av_hi, estimate, abs_tol, rel_tol,
                            Av_lo > 0.0).cost(rho);
}

inline double gp_bracket_cost(const early_bracket_t &bracket,
                              const double estimate, const double Av_lo,
                              const double abs_tol, const double rel_tol,
                              const double rho)
{
  return simulate_bisection(bracket.lo, bracket.hi, estimate, abs_tol, rel_tol,
                            bracket.lo > Av_lo).cost(rho);
}

inline bool gp_bracket_preferred(const early_bracket_t &bracket,
                                 const double median, const double Av_lo,
                                 const double Av_hi, const double abs_tol,
                                 const double rel_tol, const double rho)
{
  if (!bracket.available || !(median > 0.0) || !(rho > 0.0))
    return false;
  return gp_bracket_cost(bracket, median, Av_lo, abs_tol, rel_tol, rho) <
         global_bracket_cost(Av_lo, Av_hi, median, abs_tol, rel_tol, rho);
}
