#pragma once

#include <cmath>

// A diagnostic admission rule, not a guarantee of out-of-sample accuracy.
// Uses the existing three-observation ordinary-GP minimum; no additional
// count milestone or fit-count hysteresis is introduced. Both schedulers
// evaluate this same rule at their normal decision opportunities.
inline bool readiness_certification_supported(
  int n_training, int n_grid, int full_loo_n,
  double full_loo_max, double full_loo_rms, int gate_loo_n,
  double gate_loo_max, double gate_loo_rms,
  double uncertainty_max, double uncertainty_rms,
  double uncertainty_max_limit, double uncertainty_rms_limit,
  double loo_max_limit, double loo_rms_limit)
{
  const auto finite_nonnegative = [](double x) {
    return std::isfinite(x) && x >= 0.0;
  };
  const auto positive = [](double x) {
    return std::isfinite(x) && x > 0.0;
  };
  return n_training >= 3 && n_grid > 0 &&
    full_loo_n == n_training && gate_loo_n > 0 && gate_loo_n <= full_loo_n &&
    finite_nonnegative(full_loo_max) && finite_nonnegative(full_loo_rms) &&
    finite_nonnegative(gate_loo_max) && finite_nonnegative(gate_loo_rms) &&
    finite_nonnegative(uncertainty_max) && finite_nonnegative(uncertainty_rms) &&
    positive(uncertainty_max_limit) && positive(uncertainty_rms_limit) &&
    positive(loo_max_limit) && positive(loo_rms_limit) &&
    uncertainty_max <= uncertainty_max_limit &&
    uncertainty_rms <= uncertainty_rms_limit &&
    gate_loo_max <= loo_max_limit && gate_loo_rms <= loo_rms_limit;
}
