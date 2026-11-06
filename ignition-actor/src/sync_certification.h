#pragma once

#include <algorithm>
#include <cmath>

// Zero means ordinary acquisition. This changes admissions, not the stopping
// test: a completed probe must still pass the existing prospective-error rule.
// A Sync decision is made only at a drained batch barrier, using fresh metrics.
inline int sync_certification_probe_batch(
  bool enabled, bool synchronous, bool probe_stop_enabled,
  bool diagnostics_ready, int unfinished, int n_training, int warmup,
  int scored, int required, double rms, double tolerance, int capacity)
{
  if (!enabled || !synchronous || !probe_stop_enabled || !diagnostics_ready ||
      unfinished != 0 || n_training < warmup || scored < 0 ||
      required <= scored || capacity <= 0 || !(tolerance > 0.0))
    return 0;
  // Do not discard failed evidence or repeatedly add probes just to dilute it.
  // Return to ordinary learning if the already scored probes miss the target.
  if (scored > 0 && (!std::isfinite(rms) || rms > tolerance))
    return 0;
  return std::min(capacity, required - scored);
}
