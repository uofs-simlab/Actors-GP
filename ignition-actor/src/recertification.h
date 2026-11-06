#pragma once

#include <cmath>

// Bounded re-certification (--gp-probe-max-recertifications), both policies.
//
// A certification set is complete once `required` prospective probes have been
// scored. If its RMS relative error misses the target, the run used to have no
// way left to stop by accuracy: neither policy admits probes once
// scored >= required, and failed evidence is deliberately never diluted by
// adding probes to the same set. Instead the failed set is archived and a
// FRESH set of prospective probes may certify the model, at most
// `max_recertifications` times. Probes admitted for an earlier set are scored
// into the archive, never into the new set. With no failed set none of this is
// reached. Pure functions only.

inline bool certification_set_failed(const int scored, const int required,
                                     const double rms, const double tolerance)
{
  if (required <= 0 || scored < required || !(tolerance > 0.0))
    return false;
  return !std::isfinite(rms) || rms > tolerance;
}

inline bool recertification_allowed(const int failed_sets,
                                    const int max_recertifications)
{
  return max_recertifications > 0 && failed_sets >= 0 &&
         failed_sets < max_recertifications;
}
