#pragma once

#include <algorithm>

// Pure counting rules for the opt-in in-flight probe promotion and readiness
// anticipation extensions of readiness-driven certification. Neither rule
// inspects an outcome: promotion order is dispatch order (sample id) and the
// quota depends only on already-scored and already-eligible evidence.

// Unfinished ineligible probes that may receive a fresh ready forecast now.
inline int inflight_promotion_quota(int required, int scored,
                                    int eligible_in_flight)
{
  if (required <= 0 || scored < 0 || eligible_in_flight < 0)
    return 0;
  return std::max(0, required - scored - eligible_in_flight);
}

// Probes to admit before readiness so that enough promotable probes are in
// flight when the forecast readiness arrives. Capacity bounds admissions.
inline int anticipation_probe_count(bool anticipated, int required, int scored,
                                    int eligible_in_flight,
                                    int promotable_in_flight, int capacity)
{
  if (!anticipated || required <= 0 || scored < 0 || eligible_in_flight < 0 ||
      promotable_in_flight < 0 || capacity <= 0)
    return 0;
  const int need =
    required - scored - eligible_in_flight - promotable_in_flight;
  return need > 0 ? std::min(capacity, need) : 0;
}
