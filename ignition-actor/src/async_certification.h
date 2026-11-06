#pragma once

#include <algorithm>

struct async_certification_admission
{
  int probes = 0;
  bool drain = false;
};

// Only allocate admissions: callers retain the existing readiness/sprint
// triggers, accuracy test, prospective-error history and cancellation policy.
// In-flight probes already provide future evidence and must not be duplicated.
inline async_certification_admission async_postfit_certification(
  bool enabled, bool asynchronous, bool probe_stop_enabled, bool sprint_ready,
  int n_training, int warmup, int scored, int required, int in_flight,
  int capacity)
{
  if (!enabled || !asynchronous || !probe_stop_enabled || !sprint_ready ||
      n_training < warmup || scored < 0 || required <= scored ||
      in_flight < 0 || capacity < 0)
    return {};
  const int missing = required - scored - in_flight;
  if (missing <= 0)
    return {0, true};
  return {std::min(capacity, missing), false};
}
