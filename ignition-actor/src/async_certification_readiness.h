#pragma once

// This option controls speculative probe admissions and draining, not the
// prospective stopping test. Keep the scope explicit so Sync, one-dimensional
// fits, and the inherited non-postfit policy retain their original behavior.
inline bool async_certification_readiness_guard_active(
  bool enabled, bool surface, bool asynchronous, bool postfit, bool probe_stop)
{
  return enabled && surface && asynchronous && postfit && probe_stop;
}

inline bool async_certification_readiness_permits(bool guard, bool ready)
{
  return !guard || ready;
}

inline bool async_certification_sprint_trigger(
  bool guard, bool ready, bool latched_sprint, bool scored_probe_trigger)
{
  return guard ? ready : latched_sprint || scored_probe_trigger;
}

// A drain may reuse current readiness only until a new accepted observation
// arrives. The caller checks a completed passing probe criterion first, then
// requests fresh diagnostics rather than preserving a stale sprint latch.
inline bool async_certification_drain_needs_refit(
  bool guard, int n_valid, int last_fit_n_valid)
{
  return guard && n_valid > last_fit_n_valid;
}
