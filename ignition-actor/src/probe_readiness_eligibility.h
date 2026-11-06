#pragma once

// Eligibility is fixed when the prediction is made, before its threshold is
// observed. The option applies equally to the two surface scheduling policies;
// it does not change the one-dimensional workflow or the default-off rule.
inline bool probe_readiness_eligibility_active(bool enabled, bool surface)
{
  return enabled && surface;
}

// This describes admission eligibility, not a completed valid observation.
// Completion-time prediction/threshold/failure checks remain the caller's
// responsibility. Keeping them separate also preserves the historical
// default-off unfinished-probe count exactly.
inline bool probe_certification_eligible_at_admission(
  bool is_probe, int prediction_training_count, int warmup_count,
  bool require_readiness, bool prediction_readiness)
{
  return is_probe && prediction_training_count >= warmup_count &&
         (!require_readiness || prediction_readiness);
}
