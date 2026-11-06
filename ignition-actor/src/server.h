#ifndef RADIAL_IGNITION_SERVER_H
#define RADIAL_IGNITION_SERVER_H

#include "radial_common.h"
#include "gp_advisor_process.h"

#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct sample_t
{
  int         id = 0;
  double      R = 0.0;
  double      Aw = 0.0;
  double      gamma = 0.0;
  double      fast_wave_speed = 0.0;
  double      steepness = 0.0;
  std::string ic_type;
  bool        completed = false;
  bool        failed = false;
  bool        cancelled = false;
  bool        threshold_found = false;
  bool        converged = false;
  double      critical_Av = 0.0;
  double      Av_lo = 0.0;
  double      Av_hi = 0.0;
  unsigned int iterations = 0;
  unsigned int solver_runs = 0;
  // True when the final Av_lo and Av_hi were both set by observed solves
  // (quench below, ignition above). A GP-seeded bracket cancelled before its
  // ends were verified reports false and is never salvaged.
  bool        bracket_ends_observed = false;
  std::string error;
  std::string run_dir;
  int         gp_round = 0;
  bool        gp_prediction_available = false;
  bool        gp_bracket_prediction_available = false;
  double      gp_bracket_pred_mean = 0.0;
  double      gp_bracket_posterior_std = 0.0;
  // Training size of the fit that supplied the bracket forecast above
  // (early GP brackets refresh these for unfinished samples at every fit).
  int         gp_bracket_n_training = 0;
  // Certification set (adaptive_gamma_state::certification_epoch) this probe
  // was admitted or promoted for; it scores only into that set.
  int         gp_probe_epoch = 0;
  // Training-set size of the forecasting model, preserved for eligibility
  // provenance and the legacy count-based coverage/probe warm-up policies.
  int         gp_prediction_n_training = 0;
  // Drawn from the space-filling probe sequence rather than the acquisition.
  bool        gp_is_probe = false;
  // Frozen at admission from the same fit that supplied this prediction.
  // Later model changes must neither promote an ineligible probe nor discard
  // an eligible probe's error after its outcome becomes known.
  bool        gp_prediction_readiness = false;
  bool        gp_probe_certification_eligible = false;
  // Counted once toward prospective interval coverage. Accumulation is per
  // SAMPLE, not per round: asynchronous mode never has a fully finished round
  // at a decision point, and rounds complete out of order.
  bool        gp_prequential_counted = false;
  // Probe eligibility may precede the legacy coverage warm-up. Count these
  // immutable errors independently, exactly once, and never discard a miss.
  bool        gp_probe_counted = false;
  double      gp_pred_mean = 0.0;
  double      gp_pred_lower95 = 0.0;
  double      gp_pred_upper95 = 0.0;
  double      gp_posterior_std = 0.0;
  bool        gp_seeded_bracket = false;
  double      seed_Av_lo = 0.0;
  double      seed_Av_hi = 0.0;
  // Where the dispatched bracket came from: global or gp.
  std::string bracket_source = "global";
};

struct adaptive_gamma_state
{
  double gamma = 0.0;
  double fast_wave_speed = 0.0;
  int    rounds_started = 0;
  // Counts every GP fit, including convergence-only fits after the suggestion
  // budget is exhausted. This gives asynchronous output files unique names.
  int    fits_started = 0;
  int    last_gp_fit_valid_threshold_count = 0;
  // Admission recovery when a failed search frees capacity without supplying
  // a new observation (including the last outstanding certification probe).
  int    last_gp_decision_unfinished_count = 0;
  int    last_validation_reported_round = 0;
  double previous_integrated_model_variance =
    std::numeric_limits<double>::infinity();
  int         variance_stagnation_rounds = 0;
  int         accuracy_stop_streak = 0;
  // Running totals over every adaptive sample whose prediction was made before
  // that sample was observed.  These are the only leakage-free accuracy data
  // the campaign generates about itself, and the stopping rule uses them to
  // certify the posterior bands before trusting a width-based trigger.
  // Coverage ignores predictions below this training-count warm-up. Legacy
  // probes also require it; readiness-driven probes have separate eligibility.
  int         prequential_warmup_min = 0;
  int         prequential_total = 0;
  int         prequential_covered = 0;
  // Halton cursor for the probe sequence, and the scored probe errors.
  // Coordinate the advisor actually fitted R in; spacing is measured in it.
  bool                model_log_R = true;
  int                 probe_index = 1;
  int                 probes_dispatched = 0;
  std::vector<double> probe_relative_errors;
  // Sample ids of the errors above (same order), for the certification audit.
  std::vector<int>    probe_error_sample_ids;
  // Bounded re-certification (recertification.h): the current certification
  // set (0 = first) and how many completed sets missed the target and were
  // archived. Probes carry the set they were admitted or promoted for.
  int                 certification_epoch = 0;
  int                 certification_sets_failed = 0;
  bool                recertification_exhausted_logged = false;
  std::ofstream       certification_sets_file;
  bool        admission_freeze_announced = false;
  // Latched after the ordinary uncertainty/shape/boundary diagnostics first
  // pass in asynchronous mode.  Until enough post-warmup probes are scored or
  // in flight, every newly available slot is used for certification instead
  // of another acquisition point.
  bool        probe_certification_sprinting = false;
  // True while asynchronous scheduling is waiting only for enough already
  // dispatched, post-warmup probes to finish.  Keeping this state outside a
  // single GP evaluation lets ordinary completions return without launching
  // another expensive Python fit.
  bool        probe_certification_draining = false;
  // The surface advisor may choose its input coordinates and warp
  // automatically.  Cache that structural choice between periodic
  // re-evaluations so asynchronous mode does not fit every alternative after
  // every single completion.
  std::string selected_input_transform_R;
  std::string selected_input_transform_gamma;
  std::string selected_input_warp;
  int         last_input_model_selection_training_count = 0;
  int         next_input_model_selection_training_count = 0;
  std::string stop_reason;
  bool        generation_stopped = false;
  bool        stop_criterion_reached = false;
  bool        cancellation_requested = false;
  std::ofstream         gp_prequential_file;
  std::ofstream         gp_probe_admissions_file;
  // Opt-in in-flight probe promotion / readiness anticipation audits.
  std::ofstream         gp_probe_promotions_file;
  std::ofstream         gp_readiness_anticipation_file;
  int                   probes_promoted = 0;
  int                   anticipation_probes_admitted = 0;
  // Early GP brackets: fit-only passes made while seeds were queued, the
  // valid count of the last such pass, and the audit file of every
  // bracket decision (enqueue and dispatch).
  int                   bracket_fits_started = 0;
  int                   last_bracket_fit_valid_count = 0;
  std::ofstream         early_bracket_file;
  std::ofstream         gp_metrics_file;
  std::filesystem::path gp_output_dir;
};

struct server_state
{
  std::unique_ptr<gp_advisor_process> surface_advisor;
  std::vector<sample_t>      samples;
  std::deque<radial_job_t>   pending_jobs;
  std::unordered_set<caf::actor> active_workers;
  std::unordered_set<caf::actor> idle_workers;
  std::unordered_set<caf::actor> cancellation_controllers;
  std::unordered_map<caf::actor, caf::actor> worker_cancellation_controllers;
  std::unordered_map<caf::actor, int> worker_active_samples;
  int                        active_jobs = 0;
  int                        total_jobs_completed = 0;
  int                        total_jobs_cancelled = 0;
  bool                       termination_initiated = false;
  std::chrono::high_resolution_clock::time_point start_time;
  std::ofstream              results_file;
  std::filesystem::path      results_path;
  std::filesystem::path      gp_output_dir;
  // The bookkeeping fields are identical for a joint surface, but represent
  // one global GP rather than a single-gamma curve. Keeping a distinct state
  // prevents the legacy per-gamma scheduler from seeing a synthetic gamma.
  adaptive_gamma_state       adaptive_surface_state;
  // Early GP brackets at dispatch time (early_brackets.h); copied from the
  // configuration so the dispatch path needs no adaptive config.
  bool                       early_brackets_enabled = false;
  double                     early_bracket_sigma_k = 3.0;
  double                     early_bracket_abs_tol = 0.0;
  double                     early_bracket_rel_tol = 0.0;
  double                     early_bracket_quench_cost_ratio = 1.0;
  int                        early_brackets_applied = 0;
  double                     global_Av_lo = 0.0;
  double                     global_Av_hi = 0.0;
};

caf::behavior server(caf::stateful_actor<server_state> *self,
                     uint16_t port,
                     const config &cfg);

#endif
