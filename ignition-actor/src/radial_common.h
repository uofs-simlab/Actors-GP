#ifndef RADIAL_IGNITION_COMMON_H
#define RADIAL_IGNITION_COMMON_H

#include "caf/all.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

struct radial_job_t
{
  int         sample_id;
  double      R;
  double      Aw;
  double      gamma;
  double      fast_wave_speed;
  double      steepness;
  std::string ic_type;
  // Per-sample bisection bracket. With GP-seeded brackets disabled these are
  // the global [Av_lo, Av_hi] and the behaviour is identical to before the
  // feature existed. When seeding is on the server narrows them from the GP
  // posterior, which is why the bracket has to travel with the job rather than
  // living in the worker's static sim params.
  double      Av_lo;
  double      Av_hi;
  // A seeded lower end is NOT known to be sub-threshold, so it must be tested
  // rather than assumed. Only Av_lo == 0 is safe to assume.
  bool        verify_lower_endpoint;
};

template <class Inspector>
bool inspect(Inspector &f, radial_job_t &x)
{
  return f.object(x).fields(f.field("sample_id", x.sample_id),
                            f.field("R", x.R),
                            f.field("Aw", x.Aw),
                            f.field("gamma", x.gamma),
                            f.field("fast_wave_speed", x.fast_wave_speed),
                            f.field("steepness", x.steepness),
                            f.field("ic_type", x.ic_type),
                            f.field("Av_lo", x.Av_lo),
                            f.field("Av_hi", x.Av_hi),
                            f.field("verify_lower_endpoint",
                                    x.verify_lower_endpoint));
}

struct radial_sim_params_t
{
  double       Av_lo = 0.0;
  double       Av_hi = 10.0;
  double       bisection_tolerance = 1e-4;
  double       bisection_relative_tolerance = 0.0;
  unsigned int bisection_max_iterations = 50;
  // Probe ladder. Av_hi is not merely a bound. It is the FIRST value every
  // bisection probes, and its cost is superlinear: the FHN reaction Jacobian is
  // -3v^2 + 2(1+beta)v - beta, so v=10 is ~4x stiffer than v=5 and ~300x
  // stiffer than v=1, while the supra-threshold footprint it buys grows only
  // like ln(Av)/(2k). Escalating from a small amplitude finds the same root far
  // more cheaply and yields a TIGHTER starting bracket.
  bool         bisection_probe_ladder = false;
  double       bisection_probe_start = 1.0;
  double       bisection_probe_ratio = 2.0;

  // Pin each worker's solve to a disjoint core slice [i*M, (i+1)*M). Required
  // whenever a node hosts MORE THAN ONE worker: concurrent unpinned launches
  // can be placed on the SAME cores, which is a cliff rather than a slowdown
  // (measured 34x for an unbound 16-rank solve vs a bound one) because MPI
  // ranks busy-wait. Off by default so the validated one-worker-per-node and
  // radial direct-launch paths stay byte-identical.
  bool         pin_solver_cores = false;

  double       domain_half_length = 200.0;
  unsigned int n_refinements = 10;
  unsigned int fe_degree = 2;
  double       final_time = 100.0;
  double       fast_wave_speed = 0.50;
  // domain_half_length is L/2 for a Cartesian box of side length L. Use an
  // upper bound on the outward wave speed so observation ends before the wave
  // reaches the radial boundary.
  double       finite_time_safety_factor = 0.9;
  double       time_step = 5e-2;
  unsigned int output_stride = 1000000000u;
  unsigned int mesh_adaptation_frequency = 50;
  unsigned int max_delta_refinement_level = 4;
  double       mesh_refinement_fraction = 0.3;
  double       mesh_coarsening_fraction = 0.05;
  bool         pre_refine_initial_condition = false;

  double       arkode_minimum_step_size = 1e-8;
  unsigned int arkode_maximum_order = 2;
  unsigned int arkode_maximum_non_linear_iterations = 20;
  double       arkode_absolute_tolerance = 1e-4;
  double       arkode_relative_tolerance = 1e-3;

  unsigned int diagnostic_stride = 20;
  double       excitation_threshold = 0.3;
  double       diagnosis_start_time = 0.5;
  double       ignition_max_v_threshold = 0.5;
  double       ignition_excited_fraction_threshold = 1e-2;
  double       ignition_sustain_window = 5.0;
  double       ignition_area_decay_tolerance = 0.02;
  // Opt-in solver early exits (see tostii parallel_bdf_fhn.cc). They shorten
  // runs whose outcome is already decided and must not change any decision.
  bool         early_exit_on_quench = false;
  double       quench_max_v_threshold = 0.05;
  double       quench_window = 10.0;
  bool         early_exit_on_propagation = false;

  double       alpha_param = 0.37;
  double       diffusion_v = 1.0;
  double       diffusion_w = 0.0;
  double       beta_param = 0.131655;

  std::string  output_dir = "radial_outputs";
  std::string  solver_backend = "arkode";
  std::string  solver_binary;
  std::string  solver_runner = "direct";
  unsigned int solver_mpi_ranks = 1;
  unsigned int solver_first_core = 0;

  // --- 2D wave_segment support -------------------------------------------
  // For ic_type == "wave_segment" the swept variable (job.R) is the SEGMENT
  // LENGTH, not a radius: `segment_wave_width` is the excited band's x-extent
  // (emitted as the solver's `Radius`) and `segment_tail_width` is the
  // refractory tail behind it. Ignored for every other initial condition, so
  // radial parameter files are unaffected.
  double       segment_wave_width = 5.0;
  double       segment_tail_width = 20.0;

  // Propagation-detector oracle. Emitted ONLY when enabled, so radial
  // parameter files remain byte-identical to the pre-2D behaviour.
  bool         use_detector_oracle = false;
  double       detector_x = 50.0;
  double       detector_min_excited_area = 1.0;
  // Radial propagation detector: when positive, the station is placed at this
  // distance beyond each sample's own disk radius (x = R + offset) instead of
  // at the fixed detector_x. 0 keeps the fixed station (wave segment).
  double       detector_offset_from_radius = 0.0;
  // Hardened detector verdict (solver 'Detector require area growth'); emitted
  // only when true so existing parameter files are unchanged.
  bool         detector_require_area_growth = false;
};

template <class Inspector>
bool inspect(Inspector &f, radial_sim_params_t &x)
{
  return f.object(x).fields(
    f.field("Av_lo", x.Av_lo),
    f.field("Av_hi", x.Av_hi),
    f.field("bisection_tolerance", x.bisection_tolerance),
    f.field("bisection_relative_tolerance", x.bisection_relative_tolerance),
    f.field("bisection_max_iterations", x.bisection_max_iterations),
    f.field("pin_solver_cores", x.pin_solver_cores),
    f.field("bisection_probe_ladder", x.bisection_probe_ladder),
    f.field("bisection_probe_start", x.bisection_probe_start),
    f.field("bisection_probe_ratio", x.bisection_probe_ratio),
    f.field("domain_half_length", x.domain_half_length),
    f.field("n_refinements", x.n_refinements),
    f.field("fe_degree", x.fe_degree),
    f.field("final_time", x.final_time),
    f.field("fast_wave_speed", x.fast_wave_speed),
    f.field("finite_time_safety_factor", x.finite_time_safety_factor),
    f.field("time_step", x.time_step),
    f.field("output_stride", x.output_stride),
    f.field("mesh_adaptation_frequency", x.mesh_adaptation_frequency),
    f.field("max_delta_refinement_level", x.max_delta_refinement_level),
    f.field("mesh_refinement_fraction", x.mesh_refinement_fraction),
    f.field("mesh_coarsening_fraction", x.mesh_coarsening_fraction),
    f.field("pre_refine_initial_condition", x.pre_refine_initial_condition),
    f.field("arkode_minimum_step_size", x.arkode_minimum_step_size),
    f.field("arkode_maximum_order", x.arkode_maximum_order),
    f.field("arkode_maximum_non_linear_iterations", x.arkode_maximum_non_linear_iterations),
    f.field("arkode_absolute_tolerance", x.arkode_absolute_tolerance),
    f.field("arkode_relative_tolerance", x.arkode_relative_tolerance),
    f.field("diagnostic_stride", x.diagnostic_stride),
    f.field("excitation_threshold", x.excitation_threshold),
    f.field("diagnosis_start_time", x.diagnosis_start_time),
    f.field("ignition_max_v_threshold", x.ignition_max_v_threshold),
    f.field("ignition_excited_fraction_threshold", x.ignition_excited_fraction_threshold),
    f.field("ignition_sustain_window", x.ignition_sustain_window),
    f.field("ignition_area_decay_tolerance", x.ignition_area_decay_tolerance),
    f.field("early_exit_on_quench", x.early_exit_on_quench),
    f.field("quench_max_v_threshold", x.quench_max_v_threshold),
    f.field("quench_window", x.quench_window),
    f.field("early_exit_on_propagation", x.early_exit_on_propagation),
    f.field("alpha_param", x.alpha_param),
    f.field("diffusion_v", x.diffusion_v),
    f.field("diffusion_w", x.diffusion_w),
    f.field("beta_param", x.beta_param),
    f.field("output_dir", x.output_dir),
    f.field("solver_backend", x.solver_backend),
    f.field("solver_binary", x.solver_binary),
    f.field("solver_runner", x.solver_runner),
    f.field("solver_mpi_ranks", x.solver_mpi_ranks),
    f.field("solver_first_core", x.solver_first_core),
    f.field("segment_wave_width", x.segment_wave_width),
    f.field("segment_tail_width", x.segment_tail_width),
    f.field("use_detector_oracle", x.use_detector_oracle),
    f.field("detector_x", x.detector_x),
    f.field("detector_min_excited_area", x.detector_min_excited_area),
    f.field("detector_offset_from_radius", x.detector_offset_from_radius),
    f.field("detector_require_area_growth", x.detector_require_area_growth));
}

CAF_BEGIN_TYPE_ID_BLOCK(radial_ignition_project, caf::first_custom_type_id)
  CAF_ADD_TYPE_ID(radial_ignition_project, (radial_job_t))
  CAF_ADD_TYPE_ID(radial_ignition_project, (radial_sim_params_t))
CAF_END_TYPE_ID_BLOCK(radial_ignition_project)

// The uniform W*M rule forces the node's core count to be a multiple of M, so
// reserving even one core for the scheduler + GP costs a whole worker: on 32
// cores with M=8, reserving 1 leaves (32-1)/8 = 3 workers and 7 cores idle.
// Splitting a 31-core budget four ways gives 8,8,8,7 instead, one core freed
// for the GP at the cost of one rank on one worker. A budget of 0 reproduces
// the legacy uniform layout exactly.
inline std::pair<unsigned int, unsigned int>
worker_core_slice(int          index,
                  int          n_workers,
                  int          budget,
                  unsigned int nominal_ranks)
{
  if (n_workers <= 0 || index < 0 || index >= n_workers)
    return {0u, nominal_ranks};
  const int usable = std::min(budget, n_workers * static_cast<int>(nominal_ranks));
  const int base   = usable / n_workers;
  if (usable <= 0 || base <= 0)
    return {static_cast<unsigned int>(index) * nominal_ranks, nominal_ranks};
  const int rem   = usable % n_workers;
  // The first `rem` workers carry one extra core.
  const int ranks = base + (index < rem ? 1 : 0);
  const int first = index * base + std::min(index, rem);
  return {static_cast<unsigned int>(first), static_cast<unsigned int>(ranks)};
}

class config : public caf::actor_system_config
{
public:
  uint16_t    port = 0;
  std::string host = "localhost";
  bool        server_mode = false;
  int         n_workers = 4;
  // Cores this node makes available to its workers. 0 keeps the legacy
  // uniform rule (worker i owns [i*M, (i+1)*M)). A budget that is not a
  // multiple of M is split as evenly as possible, which is what lets the
  // server node reserve a core for the scheduler + GP without forfeiting a
  // whole worker: 31 cores among 4 workers is 8,8,8,7
  int         worker_core_budget = 0;

  std::string ic_type = "smooth_disk";
  std::string gamma_values;
  std::string gamma_wave_speeds;
  std::string R_values;

  double R_min = 20.0;
  double R_max = 100.0;
  int    n_R = 40;
  double gam_min = 0.025;
  double gam_max = 0.025;
  int    n_gam = 1;

  double steepness = 5.0;

  bool        adaptive_gp = false;
  // Fit one joint A_v^*(R, gamma) surface instead of one independent R curve
  // for every discrete gamma. In surface mode gamma-min/max define a
  // continuous parameter interval and adaptive-initial-samples is the TOTAL
  // initial design size, not a per-gamma count.
  bool        adaptive_gp_surface = false;
  std::string adaptive_gp_mode = "synchronous";
  int         adaptive_initial_samples = 5;
  int         adaptive_batch_size = 5;
  int         adaptive_rounds = 0;
  int         adaptive_max_samples = 0;
  bool        adaptive_cancel_running_on_stop = false;
  int         adaptive_async_fill_cap = 0;
  int         adaptive_async_target_active = 0;
  std::string gp_python = "python3";
  std::string gp_script = "gp_avstar_surrogate_botorch.py";
  std::string gp_surface_script = "gp_avstar_surface_botorch.py";
  bool        gp_persistent_advisor = false;
  std::string gp_out_dir;
  int         gp_grid_size = 2000;
  std::string gp_acquisition = "variation-nipv";
  int         gp_integration_grid_size = 512;
  double      gp_variation_weight = 0.5;
  double      gp_focus_relative_uncertainty_target = 0.0;
  double      gp_focus_variation_coverage = 1.0;
  int         gp_acquisition_chunk_size = 128;
  std::string gp_surface_nipv_backend = "botorch";
  int         gp_joint_num_restarts = 12;
  int         gp_joint_raw_samples = 256;
  int         gp_joint_maxiter = 200;
  int         gp_joint_retries = 3;
  std::string gp_joint_fallback = "greedy-nipv";
  double      gp_surface_suggest_min_distance = 0.01;
  double      gp_surface_suggest_min_distance_lengthscales = 0.15;
  std::string gp_surface_input_transform_R = "auto";
  std::string gp_surface_input_transform_gamma = "auto";
  // Async surface fits reuse the selected coordinate/warp structure between
  // periodic full comparisons. 0 locks the first selection for the run.
  int         gp_surface_model_reselect_every = 10;
  // Reconsider cached Async structure when the advisor offers new choices.
  // Default off; a zero reselect interval continues to lock the first choice.
  bool        gp_surface_model_eligibility_refresh = false;
  double      gp_acquisition_value_target = 0.0;
  int         adaptive_min_fit_samples = 0;
  bool        gp_async_early_start = false;
  // Decide early, bracket conservatively: with a positive value the first
  // Async decision waits only for this many valid thresholds (>= 3) and an
  // idle worker, not for the corner/half-design gate; GP-seeded brackets
  // still wait for that gate, so a weak early model costs locations, never
  // verification solves. Later decisions follow every completion as before.
  int         gp_async_start_min_valid = 0;
  // Async early GP brackets: fit the surface as soon as the first-decision
  //     count of valid thresholds exists, even while seeds are still queued,
  //     and give every later dispatch (queued seeds included) the GP bracket
  //     whenever it is expected to need fewer solves than the global one
  //     (early_brackets.h). No corner/half-design gate. Requires
  //     --gp-seeded-brackets.
  bool        gp_async_early_brackets = false;
  //     Cost of a quenched solve relative to an igniting one in that rule
  //     (early-exit solver: measured ~0.02 radial); 1 = plain solve count.
  double      gp_bracket_quench_cost_ratio = 1.0;
  // Bounded re-certification (both policies): after a completed
  // certification set misses the probe RMS target, archive it and let a fresh
  // set of prospective probes certify, at most this many times. 0 keeps the
  // original behaviour (no further accuracy stop after a failed set).
  int         gp_probe_max_recertifications = 0;
  // Under readiness-driven certification a probe admitted before readiness
  // can never certify; the regular 1-in-3 background probes are then only
  // space-filling training points. This drops them in both policies and
  // leaves certification (Sync batch, Async anticipation/sprint) untouched.
  bool        gp_background_probes_off = false;
  double      gp_salvage_max_relative_half_width = 0.25;
  std::string gp_surface_sample_list;
  std::string gp_surface_design_ladder_R = "blend";
  std::string gp_surface_design_ladder_gamma = "blend";
  double      gp_surface_interior_margin = 0.05;
  int         gp_surface_candidate_count = 4096;
  int         gp_surface_prediction_r_count = 160;
  int         gp_surface_prediction_gamma_count = 100;
  double      gp_surface_boundary_tolerance = 0.02;
  double      gp_relative_uncertainty_max_tolerance = 0.0;
  double      gp_relative_uncertainty_rms_tolerance = 0.0;
  double      gp_shape_loo_max_tolerance = 0.0;
  double      gp_shape_loo_rms_tolerance = 0.0;
  double      gp_integrated_variance_tolerance = 0.0;
  double      gp_variance_min_relative_improvement = 0.01;
  int         gp_variance_patience = 0;
  int         gp_stop_hysteresis = 2;
  // Prospective-calibration gate.  Stopping triggers are all functions of the
  // posterior width, which only means something if the posterior is calibrated.
  // Require that many scored before-the-fact predictions, and that their 95%
  // coverage clears the lower bound, before any width-based stop is honoured.
  // Set the bound to 0 to disable the gate.
  // Space-filling probes.  One in every gp_probe_every adaptive dispatches is
  // drawn from a Halton sequence over the model square instead of from the
  // acquisition, and its prediction is recorded before it runs.  The RMS
  // relative error of those predictions is an UNBIASED estimate of
  // generalization error under that measure -- unlike leave-one-out, which is
  // optimistic on an acquisition-chosen design, and unlike scoring the
  // acquisition's own picks, which migrate away from whatever region is already
  // resolved (measured: 0% of mature acquisition predictions at R<3 against 46%
  // of the hold-out, reading 2-5x optimistic).
  int         gp_probe_every = 4;
  int         gp_probe_min_samples = 10;
  double      gp_probe_rms_tolerance = 0.01;
  // Freeze certification eligibility before observing a probe: when enabled,
  // the forecasting surface model must satisfy readiness as well as warm-up.
  // Background probes still train the GP and retain their prediction errors.
  bool        gp_probe_require_readiness = false;
  // Shared surface policy: readiness replaces count-based probe eligibility.
  // The legacy coverage warm-up is unchanged.
  bool        gp_probe_readiness_transition = false;
  bool        gp_sync_certification_batches = false;
  bool        gp_async_postfit_certification = false;
  bool        gp_async_certification_require_readiness = false;
  // Promote in-flight space-filling probes at a ready fit, with a fresh
  // forecast frozen before their outcome exists (Async only in practice:
  // a Sync barrier never has in-flight work at a fit).
  bool        gp_probe_inflight_promotion = false;
  // Admit probes before readiness when current LOO diagnostics pass and the
  // in-flight work is forecast to satisfy the uncertainty limits.
  bool        gp_readiness_anticipation = false;
  int         gp_prequential_min_samples = 12;
  // Coverage on acquisition-chosen points runs BELOW nominal by construction --
  // NIPV picks the points of largest predictive variance, i.e. the ones most
  // likely to fall outside their band.  Measured 88-93% against a nominal 95%.
  // The bound is therefore set to catch genuine breakdown, not to certify
  // nominal calibration: at n=40 it blocks below ~80% observed coverage.
  double      gp_prequential_coverage_lower_bound = 0.70;
  // Training-count warm-up for prospective coverage and legacy probe
  // eligibility. Readiness-driven probes do not use this count; all forecasts
  // remain recorded, including those excluded from either statistic.
  double      gp_prequential_warmup_factor = 1.5;
  double      gp_convergence_r_min = 0.0;
  double      gp_noise_floor = 1e-3;
  std::string gp_target_transform = "log";
  bool        gp_no_plot = false;

  // GP-seeded bisection brackets. Off by default.
  // When on, a sample whose GP prediction is available
  // starts from [median*exp(-k*sigma), median*exp(+k*sigma)] instead of the
  // global bracket. Both endpoints are then verified,
  // which is what keeps a wrong prediction detectable rather than silently
  // biasing the result toward the prior.
  bool        gp_seeded_brackets = false;
  double      gp_seed_sigma_k = 3.0;

  double      Aw = 0.0;

  // Use a single observation window (--final-time) for every sample instead of
  // the radial override T(R) = safety*(boundary - R)/c. Required by the 2D
  // wave-segment study, whose detector oracle is only comparable across samples
  // if T is the same for all of them; it also makes a wave speed meaningless,
  // so a non-positive one stops being an error.
  bool        fixed_final_time = false;

  // Optional continuation data for resolving the stable fast-wave speed at
  // arbitrary gamma values. The expected whitespace format has gamma in
  // column 1 and c in column 6 and contains slow and fast branches separated
  // by the maximum-gamma fold.
  std::string wave_speed_continuation_file;
  double      wave_speed_min_reliable_gamma = 0.00048828125; // 2^-11
  double      wave_speed_low_gamma_plateau = 0.5192;
  double      wave_speed_round_up = 0.0001;

  radial_sim_params_t sim;

  config()
  {
    opt_group{custom_options_, "global"}
      .add(port, "port,p", "server port")
      .add(host, "host,H", "server hostname in client mode")
      .add(server_mode, "server-mode,s", "run as server / coordinator")
      .add(n_workers, "n-workers,w", "number of local worker actors")
      .add(worker_core_budget, "worker-core-budget",
           "cores on this node available to its workers "
           "(0 = n-workers * solver-mpi-ranks)")
      .add(ic_type, "ic-type", "radial initial condition: disk | smooth_disk")
      .add(gamma_values, "gamma-values",
           "comma-separated gamma list; overrides gamma-min/max/n")
      .add(gamma_wave_speeds, "gamma-wave-speeds",
           "comma-separated gamma:speed mapping; each adaptive gamma must "
           "have a positive speed")
      .add(R_values, "R-values",
           "comma-separated radius list; overrides R-min/max/n-R")
      .add(R_min, "R-min", "minimum disk radius")
      .add(R_max, "R-max", "maximum disk radius")
      .add(n_R, "n-R", "number of radius sample points")
      .add(gam_min, "gamma-min", "minimum gamma")
      .add(gam_max, "gamma-max", "maximum gamma")
      .add(n_gam, "n-gamma", "number of gamma points")
      .add(adaptive_gp, "adaptive-gp",
           "enable adaptive GP sampling after an initial radius batch")
      .add(adaptive_gp_surface, "adaptive-gp-surface",
           "fit a joint two-input GP over continuous (R,gamma); gamma-min/max "
           "define the surface interval")
      .add(adaptive_gp_mode, "adaptive-gp-mode",
           "adaptive scheduling mode: synchronous or asynchronous")
      .add(adaptive_initial_samples, "adaptive-initial-samples",
           "initial evenly spaced R samples before the first GP fit")
      .add(adaptive_batch_size, "adaptive-batch-size",
           "synchronous GP batch size; asynchronous mode refills idle slots")
      .add(adaptive_rounds, "adaptive-rounds",
           "maximum number of GP suggestion rounds after the initial batch")
      .add(adaptive_max_samples, "adaptive-max-samples",
           "maximum attempted R samples per gamma, including its initial "
           "batch; 0 disables")
      .add(adaptive_cancel_running_on_stop,
           "adaptive-cancel-running-on-stop",
           "cancel active asynchronous samples when a GP stopping criterion "
           "is reached")
      .add(adaptive_async_fill_cap,
           "adaptive-async-fill-cap",
           "asynchronous mode: max new samples requested per gamma per refill "
           "pass (0 = fill idle workers, fair-shared across gammas)")
      .add(adaptive_async_target_active,
           "adaptive-async-target-active",
           "asynchronous surface mode: target number of active/reserved "
           "samples; refill only the deficit after each completion (0 = use "
           "the legacy fill-idle policy)")
      .add(gp_python, "gp-python",
           "Python executable used to run the GP advisor")
      .add(gp_script, "gp-script",
           "path to gp_avstar_surrogate_botorch.py")
      .add(gp_surface_script, "gp-surface-script",
           "path to gp_avstar_surface_botorch.py")
      .add(gp_persistent_advisor, "gp-persistent-advisor",
           "experimental surface advisor: retain one Python process and fitted GP")
      .add(gp_out_dir, "gp-out-dir",
           "directory for GP advisor outputs; defaults to output-dir/gp_adaptive")
      .add(gp_grid_size, "gp-grid-size",
           "number of posterior grid points used by the GP advisor")
      .add(gp_acquisition, "gp-acquisition",
           "GP batch acquisition: max-std, greedy-nipv, variation-nipv, or "
           "joint-nipv")
      .add(gp_integration_grid_size, "gp-integration-grid-size",
           "number of R points used to integrate posterior variance")
      .add(gp_variation_weight, "gp-variation-weight",
           "variation-nipv mixture weight for learned response variation; "
           "the remainder is uniform exploration")
      .add(gp_focus_relative_uncertainty_target,
           "gp-focus-relative-uncertainty-target",
           "variation-nipv candidate focus: retain locations above this "
           "relative 95% uncertainty; 0 disables")
      .add(gp_focus_variation_coverage,
           "gp-focus-variation-coverage",
           "variation-nipv candidate focus: also retain largest-slope "
           "locations accounting for this response-variation fraction; "
           "1 disables")
      .add(gp_acquisition_chunk_size, "gp-acquisition-chunk-size",
           "number of candidates or initial batches evaluated per NIPV chunk")
      .add(gp_surface_nipv_backend, "gp-surface-nipv-backend",
           "surface NIPV evaluator: botorch or analytic (same objective)")
      .add(gp_joint_num_restarts, "gp-joint-num-restarts",
           "multistart optimizations used by joint-nipv")
      .add(gp_joint_raw_samples, "gp-joint-raw-samples",
           "feasible batches screened to initialize joint-nipv")
      .add(gp_joint_maxiter, "gp-joint-maxiter",
           "maximum optimizer iterations per joint-nipv restart")
      .add(gp_joint_retries, "gp-joint-retries",
           "independent joint-nipv attempts before fallback")
      .add(gp_joint_fallback, "gp-joint-fallback",
           "joint-nipv failure behavior: greedy-nipv or error")
      .add(gp_surface_suggest_min_distance,
           "gp-surface-suggest-min-distance",
           "hard floor on the distance to an attempted or pending location, "
           "in the model unit square")
      .add(gp_surface_suggest_min_distance_lengthscales,
           "gp-surface-suggest-min-distance-lengthscales",
           "preferred separation in fitted-lengthscale units, so the guard "
           "tracks the anisotropy the data reveals; 0 disables")
      .add(gp_surface_input_transform_R,
           "gp-surface-input-transform-r",
           "coordinate the R axis is modelled in: auto, log or linear")
      .add(gp_surface_input_transform_gamma,
           "gp-surface-input-transform-gamma",
           "coordinate the gamma axis is modelled in: auto, log or linear")
      .add(gp_surface_model_reselect_every,
           "gp-surface-model-reselect-every",
           "asynchronous surface mode: completed thresholds between full "
           "input-coordinate/warp comparisons; 0 locks the first selection")
      .add(gp_surface_model_eligibility_refresh,
           "gp-surface-model-eligibility-refresh",
           "Async surface only: also reselect when new model candidates "
           "become eligible; requires a positive model-reselect interval")
      .add(gp_acquisition_value_target,
           "gp-acquisition-value-target",
           "let the surrogate size its own batch: commit only the prefix "
           "capturing this fraction of the variance reduction the full "
           "requested batch would achieve; 0 requests the full batch")
      .add(adaptive_min_fit_samples,
           "adaptive-min-fit-samples",
           "asynchronous mode only: valid thresholds needed before the FIRST "
           "GP fit, instead of waiting for the whole seed design; 0 keeps the "
           "old behaviour of waiting for all of it")
      .add(gp_async_early_start, "gp-async-early-start",
           "Async surface only: allow first fit during seed drain after "
           "at least max(12, ceil(half the design)) valid seeds, capped by "
           "the full design size, and all four "
           "domain corners; preserves the full initial design and warm-up")
      .add(gp_background_probes_off, "gp-background-probes-off",
           "both policies: do not interleave the regular 1-in-N space-filling "
           "probes into ordinary acquisitions; probes come only from "
           "certification (Sync readiness batch, Async anticipation/sprint)")
      .add(gp_async_start_min_valid, "gp-async-start-min-valid",
           "Async surface only: when > 0, make the first adaptive decision as "
           "soon as max(3, this many) valid thresholds exist and a worker is "
           "idle, without the corner/half-design gate; GP-seeded brackets "
           "still wait for that gate")
      .add(gp_async_early_brackets, "gp-async-early-brackets",
           "Async surface only: fit the GP as soon as the first-decision "
           "count of valid thresholds exists, even while seeds are queued, "
           "and give every later dispatch (queued seeds included) the "
           "GP-seeded bracket when it is expected to need fewer solves than "
           "the global one; replaces the corner gate for brackets")
      .add(gp_probe_max_recertifications, "gp-probe-max-recertifications",
           "both policies: after a completed certification set misses the "
           "probe RMS target, archive it and allow a fresh set of prospective "
           "probes, at most this many times; 0 keeps the original behaviour")
      .add(gp_bracket_quench_cost_ratio, "gp-bracket-quench-cost-ratio",
           "early GP brackets: cost of a quenched solve relative to an "
           "igniting one in the expected-cost rule (early-exit solver: "
           "measured ~0.02); 1 = plain solve counting")
      .add(gp_salvage_max_relative_half_width,
           "gp-salvage-max-relative-half-width",
           "keep a cancelled bisection's partial bracket as a noisy "
           "observation when its relative half-width is at most this; "
           "0 disables salvage")
      .add(gp_surface_sample_list,
           "gp-surface-sample-list",
           "CSV of explicit (R,gamma) rows to run instead of the generated "
           "seed design; used for fixed hold-out sets")
      .add(gp_surface_design_ladder_R,
           "gp-surface-design-ladder-r",
           "spacing of the seed design along R: blend, log or linear")
      .add(gp_surface_design_ladder_gamma,
           "gp-surface-design-ladder-gamma",
           "spacing of the seed design along gamma: blend, log or linear")
      .add(gp_surface_interior_margin,
           "gp-surface-interior-margin",
           "model unit-square margin separating interior from boundary "
           "leave-one-out residuals")
      .add(gp_surface_candidate_count, "gp-surface-candidate-count",
           "number of Sobol candidates considered by the surface advisor")
      .add(gp_surface_prediction_r_count,
           "gp-surface-prediction-r-count",
           "number of R coordinates in the saved surface prediction grid")
      .add(gp_surface_prediction_gamma_count,
           "gp-surface-prediction-gamma-count",
           "number of gamma coordinates in the saved surface prediction grid")
      .add(gp_surface_boundary_tolerance,
           "gp-surface-boundary-tolerance",
           "normalized tolerance used to certify support on all four surface "
           "boundaries")
      .add(gp_relative_uncertainty_max_tolerance,
           "gp-relative-uncertainty-max-tolerance",
           "maximum physical relative 95% posterior half-width; 0 disables")
      .add(gp_relative_uncertainty_rms_tolerance,
           "gp-relative-uncertainty-rms-tolerance",
           "RMS physical relative 95% posterior half-width; 0 disables")
      .add(gp_shape_loo_max_tolerance,
           "gp-shape-loo-max-tolerance",
           "maximum local leave-one-out relative shape error required before "
           "an uncertainty stop; 0 disables")
      .add(gp_shape_loo_rms_tolerance,
           "gp-shape-loo-rms-tolerance",
           "RMS local leave-one-out relative shape error required before an "
           "uncertainty stop; 0 disables")
      .add(gp_integrated_variance_tolerance,
           "gp-integrated-variance-tolerance",
           "domain-averaged integrated model-space posterior-variance "
           "tolerance; 0 disables")
      .add(gp_variance_min_relative_improvement,
           "gp-variance-min-relative-improvement",
           "minimum relative integrated-variance reduction that resets "
           "stagnation patience")
      .add(gp_variance_patience, "gp-variance-patience",
           "stop after this many consecutive GP fits without sufficient "
           "integrated-variance reduction; 0 disables")
      .add(gp_seeded_brackets, "gp-seeded-brackets",
           "seed each sample's bisection bracket from the GP posterior "
           "instead of the global [Av_lo, Av_hi]; both endpoints are then "
           "verified, and a miss falls back to the global bracket")
      .add(gp_seed_sigma_k, "gp-seed-sigma-k",
           "half-width of the GP-seeded bracket in posterior standard "
           "deviations (log space); larger is safer but saves fewer solves")
      .add(Aw, "aw",
           "inhibitor amplitude A_w of the initial condition (wave_segment "
           "refractory tail); 0 reproduces the original no-tail behaviour")
      .add(fixed_final_time, "fixed-final-time",
           "use --final-time for every sample instead of the radial "
           "T(R)=safety*(boundary-R)/c override; also permits a non-positive "
           "fast wave speed, which is then forwarded as 0 to the solver")
      .add(wave_speed_continuation_file, "wave-speed-continuation-file",
           "continuation data used to interpolate the stable fast-wave speed "
           "for continuous gamma")
      .add(wave_speed_min_reliable_gamma,
           "wave-speed-min-reliable-gamma",
           "below this gamma use the configured fast-wave plateau")
      .add(wave_speed_low_gamma_plateau,
           "wave-speed-low-gamma-plateau",
           "conservative fast-wave speed below the reliable continuation range")
      .add(wave_speed_round_up, "wave-speed-round-up",
           "round interpolated fast-wave speeds upward by this increment; 0 "
           "disables rounding")
      .add(gp_probe_every, "gp-probe-every",
           "draw one space-filling probe per this many adaptive dispatches; "
           "0 disables probing")
      .add(gp_probe_min_samples, "gp-probe-min-samples",
           "probe predictions that must be scored before their RMS error may "
           "satisfy the stopping rule")
      .add(gp_probe_require_readiness, "gp-probe-require-readiness",
           "opt-in surface Sync/Async probe certification eligibility fixed "
           "at admission from warm-up and current model readiness; all probe "
           "observations and prediction errors remain recorded")
      .add(gp_probe_readiness_transition, "gp-probe-readiness-transition",
           "opt-in surface Sync/Async readiness-based probe eligibility; "
           "enables policy-appropriate certification, supersedes Async "
           "warm-up holds, and requires positive uncertainty/LOO tolerances")
      .add(gp_sync_certification_batches, "gp-sync-certification-batches",
           "opt-in surface Sync probe-only batches after readiness and warm-up; "
           "does not change stopping tolerances or Async scheduling")
      .add(gp_async_postfit_certification, "gp-async-postfit-certification",
           "opt-in surface Async probe-only refills using current-fit readiness "
           "and missing evidence after counting eligible in-flight probes")
      .add(gp_async_certification_require_readiness,
           "gp-async-certification-require-readiness",
           "opt-in surface Async current-readiness guard for probe-only "
           "admissions and draining; requires gp-async-postfit-certification; "
           "does not change the prospective probe stopping criterion")
      .add(gp_probe_rms_tolerance, "gp-probe-rms-tolerance",
           "stop when the RMS relative error of scored probe predictions falls "
           "below this; 0 disables the probe stopping criterion")
      .add(gp_probe_inflight_promotion, "gp-probe-inflight-promotion",
           "opt-in readiness-transition extension: at a ready fit, unfinished "
           "probes receive a fresh frozen forecast and become eligible, "
           "earliest dispatch first, up to the missing certification count")
      .add(gp_readiness_anticipation, "gp-readiness-anticipation",
           "opt-in readiness-transition extension: admit probes before "
           "readiness when LOO passes and in-flight work is forecast to meet "
           "the uncertainty limits; requires gp-probe-inflight-promotion")
      .add(gp_prequential_min_samples, "gp-prequential-min-samples",
           "before-the-fact predictions that must be scored before a "
           "width-based stop is honoured; the acquisition picks the hardest "
           "points, so this check is conservative and leakage-free")
      .add(gp_prequential_warmup_factor, "gp-prequential-warmup-factor",
           "discard prospective predictions from models trained on fewer than "
           "this multiple of the seed; they describe a model that no longer "
           "exists")
      .add(gp_prequential_coverage_lower_bound,
           "gp-prequential-coverage-lower-bound",
           "one-sided 95% Wilson lower bound the observed prospective 95% "
           "coverage must clear before stopping; 0 disables the gate")
      .add(gp_stop_hysteresis, "gp-stop-hysteresis",
           "require the accuracy stopping target to hold for this many "
           "consecutive GP fits before stopping (hysteresis); >=1, "
           "1 disables")
      .add(gp_convergence_r_min, "gp-convergence-r-min",
           "lower R bound for convergence checks; 0 uses R-min")
      .add(gp_noise_floor, "gp-noise-floor",
           "minimum GP observation standard deviation in model target space")
      .add(gp_target_transform, "gp-target-transform",
           "GP target transform: log or raw")
      .add(gp_no_plot, "gp-no-plot",
           "skip GP plot creation during adaptive rounds")
      .add(sim.Av_lo, "Av-lo",
           "lower Av bracket (assumed non-igniting; not simulated)")
      .add(sim.Av_hi, "Av-hi", "upper Av bracket tested before bisection")
      .add(sim.bisection_tolerance, "bsec-tol",
           "absolute target width of the critical Av bracket (floor)")
      .add(sim.pin_solver_cores, "pin-solver-cores",
           "pin each worker's solve to its own core slice; required when a "
           "node hosts more than one worker")
      .add(sim.bisection_probe_ladder, "bsec-probe-ladder",
           "escalate from --bsec-probe-start instead of probing Av_hi first; "
           "cheaper (Av_hi is the stiffest point) and yields a tighter bracket")
      .add(sim.bisection_probe_start, "bsec-probe-start",
           "first amplitude tried by the probe ladder")
      .add(sim.bisection_probe_ratio, "bsec-probe-ratio",
           "geometric escalation factor for the probe ladder")
      .add(sim.bisection_relative_tolerance, "bsec-rel-tol",
           "relative target width of the critical Av bracket "
           "(stop when width < max(bsec-tol, bsec-rel-tol * midpoint)); "
           "0 disables (pure absolute)")
      .add(sim.bisection_max_iterations, "bsec-max",
           "maximum midpoint simulations per sample")
      .add(steepness, "steepness", "smooth disk edge steepness")
      .add(sim.domain_half_length, "domain-half-length", "radial domain extent")
      .add(sim.n_refinements, "n-refinements", "global mesh refinements")
      .add(sim.fe_degree, "fe-degree", "finite element degree")
      .add(sim.final_time, "final-time", "simulation final time")
      .add(sim.fast_wave_speed, "fast-wave-speed",
           "if > 0, radial solver computes T = safety*(L/2-R)/C; "
           "domain-half-length supplies L/2")
      .add(sim.finite_time_safety_factor, "finite-time-safety-factor",
           "strictly sub-boundary safety factor in (0,1)")
      .add(sim.time_step, "time-step", "initial time step")
      .add(sim.output_stride, "output-stride", "VTU output stride")
      .add(sim.mesh_adaptation_frequency, "mesh-adaptation-frequency",
           "adapt mesh every N steps")
      .add(sim.max_delta_refinement_level, "max-delta-refinement-level",
           "maximum adaptive refinement levels above base mesh")
      .add(sim.pre_refine_initial_condition, "pre-refine-ic",
           "pre-refine the origin collar to base+max-delta before seeding "
           "the disk, so a coarse n-refinements base does not smear the "
           "initial condition")
      .add(sim.mesh_refinement_fraction, "mesh-refinement-fraction",
           "Kelly refinement fraction")
      .add(sim.mesh_coarsening_fraction, "mesh-coarsening-fraction",
           "Kelly coarsening fraction")
      .add(sim.arkode_minimum_step_size, "arkode-minimum-step-size",
           "ARKode minimum step size")
      .add(sim.arkode_maximum_order, "arkode-maximum-order",
           "ARKode maximum order")
      .add(sim.arkode_maximum_non_linear_iterations,
           "arkode-maximum-nonlinear-iterations",
           "ARKode maximum nonlinear iterations")
      .add(sim.arkode_absolute_tolerance, "arkode-absolute-tolerance",
           "ARKode absolute tolerance")
      .add(sim.arkode_relative_tolerance, "arkode-relative-tolerance",
           "ARKode relative tolerance")
      .add(sim.diagnostic_stride, "diagnostic-stride", "diagnostic stride")
      .add(sim.excitation_threshold, "excitation-threshold",
           "diagnostic excitation threshold")
      .add(sim.diagnosis_start_time, "diagnosis-start-time",
           "ignition diagnosis start time")
      .add(sim.ignition_max_v_threshold, "ignition-max-v-threshold",
           "minimum peak voltage for ignition")
      .add(sim.ignition_excited_fraction_threshold,
           "ignition-excited-fraction-threshold",
           "ignition excited fraction threshold")
      .add(sim.ignition_sustain_window, "ignition-sustain-window",
           "ignition sustain window")
      .add(sim.ignition_area_decay_tolerance,
           "ignition-area-decay-tolerance",
           "ignition area decay tolerance")
      .add(sim.early_exit_on_quench, "early-exit-on-quench",
           "stop a solve once max v stays below the quench threshold for a "
           "full quench window (outcome-preserving, validated by replay)")
      .add(sim.quench_max_v_threshold, "quench-max-v-threshold",
           "domain maximum of v that counts as quenched")
      .add(sim.quench_window, "quench-window",
           "time the field must stay quenched before an early exit")
      .add(sim.early_exit_on_propagation, "early-exit-on-propagation",
           "stop a solve once the detector verdict is sustained; requires "
           "the detector oracle")
      .add(sim.alpha_param, "alpha-param", "FHN alpha parameter")
      .add(sim.diffusion_v, "diffusion-v", "v diffusion")
      .add(sim.diffusion_w, "diffusion-w", "w diffusion")
      .add(sim.beta_param, "beta-param", "FHN beta parameter")
      .add(sim.output_dir, "output-dir", "output directory")
      .add(sim.solver_backend, "solver-backend",
           "solver parameter format: arkode or bdf")
      .add(sim.solver_binary, "solver-binary",
           "path to the selected standalone radial solver executable")
      .add(sim.solver_runner, "solver-runner",
           "launcher prefix; use 'direct' for direct execution")
      .add(sim.segment_wave_width, "segment-wave-width",
           "wave_segment: x-extent of the excited band (solver 'Radius')")
      .add(sim.segment_tail_width, "segment-tail-width",
           "wave_segment: x-extent of the refractory tail behind the band")
      .add(sim.use_detector_oracle, "use-detector-oracle",
           "classify by propagation past a detector station instead of "
           "survival to the final time")
      .add(sim.detector_x, "detector-x",
           "detector station position for the propagation oracle")
      .add(sim.detector_min_excited_area, "detector-min-excited-area",
           "minimum excited area beyond the detector to count as reached")
      .add(sim.detector_require_area_growth, "detector-require-area-growth",
           "hardened detector verdict: v >= the ignition threshold beyond the "
           "station and a non-collapsing, non-decreasing excited area over the "
           "sustain window")
      .add(sim.detector_offset_from_radius, "detector-offset-from-radius",
           "radial detector: place the station at R + offset for each sample "
           "(0 = fixed station at --detector-x)")
      .add(sim.solver_mpi_ranks, "solver-mpi-ranks",
           "MPI ranks for solver launcher; radial solver should use 1");
  }
};

#endif
