/*
 * CAF-based critical-amplitude sweep for the FHN ignition solver.
 *
 * The executable entry point wires the server, client, and worker actors
 * together. Actor implementations live in server.cc, client.cc, and worker.cc.
 */

#include "client.h"
#include "server.h"
#include "worker.h"

#include "caf/io/all.hpp"

#include <cmath>
#include <filesystem>
#include <functional>
#include <sstream>
#include <stdexcept>

using namespace caf;

void caf_main(actor_system &sys, const config &cfg)
{
  scoped_actor self{sys};

  if (cfg.server_mode)
    {
      if (cfg.adaptive_gp && !cfg.adaptive_gp_surface)
        throw std::invalid_argument(
          "--adaptive-gp requires --adaptive-gp-surface");
      if (cfg.gp_probe_readiness_transition)
        {
          const auto positive = [](double value) {
            return std::isfinite(value) && value > 0.0;
          };
          if (!cfg.adaptive_gp || !cfg.adaptive_gp_surface ||
              cfg.gp_probe_every <= 0 || cfg.gp_probe_min_samples <= 0 ||
              !positive(cfg.gp_probe_rms_tolerance) ||
              !positive(cfg.gp_relative_uncertainty_max_tolerance) ||
              !positive(cfg.gp_relative_uncertainty_rms_tolerance) ||
              !positive(cfg.gp_shape_loo_max_tolerance) ||
              !positive(cfg.gp_shape_loo_rms_tolerance))
            throw std::invalid_argument(
              "--gp-probe-readiness-transition requires adaptive surface GP, "
              "probe stopping, and positive finite max/RMS uncertainty and "
              "LOO tolerances");
        }
      if ((cfg.gp_probe_inflight_promotion || cfg.gp_readiness_anticipation) &&
          !cfg.gp_probe_readiness_transition)
        throw std::invalid_argument(
          "--gp-probe-inflight-promotion and --gp-readiness-anticipation "
          "require --gp-probe-readiness-transition");
      if (cfg.gp_readiness_anticipation && !cfg.gp_probe_inflight_promotion)
        throw std::invalid_argument(
          "--gp-readiness-anticipation requires --gp-probe-inflight-promotion");
      if (cfg.gp_async_start_min_valid < 0)
        throw std::invalid_argument("--gp-async-start-min-valid must be >= 0");
      if (cfg.sim.early_exit_on_propagation && !cfg.sim.use_detector_oracle)
        throw std::invalid_argument(
          "--early-exit-on-propagation requires --use-detector-oracle");
      if (cfg.sim.early_exit_on_quench &&
          (!std::isfinite(cfg.sim.quench_max_v_threshold) ||
           cfg.sim.quench_max_v_threshold <= 0.0 ||
           cfg.sim.quench_max_v_threshold >= cfg.sim.ignition_max_v_threshold ||
           !std::isfinite(cfg.sim.quench_window) || cfg.sim.quench_window <= 0.0))
        throw std::invalid_argument(
          "--early-exit-on-quench needs 0 < --quench-max-v-threshold < "
          "--ignition-max-v-threshold and a positive --quench-window");
      if (cfg.sim.solver_backend != "arkode" &&
          cfg.sim.solver_backend != "bdf")
        throw std::invalid_argument(
          "--solver-backend must be either 'arkode' or 'bdf'");
      if (!std::isfinite(cfg.sim.Av_lo) || !std::isfinite(cfg.sim.Av_hi) ||
          cfg.sim.Av_lo >= cfg.sim.Av_hi)
        throw std::invalid_argument("--Av-lo must be smaller than --Av-hi");
      if (!std::isfinite(cfg.sim.bisection_tolerance) ||
          cfg.sim.bisection_tolerance <= 0.0)
        throw std::invalid_argument("--bsec-tol must be positive");
      if (cfg.sim.bisection_max_iterations == 0)
        throw std::invalid_argument("--bsec-max must be at least 1");
      // Client nodes must still have >= 1 worker.
      if (cfg.n_workers < 0 || (cfg.n_workers == 0 && !cfg.server_mode))
        throw std::invalid_argument(
          "--n-workers must be at least 1 (0 is allowed only in --server-mode, "
          "meaning the server node reserves its cores and hosts no workers)");
      if (!cfg.R_values.empty())
        {
          std::istringstream radii(cfg.R_values);
          std::string token;
          int count = 0;
          while (std::getline(radii, token, ','))
            {
              if (token.empty())
                continue;
              const double R = std::stod(token);
              if (!std::isfinite(R) || R <= 0.0)
                throw std::invalid_argument(
                  "--R-values must contain only positive finite radii");
              ++count;
            }
          if (count == 0)
            throw std::invalid_argument(
              "--R-values must contain at least one radius");
        }
      if (cfg.adaptive_gp)
        {
          if (cfg.adaptive_gp_mode != "synchronous" &&
              cfg.adaptive_gp_mode != "asynchronous")
            throw std::invalid_argument(
              "--adaptive-gp-mode must be 'synchronous' or 'asynchronous'");
          if (cfg.adaptive_cancel_running_on_stop &&
              cfg.adaptive_gp_mode != "asynchronous")
            throw std::invalid_argument(
              "--adaptive-cancel-running-on-stop requires asynchronous mode");
          if (cfg.gp_async_early_brackets &&
              cfg.adaptive_gp_mode != "asynchronous")
            throw std::invalid_argument(
              "--gp-async-early-brackets requires asynchronous mode");
          if (cfg.gp_async_early_brackets && !cfg.gp_seeded_brackets)
            throw std::invalid_argument(
              "--gp-async-early-brackets requires --gp-seeded-brackets");
          if (cfg.gp_probe_max_recertifications < 0)
            throw std::invalid_argument(
              "--gp-probe-max-recertifications must be >= 0");
          if (!(cfg.gp_bracket_quench_cost_ratio > 0.0) ||
              !std::isfinite(cfg.gp_bracket_quench_cost_ratio))
            throw std::invalid_argument(
              "--gp-bracket-quench-cost-ratio must be positive");
          if (cfg.adaptive_initial_samples < 2)
            throw std::invalid_argument(
              "--adaptive-initial-samples must be at least 2");
          if (cfg.adaptive_gp_surface && cfg.adaptive_initial_samples < 4)
            throw std::invalid_argument(
              "--adaptive-initial-samples must be at least 4 in surface mode");
          if (cfg.adaptive_batch_size < 1)
            throw std::invalid_argument(
              "--adaptive-batch-size must be at least 1");
          if (cfg.adaptive_rounds < 0)
            throw std::invalid_argument("--adaptive-rounds must be non-negative");
          if (cfg.adaptive_max_samples < 0)
            throw std::invalid_argument(
              "--adaptive-max-samples must be non-negative");
          if (cfg.gp_grid_size < 2)
            throw std::invalid_argument("--gp-grid-size must be at least 2");
          if (cfg.gp_acquisition != "max-std" &&
              cfg.gp_acquisition != "greedy-nipv" &&
              cfg.gp_acquisition != "variation-nipv" &&
              cfg.gp_acquisition != "joint-nipv")
            throw std::invalid_argument(
              "--gp-acquisition must be max-std, greedy-nipv, variation-nipv, "
              "or joint-nipv");
          if (cfg.adaptive_gp_mode == "asynchronous" &&
              cfg.gp_acquisition == "max-std")
            throw std::invalid_argument(
              "asynchronous GP mode requires a pending-aware NIPV acquisition "
              "so active radii can be conditioned as X_pending");
          if (cfg.gp_integration_grid_size < 2)
            throw std::invalid_argument(
              "--gp-integration-grid-size must be at least 2");
          if (!std::isfinite(cfg.gp_variation_weight) ||
              cfg.gp_variation_weight < 0.0 ||
              cfg.gp_variation_weight > 1.0)
            throw std::invalid_argument(
              "--gp-variation-weight must be in [0, 1]");
          if (!std::isfinite(cfg.gp_focus_relative_uncertainty_target) ||
              cfg.gp_focus_relative_uncertainty_target < 0.0)
            throw std::invalid_argument(
              "--gp-focus-relative-uncertainty-target must be non-negative");
          if (!std::isfinite(cfg.gp_focus_variation_coverage) ||
              cfg.gp_focus_variation_coverage <= 0.0 ||
              cfg.gp_focus_variation_coverage > 1.0)
            throw std::invalid_argument(
              "--gp-focus-variation-coverage must be in (0, 1]");
          if (cfg.gp_acquisition_chunk_size < 1)
            throw std::invalid_argument(
              "--gp-acquisition-chunk-size must be at least 1");
          if (cfg.gp_joint_num_restarts < 1)
            throw std::invalid_argument(
              "--gp-joint-num-restarts must be at least 1");
          if (cfg.gp_joint_raw_samples < cfg.gp_joint_num_restarts)
            throw std::invalid_argument(
              "--gp-joint-raw-samples must be at least "
              "--gp-joint-num-restarts");
          if (cfg.gp_joint_maxiter < 1)
            throw std::invalid_argument(
              "--gp-joint-maxiter must be at least 1");
          if (cfg.gp_joint_retries < 1)
            throw std::invalid_argument(
              "--gp-joint-retries must be at least 1");
          if (cfg.gp_joint_fallback != "greedy-nipv" &&
              cfg.gp_joint_fallback != "error")
            throw std::invalid_argument(
              "--gp-joint-fallback must be 'greedy-nipv' or 'error'");
          if (!std::isfinite(cfg.gp_surface_suggest_min_distance) ||
              cfg.gp_surface_suggest_min_distance < 0.0)
            throw std::invalid_argument(
              "--gp-surface-suggest-min-distance must be non-negative");
          if (!std::isfinite(cfg.gp_surface_suggest_min_distance_lengthscales) ||
              cfg.gp_surface_suggest_min_distance_lengthscales < 0.0)
            throw std::invalid_argument(
              "--gp-surface-suggest-min-distance-lengthscales must be "
              "non-negative");
          for (const auto &[name, value] :
               {std::pair<const char *, const std::string &>{
                  "--gp-surface-input-transform-r",
                  cfg.gp_surface_input_transform_R},
                {"--gp-surface-input-transform-gamma",
                 cfg.gp_surface_input_transform_gamma}})
            if (value != "auto" && value != "log" && value != "linear")
              throw std::invalid_argument(
                std::string(name) + " must be auto, log or linear");
          for (const auto &[name, value] :
               {std::pair<const char *, const std::string &>{
                  "--gp-surface-design-ladder-r",
                  cfg.gp_surface_design_ladder_R},
                {"--gp-surface-design-ladder-gamma",
                 cfg.gp_surface_design_ladder_gamma}})
            if (value != "blend" && value != "log" && value != "linear")
              throw std::invalid_argument(
                std::string(name) + " must be blend, log or linear");
          if (!std::isfinite(cfg.gp_surface_interior_margin) ||
              cfg.gp_surface_interior_margin < 0.0 ||
              cfg.gp_surface_interior_margin >= 0.5)
            throw std::invalid_argument(
              "--gp-surface-interior-margin must be in [0, 0.5)");
          if (cfg.gp_surface_candidate_count < 1 ||
              cfg.gp_surface_prediction_r_count < 2 ||
              cfg.gp_surface_prediction_gamma_count < 2)
            throw std::invalid_argument(
              "surface candidate count must be positive and both prediction "
              "grid counts must be at least 2");
          if (!std::isfinite(cfg.gp_surface_boundary_tolerance) ||
              cfg.gp_surface_boundary_tolerance < 0.0 ||
              cfg.gp_surface_boundary_tolerance > 0.5)
            throw std::invalid_argument(
              "--gp-surface-boundary-tolerance must be in [0, 0.5]");
          if (!std::isfinite(cfg.gp_relative_uncertainty_max_tolerance) ||
              cfg.gp_relative_uncertainty_max_tolerance < 0.0)
            throw std::invalid_argument(
              "--gp-relative-uncertainty-max-tolerance must be non-negative");
          if (!std::isfinite(cfg.gp_relative_uncertainty_rms_tolerance) ||
              cfg.gp_relative_uncertainty_rms_tolerance < 0.0)
            throw std::invalid_argument(
              "--gp-relative-uncertainty-rms-tolerance must be non-negative");
          if (!std::isfinite(cfg.gp_shape_loo_max_tolerance) ||
              cfg.gp_shape_loo_max_tolerance < 0.0)
            throw std::invalid_argument(
              "--gp-shape-loo-max-tolerance must be non-negative");
          if (!std::isfinite(cfg.gp_shape_loo_rms_tolerance) ||
              cfg.gp_shape_loo_rms_tolerance < 0.0)
            throw std::invalid_argument(
              "--gp-shape-loo-rms-tolerance must be non-negative");
          if (!std::isfinite(cfg.gp_integrated_variance_tolerance) ||
              cfg.gp_integrated_variance_tolerance < 0.0)
            throw std::invalid_argument(
              "--gp-integrated-variance-tolerance must be non-negative");
          if (!std::isfinite(cfg.gp_variance_min_relative_improvement) ||
              cfg.gp_variance_min_relative_improvement < 0.0 ||
              cfg.gp_variance_min_relative_improvement >= 1.0)
            throw std::invalid_argument(
              "--gp-variance-min-relative-improvement must be in [0, 1)");
          if (cfg.gp_variance_patience < 0)
            throw std::invalid_argument(
              "--gp-variance-patience must be non-negative");
          if (!std::isfinite(cfg.gp_convergence_r_min) ||
              cfg.gp_convergence_r_min < 0.0 ||
              (cfg.gp_convergence_r_min > 0.0 &&
               (cfg.gp_convergence_r_min < cfg.R_min ||
                cfg.gp_convergence_r_min >= cfg.R_max)))
            throw std::invalid_argument(
              "--gp-convergence-r-min must be 0 or lie in [R-min, R-max)");
          if (!std::isfinite(cfg.gp_noise_floor) || cfg.gp_noise_floor <= 0.0)
            throw std::invalid_argument("--gp-noise-floor must be positive");
          if (cfg.gp_target_transform != "log" &&
              cfg.gp_target_transform != "raw")
            throw std::invalid_argument(
              "--gp-target-transform must be either 'log' or 'raw'");
        }
      if (!std::isfinite(cfg.wave_speed_min_reliable_gamma) ||
          cfg.wave_speed_min_reliable_gamma < 0.0 ||
          !std::isfinite(cfg.wave_speed_low_gamma_plateau) ||
          cfg.wave_speed_low_gamma_plateau <= 0.0 ||
          !std::isfinite(cfg.wave_speed_round_up) ||
          cfg.wave_speed_round_up < 0.0)
        throw std::invalid_argument(
          "wave-speed continuation settings must be finite, with a positive "
          "plateau and non-negative limits/rounding");

      auto srv = sys.spawn(server, cfg.port, std::cref(cfg));
      for (int i = 0; i < cfg.n_workers; ++i)
        {
          // Each worker owns a disjoint slice of this node's cores. The server
          // node's budget is short of a full W*M when a core is held back for
          // the scheduler + GP, so the slices may be uneven (8,8,8,7).
          auto sim = cfg.sim;
          const auto slice = worker_core_slice(
            i, cfg.n_workers, cfg.worker_core_budget, cfg.sim.solver_mpi_ranks);
          sim.solver_first_core = slice.first;
          sim.solver_mpi_ranks  = slice.second;
          auto worker = sys.spawn(worker_actor, srv, sim, i);
          anon_mail(worker).send(srv);
        }
      self->println("Server started on port {} with {} local workers",
                    cfg.port,
                    cfg.n_workers);
    }
  else
    {
      sys.spawn(remote_actor, cfg.host, cfg.port, cfg.n_workers,
                cfg.worker_core_budget);
    }
}

CAF_MAIN(caf::io::middleman, caf::id_block::radial_ignition_project)
