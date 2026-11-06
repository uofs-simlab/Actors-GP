#include "server.h"
#include "async_certification.h"
#include "async_certification_readiness.h"
#include "probe_readiness_eligibility.h"
#include "readiness_certification.h"
#include "inflight_probe_promotion.h"
#include "async_early_start.h"
#include "early_brackets.h"
#include "recertification.h"
#include "surface_spacing.h"
#include "sync_certification.h"
#include "wave_speed_model.h"

#include "caf/io/all.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace caf;

namespace
{
struct adaptive_gp_config_t
{
  bool        enabled = false;
  bool        surface = false;
  std::string mode = "synchronous";
  int         batch_size = 0;
  int         initial_samples = 0;
  int         max_rounds = 0;
  int         max_samples = 0;
  bool        cancel_running_on_stop = false;
  // Asynchronous refill: max new samples to request per gamma per scheduler
  // pass. 0 = fill all currently-idle workers, fair-shared across the eligible
  // gammas (each gamma's request is bounded by its round-robin share). A
  // positive value additionally caps how far ahead of the current GP fit a
  // single curve may run speculatively in one burst.
  int         async_fill_cap = 0;
  // Surface mode can instead maintain a fixed pool of unfinished adaptive
  // samples. A completion creates one vacancy, so the next GP fit requests
  // one replacement rather than another full burst.
  int         async_target_active = 0;
  double      R_min = 0.0;
  double      R_max = 0.0;
  double      gamma = 0.0;
  double      gamma_min = 0.0;
  double      gamma_max = 0.0;
  double      steepness = 0.0;
  std::string ic_type;
  double      Av_lo = 0.0;
  double      Av_hi = 0.0;
  std::string gp_python;
  std::string gp_script;
  std::string gp_surface_script;
  bool        gp_persistent_advisor = false;
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
  int         gp_surface_model_reselect_every = 10;
  bool        gp_surface_model_eligibility_refresh = false;
  double      gp_acquisition_value_target = 0.0;
  int         adaptive_min_fit_samples = 0;
  bool        gp_async_early_start = false;
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
  int         gp_stop_hysteresis = 1;
  int         gp_probe_every = 0;
  int         gp_probe_min_samples = 0;
  double      gp_probe_rms_tolerance = 0.0;
  bool        gp_probe_require_readiness = false;
  bool        gp_probe_readiness_transition = false;
  bool        gp_sync_certification_batches = false;
  bool        gp_async_postfit_certification = false;
  bool        gp_async_certification_require_readiness = false;
  int         gp_async_start_min_valid = 0;
  bool        gp_background_probes_off = false;
  bool        gp_probe_inflight_promotion = false;
  bool        gp_readiness_anticipation = false;
  // Async-only early GP brackets (early_brackets.h); the bisection
  // tolerances feed the expected-solve cost rule.
  bool        gp_async_early_brackets = false;
  double      bisection_tolerance = 0.0;
  double      bisection_relative_tolerance = 0.0;
  double      gp_bracket_quench_cost_ratio = 1.0;
  // Bounded re-certification (recertification.h); 0 = original behaviour.
  int         gp_probe_max_recertifications = 0;
  int         gp_prequential_min_samples = 0;
  double      gp_prequential_coverage_lower_bound = 0.0;
  double      gp_prequential_warmup_factor = 0.0;
  double      gp_convergence_r_min = 0.0;
  double      gp_noise_floor = 1e-3;
  std::string gp_target_transform = "log";
  bool        gp_no_plot = false;
  bool        gp_seeded_brackets = false;
  double      gp_seed_sigma_k = 3.0;
  double      Aw = 0.0;
  std::vector<std::pair<double, double>> gamma_wave_speeds;
  double      fallback_wave_speed = 0.0;
  bool        allow_non_positive_wave_speed = false;
  std::shared_ptr<const fhn_fast_wave_speed_model> wave_speed_model;
};

struct gp_suggestion_t
{
  double      R = 0.0;
  double      gamma = std::numeric_limits<double>::quiet_NaN();
  double      posterior_std = 0.0;
  bool        prediction_available = false;
  double      pred_mean = 0.0;
  double      pred_lower95 = 0.0;
  double      pred_upper95 = 0.0;
  std::string requested_method;
  std::string selection_method;
  bool        fallback_used = false;
  int         pending_count = 0;
};

struct gp_metrics_t
{
  double      advisor_runtime_s = 0.0;
  double      R_min = 0.0;
  double      R_max = 0.0;
  double      gamma_min = std::numeric_limits<double>::quiet_NaN();
  double      gamma_max = std::numeric_limits<double>::quiet_NaN();
  int         n_grid = 0;
  int         n_training = 0;
  std::string target_transform;
  std::string input_transform_R;
  std::string input_transform_gamma;
  std::string input_transform_selected_by;
  std::string input_warp;
  // -1 distinguishes an old advisor without the eligibility contract from
  // an advisor reporting no further expansion of its model candidates (0).
  int         input_model_next_selection_n = -1;
  double      loo_log_probability =
    std::numeric_limits<double>::quiet_NaN();
  double      lengthscale_R = std::numeric_limits<double>::quiet_NaN();
  double      lengthscale_gamma = std::numeric_limits<double>::quiet_NaN();
  double      max_model_std = std::numeric_limits<double>::infinity();
  double      rms_model_std = std::numeric_limits<double>::infinity();
  double      mean_model_variance = std::numeric_limits<double>::infinity();
  double      integrated_model_variance =
    std::numeric_limits<double>::infinity();
  double      max_physical_half_width95 =
    std::numeric_limits<double>::infinity();
  double      mean_physical_half_width95 =
    std::numeric_limits<double>::infinity();
  double      max_relative_half_width95 =
    std::numeric_limits<double>::infinity();
  double      rms_relative_half_width95 =
    std::numeric_limits<double>::infinity();
  double      mean_relative_half_width95 =
    std::numeric_limits<double>::infinity();
  int         shape_loo_n = 0;
  double      shape_loo_rms_relative_error =
    std::numeric_limits<double>::infinity();
  double      shape_loo_max_relative_error =
    std::numeric_limits<double>::infinity();
  double      shape_loo_worst_R = 0.0;
  double      shape_loo_worst_gamma = std::numeric_limits<double>::quiet_NaN();
  int         interior_loo_n = 0;
  double      interior_loo_rms_relative_error =
    std::numeric_limits<double>::infinity();
  double      interior_loo_max_relative_error =
    std::numeric_limits<double>::infinity();
  double      interior_loo_worst_R = 0.0;
  double      interior_loo_worst_gamma =
    std::numeric_limits<double>::quiet_NaN();
  bool        lower_boundary_supported = false;
  bool        upper_boundary_supported = false;
  bool        gamma_lower_boundary_supported = false;
  bool        gamma_upper_boundary_supported = false;
  bool        boundary_support_reported = false;
};

struct gp_stop_evaluation_t
{
  bool        stop = false;
  bool        lower_boundary_supported = false;
  bool        upper_boundary_supported = false;
  bool        gamma_lower_boundary_supported = true;
  bool        gamma_upper_boundary_supported = true;
  bool        relative_uncertainty_ok = false;
  bool        shape_resolution_ok = false;
  bool        probe_admission_ready = false;
  bool        integrated_variance_ok = false;
  bool        variance_stagnated = false;
  bool        prequential_calibration_ok = false;
  // Accuracy targets are met but certification is still short of evidence.
  // The caller admits no new work and lets in-flight searches drain: under the
  // certification gate those searches ARE the outstanding evidence, so
  // cancelling them and dispatching replacements only defers the decision.
  bool        admission_frozen = false;
  int         probe_n = 0;
  double      probe_rms_relative_error =
    std::numeric_limits<double>::quiet_NaN();
  bool        probe_accuracy_ok = false;
  int         prequential_n = 0;
  double      prequential_coverage =
    std::numeric_limits<double>::quiet_NaN();
  double      prequential_coverage_lower95 =
    std::numeric_limits<double>::quiet_NaN();
  std::string reason = "continue: stopping criteria not reached";
  double      integrated_variance_relative_reduction =
    std::numeric_limits<double>::quiet_NaN();
};

// One-sided 95% Wilson score lower bound on a binomial proportion.  Preferred
// over the normal approximation because the counts here are small and the
// proportion sits near 1, where the normal interval leaves the unit interval.
// Fold every newly finished prediction into the running certification counts.
// Round-based accumulation cannot work for the asynchronous scheduler: it never
// has a fully completed round when it makes a decision, and its rounds finish
// out of order, so a round-indexed cursor either stalls or skips.  Marking each
// sample as it is counted is correct for both schedulers.
// One row of gp_certification_sets.csv (bounded re-certification audit; the
// file is open only when re-certification is enabled).
void record_certification_set(stateful_actor<server_state> *self,
                              adaptive_gamma_state        &state,
                              const int                    set,
                              const char                  *outcome,
                              const std::vector<double>   &errors,
                              const std::vector<int>      &ids,
                              const char                  *where)
{
  auto &out = state.certification_sets_file;
  if (!out.is_open())
    return;
  double sum_squares = 0.0, largest = 0.0;
  int    largest_id = -1;
  std::string id_list;
  for (std::size_t i = 0; i < errors.size(); ++i)
    {
      sum_squares += errors[i] * errors[i];
      if (errors[i] >= largest)
        {
          largest = errors[i];
          largest_id = i < ids.size() ? ids[i] : -1;
        }
    }
  for (const int id : ids)
    id_list += (id_list.empty() ? "" : ";") + std::to_string(id);
  const double rms = errors.empty()
                       ? std::numeric_limits<double>::quiet_NaN()
                       : std::sqrt(sum_squares / errors.size());
  out << set << "," << outcome << "," << errors.size() << "," << rms << ","
      << largest << "," << largest_id << "," << id_list << ","
      << std::chrono::duration<double>(
           std::chrono::high_resolution_clock::now() - self->state().start_time)
           .count()
      << "," << where << "\n";
  out.flush();
}

void accumulate_prequential(stateful_actor<server_state> *self,
                            adaptive_gamma_state        &state)
{
  for (auto &sample : self->state().samples)
    {
      if (!sample.completed ||
          !sample.gp_prediction_available || !sample.threshold_found ||
          sample.failed || sample.cancelled)
        continue;
      // Admission eligibility is immutable and may now precede the unrelated
      // coverage warm-up. Scoring must not silently reinstate that milestone.
      if (!sample.gp_probe_counted && sample.gp_probe_certification_eligible &&
          sample.critical_Av > 0.0)
        {
          const double error =
            std::abs(sample.gp_pred_mean - sample.critical_Av) /
            sample.critical_Av;
          sample.gp_probe_counted = true;
          // A probe scores only into the certification set it was admitted
          // (or promoted) for; one from an archived set that finishes late is
          // recorded in the archive, never mixed into the current set.
          if (sample.gp_probe_epoch == state.certification_epoch)
            {
              state.probe_relative_errors.push_back(error);
              state.probe_error_sample_ids.push_back(sample.id);
            }
          else
            record_certification_set(self, state, sample.gp_probe_epoch + 1,
                                     "late_probe_archived", {error},
                                     {sample.id}, "score");
        }
      if (sample.gp_prequential_counted ||
          sample.gp_prediction_n_training < state.prequential_warmup_min)
        continue;
      sample.gp_prequential_counted = true;
      ++state.prequential_total;
      if (sample.gp_pred_lower95 <= sample.critical_Av &&
          sample.critical_Av <= sample.gp_pred_upper95)
        ++state.prequential_covered;
    }
}

double wilson_lower_bound_95(const int successes, const int trials)
{
  if (trials <= 0)
    return 0.0;
  constexpr double z = 1.6448536269514722; // one-sided 95%
  const double     p = static_cast<double>(successes) / trials;
  const double     z2 = z * z;
  const double     centre = p + z2 / (2.0 * trials);
  const double     spread =
    z * std::sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials));
  const double     lower = (centre - spread) / (1.0 + z2 / trials);
  return std::max(0.0, std::min(1.0, lower));
}

std::string csv_escape(const std::string &value)
{
  if (value.find_first_of(",\"\n") == std::string::npos)
    return value;

  std::string out = "\"";
  for (const char c : value)
    {
      if (c == '"')
        out += "\"\"";
      else
        out += c;
    }
  out += '"';
  return out;
}

std::string shell_quote(const std::string &value)
{
  std::string quoted = "'";
  for (const char c : value)
    {
      if (c == '\'')
        quoted += "'\\''";
      else
        quoted += c;
    }
  quoted += "'";
  return quoted;
}

std::string double_value(const double value)
{
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string trim(const std::string &value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return "";
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> split_csv_simple(const std::string &line)
{
  std::vector<std::string> fields;
  std::string current;
  bool in_quotes = false;
  for (std::size_t i = 0; i < line.size(); ++i)
    {
      const char c = line[i];
      if (c == '"')
        {
          if (in_quotes && i + 1 < line.size() && line[i + 1] == '"')
            {
              current += '"';
              ++i;
            }
          else
            in_quotes = !in_quotes;
        }
      else if (c == ',' && !in_quotes)
        {
          fields.push_back(trim(current));
          current.clear();
        }
      else
        {
          current += c;
        }
    }
  fields.push_back(trim(current));
  return fields;
}

double linspace_value(const double lo, const double hi, const int n, const int i)
{
  return (n <= 1) ? lo : lo + i * (hi - lo) / (n - 1);
}

double radical_inverse(unsigned int index, const unsigned int base)
{
  double result = 0.0;
  double factor = 1.0 / base;
  while (index > 0)
    {
      result += factor * (index % base);
      index /= base;
      factor /= base;
    }
  return result;
}

// Where one seed-design level lands on an axis.  The seed design is laid out
// before any data exists to say which coordinate the response wants, so
// "blend" -- the geometric mean of the linear and logarithmic ladders -- hedges
// instead of guessing.  It costs about a factor of two against the coordinate
// that turns out to be right, and avoids the factor of six a wrong guess costs.
double ladder_value(const std::string &mode,
                    const double       lower,
                    const double       upper,
                    const double       t)
{
  const double linear = lower + t * (upper - lower);
  if (mode == "linear" || !(lower > 0.0 && upper > 0.0))
    return linear;
  const double logarithmic = lower * std::pow(upper / lower, t);
  if (mode == "log")
    return logarithmic;
  return std::sqrt(linear * logarithmic);
}

// Explicit (R,gamma) rows, for running a fixed hold-out set rather than a
// generated seed design.  Accepts a header naming R (or L) and gamma, or a
// bare two-column R,gamma file.
// A cancelled bisection still holds a verified bracket: the search maintains
// Av_lo < A*_v <= Av_hi at every step, so the midpoint is an honest
// observation rather than an approximation.  The surrogate already weights an
// observation by its bracket half-width, so a coarse bracket is down-weighted
// automatically.  One case is worthless: a bracket whose lower bound is still
// zero means no igniting amplitude has been found yet, and its midpoint says
// nothing about the threshold.
bool salvageable_bracket(const sample_t &sample, const double max_relative)
{
  if (!(max_relative > 0.0) || !sample.completed || !sample.cancelled)
    return false;
  // Both ends must have been observed by solves. A GP-seeded bracket that was
  // cancelled before its ends were verified is the GP's own prediction, not
  // an observation, and must not be fed back as data.
  if (!sample.bracket_ends_observed)
    return false;
  if (!(sample.Av_lo > 0.0) || !(sample.Av_hi > sample.Av_lo))
    return false;
  if (!std::isfinite(sample.Av_lo) || !std::isfinite(sample.Av_hi))
    return false;
  const double mid = 0.5 * (sample.Av_lo + sample.Av_hi);
  const double rel = 0.5 * (sample.Av_hi - sample.Av_lo) / mid;
  return std::isfinite(rel) && rel <= max_relative;
}

std::vector<std::pair<double, double>>
read_surface_sample_list(const std::string &path)
{
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("could not open --gp-surface-sample-list " + path);

  const auto split = [](const std::string &line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string token;
    while (std::getline(stream, token, ','))
      fields.push_back(trim(token));
    return fields;
  };

  std::vector<std::pair<double, double>> out;
  std::string line;
  int  R_column = 0, gamma_column = 1;
  bool first = true;
  while (std::getline(input, line))
    {
      if (!line.empty() && line.back() == '\r')
        line.pop_back();
      if (trim(line).empty())
        continue;
      const auto fields = split(line);
      if (first)
        {
          first = false;
          int found_R = -1, found_gamma = -1;
          for (std::size_t i = 0; i < fields.size(); ++i)
            {
              std::string key = fields[i];
              std::transform(key.begin(), key.end(), key.begin(), ::tolower);
              if (key == "r" || key == "l")
                found_R = static_cast<int>(i);
              else if (key == "gamma")
                found_gamma = static_cast<int>(i);
            }
          if (found_R >= 0 && found_gamma >= 0)
            {
              R_column = found_R;
              gamma_column = found_gamma;
              continue; // consumed a header row
            }
        }
      if (static_cast<int>(fields.size()) <= std::max(R_column, gamma_column))
        continue;
      try
        {
          const double R = std::stod(fields[R_column]);
          const double gamma = std::stod(fields[gamma_column]);
          if (std::isfinite(R) && std::isfinite(gamma) && gamma > 0.0)
            out.emplace_back(R, gamma);
        }
      catch (const std::exception &)
        {
          continue;
        }
    }
  if (out.empty())
    throw std::runtime_error("no (R,gamma) rows parsed from " + path);
  return out;
}

std::vector<std::pair<double, double>>
initial_surface_design(const int         count,
                       const double      R_min,
                       const double      R_max,
                       const double      gamma_min,
                       const double      gamma_max,
                       const std::string &ladder_R,
                       const std::string &ladder_gamma)
{
  if (count < 4)
    throw std::runtime_error(
      "joint GP surface mode requires at least four initial samples");
  if (!(R_min < R_max) || !(0.0 < gamma_min && gamma_min < gamma_max))
    throw std::runtime_error("invalid (R,gamma) surface bounds");

  // Guarantee all corners and, when the budget permits, the midpoint of every
  // edge. Fill the rest by greedily taking the maximin point from a
  // deterministic base-(2,3) Halton pool. This keeps boundary support while
  // distributing the interior points more evenly than taking the first
  // entries of the sequence.
  std::vector<std::pair<double, double>> unit = {
    {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  if (count >= 8)
    unit.insert(unit.end(),
                {{0.0, 0.5}, {1.0, 0.5}, {0.5, 0.0}, {0.5, 1.0}});

  std::vector<std::pair<double, double>> pool;
  const int pool_size = std::max(256, 32 * count);
  pool.reserve(pool_size);
  for (unsigned int i = 5; static_cast<int>(pool.size()) < pool_size; ++i)
    pool.emplace_back(radical_inverse(i, 2), radical_inverse(i, 3));
  while (static_cast<int>(unit.size()) < count)
    {
      auto best = pool.begin();
      double best_min_distance_squared = -1.0;
      for (auto candidate = pool.begin(); candidate != pool.end(); ++candidate)
        {
          double min_distance_squared = std::numeric_limits<double>::infinity();
          for (const auto &[selected_R, selected_gamma] : unit)
            {
              const double dR = candidate->first - selected_R;
              const double dgamma = candidate->second - selected_gamma;
              min_distance_squared =
                std::min(min_distance_squared, dR * dR + dgamma * dgamma);
            }
          if (min_distance_squared > best_min_distance_squared)
            {
              best = candidate;
              best_min_distance_squared = min_distance_squared;
            }
        }
      unit.push_back(*best);
      pool.erase(best);
    }

  std::vector<std::pair<double, double>> result;
  result.reserve(unit.size());
  const auto scale = [](const std::string &mode,
                        const double       coordinate,
                        const double       lower,
                        const double       upper) {
    if (coordinate <= 0.0)
      return lower;
    if (coordinate >= 1.0)
      return upper;
    return ladder_value(mode, lower, upper, coordinate);
  };
  for (const auto &[r_unit, gamma_unit] : unit)
    result.emplace_back(scale(ladder_R, r_unit, R_min, R_max),
                        scale(ladder_gamma, gamma_unit, gamma_min, gamma_max));
  return result;
}

std::vector<double> parse_gamma_values(const std::string &s)
{
  std::vector<double> out;
  if (s.empty())
    return out;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ','))
    if (!token.empty())
      out.push_back(std::stod(token));
  return out;
}

std::vector<std::pair<double, double>>
parse_gamma_wave_speeds(const std::string &s)
{
  std::vector<std::pair<double, double>> out;
  if (trim(s).empty())
    return out;

  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ','))
    {
      token = trim(token);
      if (token.empty())
        continue;
      const auto separator = token.find(':');
      if (separator == std::string::npos)
        throw std::runtime_error(
          "--gamma-wave-speeds entries must use gamma:speed");
      const double gamma = std::stod(trim(token.substr(0, separator)));
      const double speed = std::stod(trim(token.substr(separator + 1)));
      if (!std::isfinite(gamma) || !std::isfinite(speed) || speed <= 0.0)
        throw std::runtime_error(
          "--gamma-wave-speeds requires finite gamma and positive speed");
      const auto duplicate = std::find_if(
        out.begin(), out.end(), [&](const auto &entry) {
          return std::abs(entry.first - gamma) <=
                 1.0e-12 * std::max({1.0, std::abs(entry.first), std::abs(gamma)});
        });
      if (duplicate != out.end())
        throw std::runtime_error(
          "--gamma-wave-speeds contains a duplicate gamma");
      out.emplace_back(gamma, speed);
    }
  return out;
}

double wave_speed_for_gamma(
  const double gamma,
  const std::vector<std::pair<double, double>> &gamma_wave_speeds,
  const fhn_fast_wave_speed_model *continuation,
  const double fallback,
  const bool   allow_non_positive = false)
{
  const auto match = std::find_if(
    gamma_wave_speeds.begin(), gamma_wave_speeds.end(),
    [&](const auto &entry) {
      return std::abs(entry.first - gamma) <=
             1.0e-12 * std::max({1.0, std::abs(entry.first), std::abs(gamma)});
    });
  if (match != gamma_wave_speeds.end())
    return match->second;
  // An explicit table entry wins at its gamma, while the continuation fills
  // the rest of a continuous interval. This lets old discrete campaigns keep
  // their exact validated constants without preventing surface experiments.
  if (continuation != nullptr && !continuation->empty())
    return continuation->speed(gamma);
  if (!gamma_wave_speeds.empty())
    throw std::runtime_error(
      "missing wave speed for gamma=" + double_value(gamma));
  if (!std::isfinite(fallback) || fallback <= 0.0)
    {
      if (allow_non_positive)
        return 0.0;
      throw std::runtime_error(
        "a positive wave speed is required for gamma=" + double_value(gamma) +
        " (or pass --fixed-final-time to use a single observation window)");
    }
  return fallback;
}

double wave_speed_for_gamma(const double               gamma,
                            const adaptive_gp_config_t &cfg)
{
  return wave_speed_for_gamma(
    gamma,
    cfg.gamma_wave_speeds,
    cfg.wave_speed_model.get(),
    cfg.fallback_wave_speed,
    cfg.allow_non_positive_wave_speed);
}

bool same_gamma(const double a, const double b)
{
  return std::abs(a - b) <=
         1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

std::vector<double> parse_R_values(const std::string &s)
{
  std::vector<double> out;
  if (s.empty())
    return out;
  std::istringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ','))
    {
      token = trim(token);
      if (!token.empty())
        out.push_back(std::stod(token));
    }
  return out;
}

sample_t make_sample(const int          id,
                     const double       R,
                     const double       Aw,
                     const double       gamma,
                     const double       fast_wave_speed,
                     const double       steepness,
                     const std::string &ic_type,
                     const double       Av_lo,
                     const double       Av_hi)
{
  sample_t s;
  s.id = id;
  s.R = R;
  s.Aw = Aw;
  s.gamma = gamma;
  s.fast_wave_speed = fast_wave_speed;
  s.steepness = steepness;
  s.ic_type = ic_type;
  s.Av_lo = Av_lo;
  s.Av_hi = Av_hi;
  // Av_lo/Av_hi are overwritten later by the worker's converged bracket; keep
  // the dispatched one so every row records what was actually attempted.
  s.seed_Av_lo = Av_lo;
  s.seed_Av_hi = Av_hi;
  return s;
}

// Narrow a sample's bisection bracket using the GP posterior at its R.
//
// The GP target transform is log, so `posterior_std` is a standard deviation in
// log space and `pred_mean` is the lognormal median exp(mu) (see
// model_to_physical in gp_avstar_surrogate_botorch.py). The safety margin is
// therefore MULTIPLICATIVE -- [median*exp(-k*sigma), median*exp(+k*sigma)] --
// which also guarantees a positive lower end.
std::pair<double, double> seeded_bracket(const adaptive_gp_config_t &cfg,
                                         const gp_suggestion_t      &suggestion)
{
  const std::pair<double, double> global{cfg.Av_lo, cfg.Av_hi};
  if (!cfg.gp_seeded_brackets || !suggestion.prediction_available)
    return global;
  if (cfg.gp_target_transform != "log")
    return global;

  const double median = suggestion.pred_mean;
  const double sigma  = suggestion.posterior_std;
  const double k      = cfg.gp_seed_sigma_k;
  if (!std::isfinite(median) || !std::isfinite(sigma) || median <= 0.0 ||
      sigma < 0.0 || k <= 0.0)
    return global;

  const double spread = std::exp(k * sigma);
  double       lo     = median / spread;
  double       hi     = median * spread;
  if (!std::isfinite(lo) || !std::isfinite(hi))
    return global;

  lo = std::max(lo, cfg.Av_lo);
  hi = std::min(hi, cfg.Av_hi);
  // A collapsed or inverted bracket means the posterior disagrees with the
  // user's global bounds; trust the bounds.
  if (!(hi > lo))
    return global;
  return {lo, hi};
}

void enqueue_sample(stateful_actor<server_state> *self, sample_t sample)
{
  if (sample.id != static_cast<int>(self->state().samples.size()))
    throw std::runtime_error("sample id must match samples vector index");

  self->state().pending_jobs.push_back(
    {sample.id,
     sample.R,
     sample.Aw,
     sample.gamma,
     sample.fast_wave_speed,
     sample.steepness,
     sample.ic_type,
     sample.Av_lo,
     sample.Av_hi,
     sample.Av_lo > 0.0});
  self->state().samples.push_back(std::move(sample));
}

int valid_surface_threshold_count(const server_state &state)
{
  return static_cast<int>(std::count_if(
    state.samples.begin(), state.samples.end(), [](const sample_t &sample) {
      return sample.completed && sample.threshold_found && !sample.failed &&
             !sample.cancelled;
    }));
}

int unfinished_surface_count(const server_state &state)
{
  return static_cast<int>(std::count_if(
    state.samples.begin(), state.samples.end(),
    [](const sample_t &sample) { return !sample.completed; }));
}

bool probe_stop_enabled(const adaptive_gp_config_t &cfg)
{
  return cfg.gp_probe_every > 0 && cfg.gp_probe_min_samples > 0 &&
         cfg.gp_probe_rms_tolerance > 0.0;
}

int probe_eligibility_warmup(const adaptive_gp_config_t &cfg,
                            const adaptive_gamma_state &state)
{
  return cfg.gp_probe_readiness_transition ? 0 : state.prequential_warmup_min;
}

bool certification_readiness_guard_enabled(const adaptive_gp_config_t &cfg)
{
  return async_certification_readiness_guard_active(
    cfg.gp_async_certification_require_readiness, cfg.surface,
    cfg.mode == "asynchronous", cfg.gp_async_postfit_certification,
    probe_stop_enabled(cfg));
}

// Probes are the unbiased stopping evidence, but spacing them at the normal
// background rate after the model is already resolved stretches certification
// across many unnecessary acquisitions.  This reproduces the configured
// non-probe accuracy gate so asynchronous mode can switch its next vacancies
// to probes as soon as that gate is ready to be independently certified.
bool probe_sprint_diagnostics_ready(const gp_stop_evaluation_t &evaluation,
                                    const adaptive_gp_config_t &cfg)
{
  if (cfg.gp_probe_readiness_transition)
    return evaluation.probe_admission_ready;
  const bool boundaries_supported =
    evaluation.lower_boundary_supported &&
    evaluation.upper_boundary_supported &&
    evaluation.gamma_lower_boundary_supported &&
    evaluation.gamma_upper_boundary_supported;
  // Joint-surface designs already include corners. Edge occupancy remains a
  // diagnostic, not a readiness gate, for both surface scheduling policies.
  // Preserve the legacy one-dimensional endpoint requirement.
  if (!cfg.surface && !boundaries_supported)
    return false;

  const bool relative_stop_enabled =
    cfg.gp_relative_uncertainty_max_tolerance > 0.0 ||
    cfg.gp_relative_uncertainty_rms_tolerance > 0.0;
  const bool integrated_variance_enabled =
    cfg.gp_integrated_variance_tolerance > 0.0;
  const bool shape_stop_enabled = cfg.gp_shape_loo_max_tolerance > 0.0 ||
                                  cfg.gp_shape_loo_rms_tolerance > 0.0;

  if (relative_stop_enabled || integrated_variance_enabled)
    return evaluation.shape_resolution_ok &&
           (evaluation.relative_uncertainty_ok ||
            evaluation.integrated_variance_ok);
  return shape_stop_enabled && evaluation.shape_resolution_ok;
}

double probe_rms_relative_error(const adaptive_gamma_state &state)
{
  if (state.probe_relative_errors.empty())
    return std::numeric_limits<double>::quiet_NaN();
  double sum_squares = 0.0;
  for (const double error : state.probe_relative_errors)
    sum_squares += error * error;
  return std::sqrt(sum_squares / state.probe_relative_errors.size());
}

int eligible_unfinished_probe_count(const server_state         &state,
                                    const adaptive_gamma_state &gamma_state,
                                    const adaptive_gp_config_t &cfg)
{
  return static_cast<int>(std::count_if(
    state.samples.begin(), state.samples.end(), [&](const sample_t &sample) {
      return !sample.completed && sample.gp_probe_certification_eligible &&
             sample.gp_probe_epoch == gamma_state.certification_epoch &&
             (cfg.surface || same_gamma(sample.gamma, cfg.gamma));
    }));
}

// Bounded re-certification (recertification.h), both policies. Called
// wherever a completed certification set can be found to miss the target:
// every GP evaluation and the asynchronous drain service. A failed set is
// archived and a fresh set of prospective probes may certify, at most
// --gp-probe-max-recertifications times. Inert unless a completed set failed.
bool archive_failed_certification_set(stateful_actor<server_state> *self,
                                      adaptive_gamma_state        &state,
                                      const adaptive_gp_config_t  &cfg,
                                      const char                  *where)
{
  if (!probe_stop_enabled(cfg))
    return false;
  const int    scored = static_cast<int>(state.probe_relative_errors.size());
  const double rms = probe_rms_relative_error(state);
  if (!certification_set_failed(scored, cfg.gp_probe_min_samples, rms,
                                cfg.gp_probe_rms_tolerance))
    return false;
  double largest = 0.0;
  int    largest_id = -1;
  for (std::size_t i = 0; i < state.probe_relative_errors.size(); ++i)
    if (state.probe_relative_errors[i] >= largest)
      {
        largest = state.probe_relative_errors[i];
        largest_id = i < state.probe_error_sample_ids.size()
                       ? state.probe_error_sample_ids[i] : -1;
      }
  const std::string scope = cfg.surface ? "surface " : "";
  if (!recertification_allowed(state.certification_sets_failed,
                               cfg.gp_probe_max_recertifications))
    {
      if (cfg.gp_probe_max_recertifications > 0 &&
          !state.recertification_exhausted_logged)
        {
          state.recertification_exhausted_logged = true;
          record_certification_set(self, state, state.certification_epoch + 1,
                                   "failed_no_retry_left",
                                   state.probe_relative_errors,
                                   state.probe_error_sample_ids, where);
          self->println(
            "adaptive {}GP: certification set {} failed (probe RMS {:.3f}% "
            "over {}); no re-certification left ({} allowed)",
            scope, state.certification_epoch + 1, 100.0 * rms, scored,
            cfg.gp_probe_max_recertifications);
        }
      return false;
    }
  record_certification_set(self, state, state.certification_epoch + 1,
                           "failed_archived", state.probe_relative_errors,
                           state.probe_error_sample_ids, where);
  self->println(
    "adaptive {}GP: certification set {} failed: probe RMS {:.3f}% over {} "
    "scored probes (largest {:.3f}% at sample {}); archived. A fresh set of {} "
    "prospective probes may certify (re-certification {} of {})",
    scope, state.certification_epoch + 1, 100.0 * rms, scored,
    100.0 * largest, largest_id, cfg.gp_probe_min_samples,
    state.certification_sets_failed + 1, cfg.gp_probe_max_recertifications);
  ++state.certification_sets_failed;
  ++state.certification_epoch;
  state.probe_relative_errors.clear();
  state.probe_error_sample_ids.clear();
  state.probe_certification_sprinting = false;
  state.probe_certification_draining = false;
  state.admission_freeze_announced = false;
  // Readiness anticipation may supply one set's worth of probes per
  // certification set, so a fresh set is treated exactly like the first.
  state.anticipation_probes_admitted = 0;
  return true;
}

// Stop-reason suffix naming the certification set when it is not the first.
std::string certification_set_suffix(const adaptive_gamma_state &state)
{
  if (state.certification_epoch == 0)
    return "";
  return " (certification set " + std::to_string(state.certification_epoch + 1) +
         ", after " + std::to_string(state.certification_sets_failed) +
         " archived)";
}

// Spacing must be measured in the coordinate the model is actually fitted in.
// Measuring R linearly while the surrogate, the acquisition and the probes all
// work in log R makes this check roughly 5x stricter near R_min than anywhere
// else: at R ~ 2 a pair 0.011 apart in normalized log R is only ~0.0026 apart
// linearly, so the advisor legally proposes a point and the coordinator then
// rejects it.  That aborted a 55-sample run whose closest accepted pair was
// 0.039 apart -- nowhere near saturation, with room for thousands more points.
double normalized_surface_distance(const double R_a,
                                   const double gamma_a,
                                   const double R_b,
                                   const double gamma_b,
                                   const adaptive_gp_config_t &cfg,
                                   const bool   log_R,
                                   const bool   log_gamma)
{
  return surface_model_distance(R_a, gamma_a, R_b, gamma_b,
                                cfg.R_min, cfg.R_max, cfg.gamma_min, cfg.gamma_max,
                                log_R, log_gamma);
}

bool has_nearby_attempted_surface_location(
  const server_state          &state,
  const double                 R,
  const double                 gamma,
  const adaptive_gp_config_t  &cfg,
  const bool                   log_R,
  const bool                   log_gamma)
{
  return std::any_of(
    state.samples.begin(), state.samples.end(), [&](const sample_t &sample) {
      return normalized_surface_distance(
               sample.R, sample.gamma, R, gamma, cfg, log_R, log_gamma) <
             cfg.gp_surface_suggest_min_distance;
    });
}

std::vector<gp_suggestion_t>
read_gp_suggestions(const std::filesystem::path &path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("could not open GP suggestions " + path.string());

  std::string header_line;
  if (!std::getline(in, header_line))
    return {};

  std::map<std::string, std::size_t> header;
  const auto header_fields = split_csv_simple(header_line);
  for (std::size_t i = 0; i < header_fields.size(); ++i)
    header[header_fields[i]] = i;

  auto require_column = [&](const std::string &name) -> std::size_t {
    const auto it = header.find(name);
    if (it == header.end())
      throw std::runtime_error("missing column '" + name + "' in " + path.string());
    return it->second;
  };

  const auto R_col = require_column("R");
  const auto gamma_col = header.find("gamma");
  const auto std_it = header.find("posterior_std_in_model_space");
  const auto surface_std_it = header.find("posterior_std");
  if (std_it == header.end() && surface_std_it == header.end())
    throw std::runtime_error(
      "missing posterior standard-deviation column in " + path.string());
  const auto std_col = std_it != header.end()
                         ? std_it->second
                         : surface_std_it->second;
  auto find_either = [&](const std::string &legacy,
                         const std::string &surface) {
    const auto legacy_it = header.find(legacy);
    return legacy_it != header.end() ? legacy_it : header.find(surface);
  };
  const auto mean_col = find_either("Av_star_pred_mean", "pred_mean");
  const auto lower_col = find_either("Av_star_pred_lower95", "pred_lower95");
  const auto upper_col = find_either("Av_star_pred_upper95", "pred_upper95");
  const auto requested_method_col = header.find("requested_method");
  const auto selection_method_col = header.find("selection_method");
  const auto fallback_used_col = header.find("fallback_used");
  const auto pending_count_col = header.find("pending_count");

  std::vector<gp_suggestion_t> suggestions;
  std::string line;
  while (std::getline(in, line))
    {
      if (trim(line).empty())
        continue;
      const auto fields = split_csv_simple(line);
      if (fields.size() <= std::max(R_col, std_col))
        continue;

      gp_suggestion_t suggestion;
      suggestion.R = std::stod(fields[R_col]);
      if (gamma_col != header.end() && fields.size() > gamma_col->second)
        suggestion.gamma = std::stod(fields[gamma_col->second]);
      suggestion.posterior_std = std::stod(fields[std_col]);
      if (requested_method_col != header.end() &&
          fields.size() > requested_method_col->second)
        suggestion.requested_method = fields[requested_method_col->second];
      if (selection_method_col != header.end() &&
          fields.size() > selection_method_col->second)
        suggestion.selection_method = fields[selection_method_col->second];
      if (fallback_used_col != header.end() &&
          fields.size() > fallback_used_col->second)
        {
          const auto value = fields[fallback_used_col->second];
          suggestion.fallback_used = value == "true" || value == "True" ||
                                     value == "1";
        }
      if (pending_count_col != header.end() &&
          fields.size() > pending_count_col->second)
        suggestion.pending_count = std::stoi(fields[pending_count_col->second]);
      if (mean_col != header.end() && lower_col != header.end() &&
          upper_col != header.end() &&
          fields.size() > std::max({mean_col->second,
                                    lower_col->second,
                                    upper_col->second}))
        {
          suggestion.prediction_available = true;
          suggestion.pred_mean = std::stod(fields[mean_col->second]);
          suggestion.pred_lower95 = std::stod(fields[lower_col->second]);
          suggestion.pred_upper95 = std::stod(fields[upper_col->second]);
        }
      suggestions.push_back(suggestion);
    }
  return suggestions;
}

gp_metrics_t
read_gp_metrics(const std::filesystem::path &path)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("could not open GP metrics " + path.string());

  std::string header_line;
  std::string value_line;
  if (!std::getline(in, header_line) || !std::getline(in, value_line))
    throw std::runtime_error("GP metrics are empty in " + path.string());

  std::map<std::string, std::size_t> header;
  const auto header_fields = split_csv_simple(header_line);
  for (std::size_t i = 0; i < header_fields.size(); ++i)
    header[header_fields[i]] = i;
  const auto values = split_csv_simple(value_line);

  auto value = [&](const std::string &name) -> const std::string & {
    const auto it = header.find(name);
    if (it == header.end() || it->second >= values.size())
      throw std::runtime_error("missing column '" + name + "' in " + path.string());
    return values[it->second];
  };
  auto optional_value = [&](const std::string &name) -> const std::string * {
    const auto it = header.find(name);
    if (it == header.end() || it->second >= values.size())
      return nullptr;
    return &values[it->second];
  };
  auto csv_bool = [](const std::string &text) {
    return text == "true" || text == "True" || text == "1";
  };

  gp_metrics_t metrics;
  metrics.R_min = std::stod(value("R_min"));
  metrics.R_max = std::stod(value("R_max"));
  if (const auto entry = optional_value("gamma_min"))
    metrics.gamma_min = std::stod(*entry);
  if (const auto entry = optional_value("gamma_max"))
    metrics.gamma_max = std::stod(*entry);
  metrics.n_grid = std::stoi(value("n_grid"));
  metrics.n_training = std::stoi(value("n_training"));
  metrics.target_transform = value("target_transform");
  if (const auto entry = optional_value("input_transform_R"))
    metrics.input_transform_R = *entry;
  if (const auto entry = optional_value("input_transform_gamma"))
    metrics.input_transform_gamma = *entry;
  if (const auto entry = optional_value("input_transform_selected_by"))
    metrics.input_transform_selected_by = *entry;
  if (const auto entry = optional_value("input_warp"))
    metrics.input_warp = *entry;
  if (const auto entry = optional_value("input_model_next_selection_n"))
    {
      std::size_t parsed = 0;
      metrics.input_model_next_selection_n = std::stoi(*entry, &parsed);
      if (parsed != entry->size())
        throw std::runtime_error("invalid input_model_next_selection_n");
    }
  if (const auto entry = optional_value("loo_log_probability"))
    metrics.loo_log_probability = std::stod(*entry);
  if (const auto entry = optional_value("lengthscale_R"))
    metrics.lengthscale_R = std::stod(*entry);
  if (const auto entry = optional_value("lengthscale_gamma"))
    metrics.lengthscale_gamma = std::stod(*entry);
  metrics.max_model_std = std::stod(value("max_model_std"));
  metrics.rms_model_std = std::stod(value("rms_model_std"));
  metrics.mean_model_variance = std::stod(value("mean_model_variance"));
  metrics.integrated_model_variance =
    std::stod(value("integrated_model_variance"));
  metrics.max_physical_half_width95 =
    std::stod(value("max_physical_half_width95"));
  metrics.mean_physical_half_width95 =
    std::stod(value("mean_physical_half_width95"));
  metrics.max_relative_half_width95 =
    std::stod(value("max_relative_half_width95"));
  metrics.rms_relative_half_width95 =
    std::stod(value("rms_relative_half_width95"));
  metrics.mean_relative_half_width95 =
    std::stod(value("mean_relative_half_width95"));
  metrics.shape_loo_n = std::stoi(value("shape_loo_n"));
  metrics.shape_loo_rms_relative_error =
    std::stod(value("shape_loo_rms_relative_error"));
  metrics.shape_loo_max_relative_error =
    std::stod(value("shape_loo_max_relative_error"));
  metrics.shape_loo_worst_R = std::stod(value("shape_loo_worst_R"));
  if (const auto entry = optional_value("shape_loo_worst_gamma"))
    metrics.shape_loo_worst_gamma = std::stod(*entry);
  if (const auto entry = optional_value("interior_loo_n"))
    metrics.interior_loo_n = std::stoi(*entry);
  if (const auto entry = optional_value("interior_loo_rms_relative_error"))
    metrics.interior_loo_rms_relative_error = std::stod(*entry);
  if (const auto entry = optional_value("interior_loo_max_relative_error"))
    metrics.interior_loo_max_relative_error = std::stod(*entry);
  if (const auto entry = optional_value("interior_loo_worst_R"))
    metrics.interior_loo_worst_R = std::stod(*entry);
  if (const auto entry = optional_value("interior_loo_worst_gamma"))
    metrics.interior_loo_worst_gamma = std::stod(*entry);
  if (const auto entry = optional_value("lower_boundary_supported"))
    {
      metrics.lower_boundary_supported = csv_bool(*entry);
      metrics.boundary_support_reported = true;
    }
  if (const auto entry = optional_value("upper_boundary_supported"))
    {
      metrics.upper_boundary_supported = csv_bool(*entry);
      metrics.boundary_support_reported = true;
    }
  if (const auto entry = optional_value("gamma_lower_boundary_supported"))
    metrics.gamma_lower_boundary_supported = csv_bool(*entry);
  if (const auto entry = optional_value("gamma_upper_boundary_supported"))
    metrics.gamma_upper_boundary_supported = csv_bool(*entry);

  const bool valid =
    metrics.n_grid > 0 && metrics.n_training >= 2 &&
    std::isfinite(metrics.max_model_std) &&
    std::isfinite(metrics.rms_model_std) &&
    std::isfinite(metrics.mean_model_variance) &&
    std::isfinite(metrics.integrated_model_variance) &&
    std::isfinite(metrics.max_physical_half_width95) &&
    std::isfinite(metrics.mean_physical_half_width95) &&
    std::isfinite(metrics.max_relative_half_width95) &&
    std::isfinite(metrics.rms_relative_half_width95) &&
    std::isfinite(metrics.mean_relative_half_width95) &&
    metrics.shape_loo_n >= 0 &&
    std::isfinite(metrics.shape_loo_rms_relative_error) &&
    std::isfinite(metrics.shape_loo_max_relative_error) &&
    std::isfinite(metrics.shape_loo_worst_R);
  if (!valid)
    throw std::runtime_error("invalid GP metrics in " + path.string());
  return metrics;
}

void record_early_bracket(adaptive_gamma_state &surface_state,
                          const int sample_id, const char *where,
                          const int n_training, const double median,
                          const double sigma, const early_bracket_t &bracket,
                          const double gp_solves, const double global_solves,
                          const bool chosen)
{
  auto &out = surface_state.early_bracket_file;
  if (!out.is_open())
    return;
  out << sample_id << "," << where << "," << n_training << "," << median << ","
      << sigma << "," << bracket.lo << "," << bracket.hi << "," << gp_solves
      << "," << global_solves << "," << (chosen ? "true" : "false") << "\n";
  out.flush();
}

// Early GP brackets at dispatch time: a job still carrying the global bracket
// (a queued seed, or an adaptive sample for which the enqueue-time rule chose
// global) gets the GP bracket from the newest forecast when the expected
// solve count is lower (early_brackets.h). Both ends are verified by the
// worker, so this can cost solves but never correctness.
void apply_gp_bracket(stateful_actor<server_state> *self, radial_job_t &job)
{
  auto &st = self->state();
  if (!st.early_brackets_enabled || job.Av_lo != st.global_Av_lo ||
      job.Av_hi != st.global_Av_hi)
    return;
  auto &sample = st.samples.at(job.sample_id);
  if (!sample.gp_bracket_prediction_available)
    return;
  const double median = sample.gp_bracket_pred_mean;
  const double sigma  = sample.gp_bracket_posterior_std;
  const auto bracket = gp_bracket_from_forecast(
    median, sigma, st.early_bracket_sigma_k, st.global_Av_lo, st.global_Av_hi);
  const double rho = st.early_bracket_quench_cost_ratio;
  const double global_cost = global_bracket_cost(
    st.global_Av_lo, st.global_Av_hi, median, st.early_bracket_abs_tol,
    st.early_bracket_rel_tol, rho);
  const double gp_cost = bracket.available
    ? gp_bracket_cost(bracket, median, st.global_Av_lo,
                      st.early_bracket_abs_tol, st.early_bracket_rel_tol, rho)
    : -1.0; // no usable bracket (audit value only)
  const bool chosen = gp_bracket_preferred(
    bracket, median, st.global_Av_lo, st.global_Av_hi,
    st.early_bracket_abs_tol, st.early_bracket_rel_tol, rho);
  record_early_bracket(st.adaptive_surface_state, job.sample_id, "dispatch",
                       sample.gp_bracket_n_training, median, sigma, bracket,
                       gp_cost, global_cost, chosen);
  if (!chosen)
    return;
  job.Av_lo = bracket.lo;
  job.Av_hi = bracket.hi;
  job.verify_lower_endpoint = bracket.lo > st.global_Av_lo;
  sample.seed_Av_lo = bracket.lo;
  sample.seed_Av_hi = bracket.hi;
  sample.gp_seeded_bracket = true;
  sample.bracket_source = "gp";
  ++st.early_brackets_applied;
  self->println(
    "early GP bracket: sample={} (R={:.4f} gamma={:.5f}) -> [{:.6f}, {:.6f}] "
    "median={:.6f} sigma={:.4f} from {} thresholds; expected cost {:.2f} vs "
    "{:.2f} global (igniting-solve units)",
    job.sample_id, job.R, job.gamma, bracket.lo, bracket.hi, median, sigma,
    sample.gp_bracket_n_training, gp_cost, global_cost);
}

// Single dispatch path for both the idle-worker loop and worker registration,
// so every job sees the same dispatch-time bracket rules.
void dispatch_job(stateful_actor<server_state> *self, const actor &worker,
                  radial_job_t job)
{
  ++self->state().active_jobs;
  self->state().worker_active_samples[worker] = job.sample_id;
  apply_gp_bracket(self, job);
  self->anon_send(worker, std::move(job));
}

void wake_idle_workers(stateful_actor<server_state> *self)
{
  if (self->state().termination_initiated)
    return;

  auto it = self->state().idle_workers.begin();
  while (it != self->state().idle_workers.end() &&
         !self->state().pending_jobs.empty())
    {
      auto worker = *it;
      it = self->state().idle_workers.erase(it);
      auto job = self->state().pending_jobs.front();
      self->state().pending_jobs.pop_front();
      dispatch_job(self, worker, std::move(job));
    }
}

// Readiness gate shared by fresh diagnostics and the optional uncertainty
// forecast. Only the two uncertainty inputs may differ between those uses;
// LOO, grid and provenance requirements always come from the current fit.
bool readiness_from_metrics(const gp_metrics_t         &metrics,
                            const adaptive_gp_config_t &cfg,
                            const double                uncertainty_max,
                            const double                uncertainty_rms)
{
  const bool   interior = metrics.interior_loo_n > 0;
  const int    gate_n = interior ? metrics.interior_loo_n : metrics.shape_loo_n;
  const double gate_max = interior ? metrics.interior_loo_max_relative_error
                                   : metrics.shape_loo_max_relative_error;
  const double gate_rms = interior ? metrics.interior_loo_rms_relative_error
                                   : metrics.shape_loo_rms_relative_error;
  return readiness_certification_supported(
    metrics.n_training, metrics.n_grid,
    metrics.shape_loo_n, metrics.shape_loo_max_relative_error,
    metrics.shape_loo_rms_relative_error, gate_n, gate_max, gate_rms,
    uncertainty_max, uncertainty_rms,
    cfg.gp_relative_uncertainty_max_tolerance,
    cfg.gp_relative_uncertainty_rms_tolerance,
    cfg.gp_shape_loo_max_tolerance, cfg.gp_shape_loo_rms_tolerance);
}

struct pending_forecast_t
{
  double R = 0.0;
  double gamma = 0.0;
  double pred_mean = 0.0;
  double pred_lower95 = 0.0;
  double pred_upper95 = 0.0;
  double posterior_std = 0.0;
  int    n_training = 0;
};

struct readiness_forecast_t
{
  int    n_training = 0;
  int    n_pending = 0;
  int    n_grid = 0;
  double max_now = std::numeric_limits<double>::infinity();
  double rms_now = std::numeric_limits<double>::infinity();
  double max_after = std::numeric_limits<double>::infinity();
  double rms_after = std::numeric_limits<double>::infinity();
};

std::vector<std::vector<std::string>>
read_simple_csv_records(const std::filesystem::path &path,
                        const std::vector<std::string> &expected_header)
{
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("could not open " + path.string());
  std::string line;
  if (!std::getline(in, line) || split_csv_simple(trim(line)) != expected_header)
    throw std::runtime_error("unexpected header in " + path.string());
  std::vector<std::vector<std::string>> records;
  while (std::getline(in, line))
    {
      line = trim(line);
      if (line.empty())
        continue;
      auto fields = split_csv_simple(line);
      if (fields.size() != expected_header.size())
        throw std::runtime_error("malformed record in " + path.string());
      records.push_back(std::move(fields));
    }
  return records;
}

std::vector<pending_forecast_t>
read_pending_forecasts(const std::filesystem::path &path)
{
  std::vector<pending_forecast_t> result;
  for (const auto &f : read_simple_csv_records(
         path, {"R", "gamma", "pred_mean", "pred_lower95", "pred_upper95",
                "posterior_std", "n_training"}))
    {
      pending_forecast_t row;
      row.R = std::stod(f[0]);
      row.gamma = std::stod(f[1]);
      row.pred_mean = std::stod(f[2]);
      row.pred_lower95 = std::stod(f[3]);
      row.pred_upper95 = std::stod(f[4]);
      row.posterior_std = std::stod(f[5]);
      row.n_training = std::stoi(f[6]);
      result.push_back(row);
    }
  return result;
}

readiness_forecast_t read_readiness_forecast(const std::filesystem::path &path)
{
  const auto records = read_simple_csv_records(
    path, {"n_training", "n_pending", "n_grid",
           "max_relative_half_width95_now", "rms_relative_half_width95_now",
           "max_relative_half_width95_after_pending",
           "rms_relative_half_width95_after_pending"});
  if (records.size() != 1)
    throw std::runtime_error("expected one readiness forecast in " + path.string());
  const auto &f = records.front();
  readiness_forecast_t row;
  row.n_training = std::stoi(f[0]);
  row.n_pending = std::stoi(f[1]);
  row.n_grid = std::stoi(f[2]);
  row.max_now = std::stod(f[3]);
  row.rms_now = std::stod(f[4]);
  row.max_after = std::stod(f[5]);
  row.rms_after = std::stod(f[6]);
  return row;
}

bool same_location(const double a, const double b)
{
  return std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

gp_stop_evaluation_t
evaluate_gp_stopping(stateful_actor<server_state> *self,
                     adaptive_gamma_state       &gamma_state,
                     const adaptive_gp_config_t  &cfg,
                     const gp_metrics_t          &metrics)
{
  gp_stop_evaluation_t evaluation;

  if (cfg.surface)
    {
      evaluation.lower_boundary_supported =
        metrics.boundary_support_reported && metrics.lower_boundary_supported;
      evaluation.upper_boundary_supported =
        metrics.boundary_support_reported && metrics.upper_boundary_supported;
      evaluation.gamma_lower_boundary_supported =
        metrics.gamma_lower_boundary_supported;
      evaluation.gamma_upper_boundary_supported =
        metrics.gamma_upper_boundary_supported;
    }

  const bool has_new_training_data =
    metrics.n_training > gamma_state.last_gp_fit_valid_threshold_count;
  if (has_new_training_data &&
      std::isfinite(gamma_state.previous_integrated_model_variance) &&
      gamma_state.previous_integrated_model_variance > 0.0)
    {
      evaluation.integrated_variance_relative_reduction =
        (gamma_state.previous_integrated_model_variance -
         metrics.integrated_model_variance) /
        gamma_state.previous_integrated_model_variance;

      if (cfg.gp_variance_patience > 0 &&
          evaluation.integrated_variance_relative_reduction <
            cfg.gp_variance_min_relative_improvement)
        ++gamma_state.variance_stagnation_rounds;
      else
        gamma_state.variance_stagnation_rounds = 0;
    }
  else if (has_new_training_data)
    {
      gamma_state.variance_stagnation_rounds = 0;
    }
  if (has_new_training_data)
    gamma_state.previous_integrated_model_variance =
      metrics.integrated_model_variance;

  const bool max_relative_enabled =
    cfg.gp_relative_uncertainty_max_tolerance > 0.0;
  const bool rms_relative_enabled =
    cfg.gp_relative_uncertainty_rms_tolerance > 0.0;
  const bool relative_stop_enabled =
    max_relative_enabled || rms_relative_enabled;
  const bool max_relative_ok =
    !max_relative_enabled ||
    metrics.max_relative_half_width95 <=
      cfg.gp_relative_uncertainty_max_tolerance;
  const bool rms_relative_ok =
    !rms_relative_enabled ||
    metrics.rms_relative_half_width95 <=
      cfg.gp_relative_uncertainty_rms_tolerance;
  evaluation.relative_uncertainty_ok =
    relative_stop_enabled && max_relative_ok && rms_relative_ok;

  const bool shape_max_enabled = cfg.gp_shape_loo_max_tolerance > 0.0;
  const bool shape_rms_enabled = cfg.gp_shape_loo_rms_tolerance > 0.0;
  const bool shape_stop_guard_enabled =
    shape_max_enabled || shape_rms_enabled;
  // Leave-one-out at a design corner has no surrounding data to interpolate
  // from, so it measures the design rather than the model and would keep an
  // accuracy stop from ever firing.  Gate on the interior residuals whenever
  // the advisor reports them; the 1D advisor does not, and falls back.
  const bool   interior_loo_reported = metrics.interior_loo_n > 0;
  const int    gate_loo_n =
    interior_loo_reported ? metrics.interior_loo_n : metrics.shape_loo_n;
  const double gate_loo_max = interior_loo_reported
                                ? metrics.interior_loo_max_relative_error
                                : metrics.shape_loo_max_relative_error;
  const double gate_loo_rms = interior_loo_reported
                                ? metrics.interior_loo_rms_relative_error
                                : metrics.shape_loo_rms_relative_error;
  const bool shape_max_ok =
    !shape_max_enabled || gate_loo_max <= cfg.gp_shape_loo_max_tolerance;
  const bool shape_rms_ok =
    !shape_rms_enabled || gate_loo_rms <= cfg.gp_shape_loo_rms_tolerance;
  evaluation.shape_resolution_ok =
    !shape_stop_guard_enabled ||
    (gate_loo_n > 0 && shape_max_ok && shape_rms_ok);

  if (cfg.gp_probe_readiness_transition)
    evaluation.probe_admission_ready = readiness_from_metrics(
      metrics, cfg, metrics.max_relative_half_width95,
      metrics.rms_relative_half_width95);

  const bool integrated_variance_enabled =
    cfg.gp_integrated_variance_tolerance > 0.0;
  evaluation.integrated_variance_ok =
    integrated_variance_enabled &&
    metrics.integrated_model_variance <=
      cfg.gp_integrated_variance_tolerance;

  evaluation.variance_stagnated =
    cfg.gp_variance_patience > 0 &&
    gamma_state.variance_stagnation_rounds >= cfg.gp_variance_patience;

  if (cfg.surface)
    self->println(
      "adaptive surface GP metrics: n_train={}, transform={}, R=[{}, {}], "
      "gamma=[{}, {}], max/RMS relative 95% half-width={:.3f}%/{:.3f}%, "
      "integrated model variance={:.6g}, interior GP LOO max/RMS="
      "{:.3f}%/{:.3f}% over n={} (worst R={}, gamma={}), all-point LOO "
      "max/RMS={:.3f}%/{:.3f}%, boundary support R=({}, {}), gamma=({}, {})",
      metrics.n_training,
      metrics.target_transform,
      metrics.R_min,
      metrics.R_max,
      metrics.gamma_min,
      metrics.gamma_max,
      100.0 * metrics.max_relative_half_width95,
      100.0 * metrics.rms_relative_half_width95,
      metrics.integrated_model_variance,
      100.0 * gate_loo_max,
      100.0 * gate_loo_rms,
      gate_loo_n,
      interior_loo_reported ? metrics.interior_loo_worst_R
                            : metrics.shape_loo_worst_R,
      interior_loo_reported ? metrics.interior_loo_worst_gamma
                            : metrics.shape_loo_worst_gamma,
      100.0 * metrics.shape_loo_max_relative_error,
      100.0 * metrics.shape_loo_rms_relative_error,
      evaluation.lower_boundary_supported ? "pass" : "fail",
      evaluation.upper_boundary_supported ? "pass" : "fail",
      evaluation.gamma_lower_boundary_supported ? "pass" : "fail",
      evaluation.gamma_upper_boundary_supported ? "pass" : "fail");

  if (!has_new_training_data)
    self->println(
      "adaptive GP variance progress: unchanged training set; stagnation "
      "counter not updated");
  else if (std::isfinite(evaluation.integrated_variance_relative_reduction))
    self->println(
      "adaptive GP variance progress: relative reduction={:.3f}% "
      "(required={:.3f}%, stagnation {}/{})",
      100.0 * evaluation.integrated_variance_relative_reduction,
      100.0 * cfg.gp_variance_min_relative_improvement,
      gamma_state.variance_stagnation_rounds,
      cfg.gp_variance_patience);
  else
    self->println(
      "adaptive GP variance progress: baseline integrated variance recorded");

  // In joint-surface mode, boundary support is diagnostic only. Do not block
  // prospective scoring or termination on edge occupancy.

  // Prospective calibration.  Every adaptive sample carries the prediction the
  // model made before that sample was run, so scoring them leaks nothing -- and
  // because the acquisition selects the points of largest predictive variance,
  // they are the hardest points available and the estimate errs conservative.
  // Leave-one-out cannot do this job on an acquisition-chosen design: dropping a
  // point leaves a near neighbour behind, so it reads optimistic.
  //
  // This is measured here but applied AFTER the accuracy criteria below.  It
  // gates the STOP, never the sampling: a run that is not yet accurate needs
  // more points regardless, and a run that is accurate but uncertified needs
  // only the evidence already in flight.  Evaluating it first cannot tell those
  // two states apart.
  accumulate_prequential(self, gamma_state);
  // A completed set that missed the target is archived here (both policies),
  // so the probe test below starts a fresh set instead of failing forever.
  archive_failed_certification_set(self, gamma_state, cfg, "fit");
  evaluation.prequential_n = gamma_state.prequential_total;
  if (evaluation.prequential_n > 0)
    {
      evaluation.prequential_coverage =
        static_cast<double>(gamma_state.prequential_covered) /
        evaluation.prequential_n;
      evaluation.prequential_coverage_lower95 = wilson_lower_bound_95(
        gamma_state.prequential_covered, evaluation.prequential_n);
    }
  const bool certification_enabled =
    cfg.gp_prequential_coverage_lower_bound > 0.0;
  const bool certification_short =
    evaluation.prequential_n < cfg.gp_prequential_min_samples;
  const bool certification_failed =
    !certification_short &&
    evaluation.prequential_coverage_lower95 <
      cfg.gp_prequential_coverage_lower_bound;
  evaluation.prequential_calibration_ok =
    !certification_enabled || (!certification_short && !certification_failed);

  // Probe accuracy: the RMS relative error of predictions made, before the
  // corresponding thresholds were computed, at space-filling locations drawn
  // independently of the model.  This is the only quantity here that estimates
  // generalization error without bias.  Leave-one-out is optimistic on an
  // acquisition-chosen design; scoring the acquisition's own picks is worse,
  // because they migrate away from whatever region is already resolved and read
  // 2-5x optimistic; posterior width is a self-assessment.  When probing is
  // configured this measurement IS the accuracy criterion, and the others are
  // recorded as diagnostics only.
  evaluation.probe_n =
    static_cast<int>(gamma_state.probe_relative_errors.size());
  if (evaluation.probe_n > 0)
    {
      evaluation.probe_rms_relative_error =
        probe_rms_relative_error(gamma_state);
    }
  if (cfg.gp_probe_every > 0 && cfg.gp_probe_rms_tolerance > 0.0)
    {
      evaluation.probe_accuracy_ok =
        evaluation.probe_n >= cfg.gp_probe_min_samples &&
        evaluation.probe_rms_relative_error <= cfg.gp_probe_rms_tolerance;
      gamma_state.accuracy_stop_streak = 0;
      if (!evaluation.probe_accuracy_ok)
        {
          // Short of scored probes, but the error on those already scored is
          // inside the target. Freeze only when enough POST-WARMUP probes are
          // actually unfinished. Ordinary acquisitions and pre-warmup probes
          // cannot supply this evidence, so treating every active job as a
          // future probe can deadlock the drain or make it restart late.
          const bool only_count_missing =
            evaluation.probe_n > 0 &&
            evaluation.probe_n < cfg.gp_probe_min_samples &&
            evaluation.probe_rms_relative_error <=
              cfg.gp_probe_rms_tolerance;
          const int eligible_unfinished = eligible_unfinished_probe_count(
            self->state(), gamma_state, cfg);
          if (only_count_missing &&
              async_certification_readiness_permits(
                certification_readiness_guard_enabled(cfg),
                probe_sprint_diagnostics_ready(evaluation, cfg)) &&
              evaluation.probe_n + eligible_unfinished >=
                cfg.gp_probe_min_samples)
            {
              evaluation.admission_frozen = true;
              evaluation.reason =
                "draining: probe RMS within the accuracy target on " +
                std::to_string(evaluation.probe_n) + " of " +
                std::to_string(cfg.gp_probe_min_samples) +
                " scored predictions; admitting no new work while " +
                std::to_string(eligible_unfinished) +
                " eligible in-flight probe(s) supply the rest";
              return evaluation;
            }
          evaluation.reason =
            evaluation.probe_n < cfg.gp_probe_min_samples
              ? "continue: only " + std::to_string(evaluation.probe_n) +
                  " of " + std::to_string(cfg.gp_probe_min_samples) +
                  " space-filling probe predictions scored"
              : "continue: probe RMS relative error is above the accuracy "
                "target";
          return evaluation;
        }
      evaluation.stop = true;
      evaluation.reason =
        "unbiased probe RMS relative error over " +
        std::to_string(evaluation.probe_n) +
        " space-filling predictions is within the accuracy target" +
        certification_set_suffix(gamma_state);
      record_certification_set(self, gamma_state,
                               gamma_state.certification_epoch + 1, "passed",
                               gamma_state.probe_relative_errors,
                               gamma_state.probe_error_sample_ids, "fit");
      return evaluation;
    }

  // Decide whether an accuracy-based stopping criterion is met this round; the
  // decision to actually stop is deferred to the hysteresis check below.
  bool        accuracy_met = false;
  std::string accuracy_reason;

  // Primary accuracy-based stop: when no posterior-uncertainty target is
  // configured, the cross-validated local shape-resolution error is itself the
  // stopping trigger.
  // It is the model's held-out predictive error computed from the collected data alone,
  // small where the curve is flat and large where it bends -- so it does not
  // chase precision on small-valued regions the way a fixed relative
  // posterior-uncertainty target does. Backward compatible: if a relative or
  // integrated-variance target is enabled, shape resolution stays a guard on
  // those triggers and this branch is skipped.
  if (!relative_stop_enabled && !integrated_variance_enabled &&
      shape_stop_guard_enabled && evaluation.shape_resolution_ok)
    {
      accuracy_met = true;
      accuracy_reason =
        "local leave-one-out shape-resolution accuracy target reached";
    }
  else if (evaluation.relative_uncertainty_ok ||
           evaluation.integrated_variance_ok)
    {
      if (!evaluation.shape_resolution_ok)
        {
          gamma_state.accuracy_stop_streak = 0;
          evaluation.reason =
            "continue: GP uncertainty target reached but local "
            "leave-one-out shape resolution is insufficient";
          return evaluation;
        }

      accuracy_met = true;
      accuracy_reason = evaluation.relative_uncertainty_ok
        ? "global physical relative uncertainty and local shape-resolution "
          "tolerances reached"
        : "integrated model-space posterior variance and local "
          "shape-resolution tolerances reached";
    }

  if (accuracy_met && certification_enabled &&
      !evaluation.prequential_calibration_ok)
    {
      // Accurate by every width and shape measure, but the bands are not yet
      // certified.  Two different situations, and they need opposite responses.
      gamma_state.accuracy_stop_streak = 0;
      if (certification_failed)
        {
          // Coverage is measurably below the bound: the posterior understates
          // the error, and the answer to that is more data, so keep admitting.
          evaluation.reason =
            "continue: prospective 95% coverage " +
            std::to_string(gamma_state.prequential_covered) + "/" +
            std::to_string(evaluation.prequential_n) +
            " is below the certification bound; posterior bands understate "
            "the error";
        }
      else if (self->state().active_jobs > 0)
        {
          // Merely short of scored predictions, and searches are already
          // running.  Those searches are the outstanding evidence: admitting
          // replacements for them cannot make the decision arrive sooner, and
          // cancelling them at the stop would discard the very predictions the
          // gate is waiting on.  Drain instead.  The asynchronous scheduler
          // holds W-1 searches in flight, so this is the difference between
          // finishing on work already paid for and dispatching another W-1.
          evaluation.admission_frozen = true;
          evaluation.reason =
            "draining: accuracy targets met, " +
            std::to_string(evaluation.prequential_n) + " of " +
            std::to_string(cfg.gp_prequential_min_samples) +
            " prospective predictions scored; admitting no new work while "
            "in-flight searches finish";
        }
      else
        {
          // Nothing left to drain, so the evidence has to be generated.
          evaluation.reason =
            "continue: only " + std::to_string(evaluation.prequential_n) +
            " of " + std::to_string(cfg.gp_prequential_min_samples) +
            " prospective predictions scored and nothing in flight";
        }
      return evaluation;
    }

  if (accuracy_met)
    {
      // Stopping hysteresis: require the accuracy target to hold for a
      // configurable number of consecutive evaluation rounds before stopping,
      // so a transient dip does not end the run early.
      // This benefits both schedulers and is most impactful for the
      // asynchronous mode, which re-evaluates stopping after every completed
      // sample and is thus the most exposed to a one-round fluctuation.
      ++gamma_state.accuracy_stop_streak;
      const int needed = std::max(1, cfg.gp_stop_hysteresis);
      if (gamma_state.accuracy_stop_streak >= needed)
        {
          evaluation.stop = true;
          evaluation.reason = accuracy_reason;
        }
      else
        {
          evaluation.reason =
            "continue: accuracy target met (" +
            std::to_string(gamma_state.accuracy_stop_streak) + "/" +
            std::to_string(needed) +
            " consecutive rounds); awaiting hysteresis confirmation";
        }
      return evaluation;
    }

  // Accuracy target not met this round: reset the confirmation streak. The
  // variance-stagnation safety stop below has its own patience counter and is
  // intentionally not subject to the accuracy hysteresis.
  gamma_state.accuracy_stop_streak = 0;

  if (evaluation.variance_stagnated)
    {
      evaluation.stop = true;
      evaluation.reason =
        "safety stop: integrated model-space posterior variance stagnated "
        "before the uncertainty target was reached";
      return evaluation;
    }

  evaluation.reason =
    "continue: uncertainty and variance stopping criteria not reached";
  return evaluation;
}

void report_surface_prequential_validation(
  stateful_actor<server_state> *self,
  adaptive_gamma_state        &surface_state,
  const int                     round)
{
  if (round <= 0 || round <= surface_state.last_validation_reported_round)
    return;

  const bool round_unfinished = std::any_of(
    self->state().samples.begin(), self->state().samples.end(),
    [&](const sample_t &sample) {
      return sample.gp_round == round && !sample.completed;
    });
  if (round_unfinished)
    return;

  int    n = 0;
  double sum_sq = 0.0;
  double sum_abs = 0.0;
  double max_abs = 0.0;
  double sum_rel = 0.0;
  double max_rel = 0.0;
  int    coverage = 0;
  for (const auto &sample : self->state().samples)
    {
      if (sample.gp_round != round || !sample.gp_prediction_available ||
          !sample.threshold_found || sample.failed || sample.cancelled)
        continue;
      if (sample.gp_prediction_n_training < surface_state.prequential_warmup_min)
        continue;
      const double error = sample.gp_pred_mean - sample.critical_Av;
      const double absolute = std::abs(error);
      const double relative =
        absolute / std::max(std::abs(sample.critical_Av), 1.0e-300);
      ++n;
      sum_sq += error * error;
      sum_abs += absolute;
      max_abs = std::max(max_abs, absolute);
      sum_rel += relative;
      max_rel = std::max(max_rel, relative);
      if (sample.gp_pred_lower95 <= sample.critical_Av &&
          sample.critical_Av <= sample.gp_pred_upper95)
        ++coverage;
    }

  if (n == 0)
    return;
  surface_state.last_validation_reported_round = round;
  const double coverage_fraction = static_cast<double>(coverage) / n;
  self->println(
    "adaptive surface GP prequential validation round {}: n={}, RMSE={:.6g}, "
    "MAE={:.6g}, max abs={:.6g}, mean rel={:.3f}%, max rel={:.3f}%, "
    "95% coverage={:.1f}% (count-warmed prospective coverage diagnostic)",
    round,
    n,
    std::sqrt(sum_sq / n),
    sum_abs / n,
    max_abs,
    100.0 * sum_rel / n,
    100.0 * max_rel,
    100.0 * coverage_fraction);
  if (surface_state.gp_prequential_file)
    {
      surface_state.gp_prequential_file
        << round << "," << n << "," << std::sqrt(sum_sq / n) << ","
        << sum_abs / n << "," << max_abs << "," << sum_rel / n << ","
        << max_rel << "," << coverage_fraction << "\n";
      surface_state.gp_prequential_file.flush();
    }
}

void record_surface_gp_round_metrics(
  stateful_actor<server_state> *self,
  adaptive_gamma_state        &surface_state,
  const int                     decision_round,
  const adaptive_gp_config_t   &cfg,
  const int                     requested_suggestions,
  const gp_metrics_t           &metrics,
  const gp_stop_evaluation_t   &evaluation)
{
  if (!surface_state.gp_metrics_file)
    return;
  surface_state.gp_metrics_file
    << decision_round << "," << csv_escape(cfg.mode) << ","
    << self->state().total_jobs_completed << "," << self->state().active_jobs
    << "," << self->state().pending_jobs.size() << ","
    << self->state().idle_workers.size() << "," << unfinished_surface_count(self->state())
    << "," << requested_suggestions << "," << metrics.advisor_runtime_s
    << "," << metrics.n_training << ","
    << csv_escape(metrics.target_transform) << ","
    << csv_escape(metrics.input_transform_R) << ","
    << csv_escape(metrics.input_transform_gamma) << ","
    << csv_escape(metrics.input_warp) << ","
    << csv_escape(metrics.input_transform_selected_by) << ","
    << metrics.loo_log_probability << "," << metrics.lengthscale_R << ","
    << metrics.lengthscale_gamma << "," << metrics.R_min << ","
    << metrics.R_max << "," << metrics.gamma_min << "," << metrics.gamma_max
    << "," << metrics.n_grid << "," << metrics.max_model_std << ","
    << metrics.rms_model_std << "," << metrics.mean_model_variance << ","
    << metrics.integrated_model_variance << ","
    << metrics.max_relative_half_width95 << ","
    << metrics.rms_relative_half_width95 << "," << metrics.shape_loo_n << ","
    << metrics.shape_loo_rms_relative_error << ","
    << metrics.shape_loo_max_relative_error << "," << metrics.shape_loo_worst_R
    << "," << metrics.shape_loo_worst_gamma << "," << metrics.interior_loo_n
    << "," << metrics.interior_loo_rms_relative_error << ","
    << metrics.interior_loo_max_relative_error << ","
    << metrics.interior_loo_worst_R << ","
    << metrics.interior_loo_worst_gamma << ",";
  if (std::isfinite(evaluation.integrated_variance_relative_reduction))
    surface_state.gp_metrics_file
      << evaluation.integrated_variance_relative_reduction;
  surface_state.gp_metrics_file
    << "," << surface_state.variance_stagnation_rounds << ","
    << (evaluation.lower_boundary_supported ? "true" : "false") << ","
    << (evaluation.upper_boundary_supported ? "true" : "false") << ","
    << (evaluation.gamma_lower_boundary_supported ? "true" : "false") << ","
    << (evaluation.gamma_upper_boundary_supported ? "true" : "false") << ","
    << (evaluation.relative_uncertainty_ok ? "true" : "false") << ","
    << (evaluation.shape_resolution_ok ? "true" : "false") << ","
    << (evaluation.integrated_variance_ok ? "true" : "false") << ","
    << (evaluation.variance_stagnated ? "true" : "false") << ","
    << evaluation.probe_n << ",";
  if (std::isfinite(evaluation.probe_rms_relative_error))
    surface_state.gp_metrics_file << evaluation.probe_rms_relative_error;
  surface_state.gp_metrics_file
    << "," << (evaluation.probe_accuracy_ok ? "true" : "false") << ","
    << evaluation.prequential_n << ",";
  if (std::isfinite(evaluation.prequential_coverage))
    surface_state.gp_metrics_file << evaluation.prequential_coverage;
  surface_state.gp_metrics_file << ",";
  if (std::isfinite(evaluation.prequential_coverage_lower95))
    surface_state.gp_metrics_file << evaluation.prequential_coverage_lower95;
  surface_state.gp_metrics_file
    << "," << (evaluation.prequential_calibration_ok ? "true" : "false") << ","
    << (evaluation.stop ? "stop" : "continue") << ","
    << csv_escape(evaluation.reason) << ","
    << (cfg.gp_probe_readiness_transition ? "true" : "false") << ","
    << (probe_sprint_diagnostics_ready(evaluation, cfg) ? "true" : "false")
    << "\n";
  surface_state.gp_metrics_file.flush();
}

int run_surface_advisor(stateful_actor<server_state> *self,
                        const adaptive_gp_config_t &cfg,
                        const std::string &command)
{
  if (!cfg.gp_persistent_advisor)
    return std::system(command.c_str());
  try
    {
      auto &advisor = self->state().surface_advisor;
      if (!advisor)
        {
          advisor = std::make_unique<gp_advisor_process>(cfg.gp_python,
                                                         cfg.gp_surface_script);
          self->println("adaptive surface GP: persistent advisor pid={}", advisor->pid());
        }
      return advisor->run(command);
    }
  catch (const std::exception &error)
    {
      self->println("adaptive surface GP: persistent advisor failed: {}", error.what());
      return 1; // Never reuse old files after a failed request or retry silently.
    }
}

// ---------------------------------------------------------------------------
// Asynchronous extensions: early GP brackets and certification at readiness.
// ---------------------------------------------------------------------------

// Store the newest forecast at every unfinished location (queued seeds
// included). Only the bracket fields change: gp_pred_* and a probe's
// certification provenance are frozen at admission and never touched here.
int refresh_bracket_forecasts(stateful_actor<server_state> *self,
                              const std::vector<pending_forecast_t> &forecasts)
{
  int refreshed = 0;
  for (auto &sample : self->state().samples)
    {
      if (sample.completed)
        continue;
      for (const auto &row : forecasts)
        if (same_location(row.R, sample.R) &&
            same_location(row.gamma, sample.gamma))
          {
            sample.gp_bracket_prediction_available = true;
            sample.gp_bracket_pred_mean = row.pred_mean;
            sample.gp_bracket_posterior_std = row.posterior_std;
            sample.gp_bracket_n_training = row.n_training;
            ++refreshed;
            break;
          }
    }
  return refreshed;
}

// Same rule as the asynchronous drive's first decision.
int async_first_fit_threshold(const adaptive_gp_config_t &cfg)
{
  return cfg.gp_async_start_min_valid > 0
      ? std::min(std::max(3, cfg.gp_async_start_min_valid), cfg.initial_samples) :
    cfg.gp_async_early_start ? async_early_start_minimum(cfg.initial_samples) :
    (cfg.mode == "asynchronous" && cfg.adaptive_min_fit_samples > 0)
      ? std::min(cfg.adaptive_min_fit_samples, cfg.initial_samples)
      : cfg.initial_samples;
}

// Fit-only pass while seeds are still queued: no acquisition, no stopping
// evaluation and no round bookkeeping; only the bracket forecasts of the
// unfinished samples are refreshed. Any failure is logged and leaves the
// brackets global.
void fit_surface_for_brackets(stateful_actor<server_state> *self,
                              adaptive_gamma_state        &surface_state,
                              const adaptive_gp_config_t  &cfg)
{
  const int n_valid = valid_surface_threshold_count(self->state());
  const int next_fit = surface_state.bracket_fits_started + 1;
  std::ostringstream prefix;
  prefix << "adaptive_surface_bracketfit_" << std::setw(3) << std::setfill('0')
         << next_fit;
  const auto attempted_path = surface_state.gp_output_dir /
                              (prefix.str() + "_attempted_locations.csv");
  const auto pending_path = surface_state.gp_output_dir /
                            (prefix.str() + "_pending_locations.csv");
  const auto threshold_path = surface_state.gp_output_dir /
                              (prefix.str() + "_threshold_input.csv");
  try
    {
      {
        std::ofstream out(attempted_path.string(), std::ios::out);
        if (!out)
          throw std::runtime_error("could not create " + attempted_path.string());
        out << "R,gamma\n" << std::setprecision(17);
        for (const auto &sample : self->state().samples)
          out << sample.R << "," << sample.gamma << "\n";
      }
      {
        std::ofstream out(pending_path.string(), std::ios::out);
        if (!out)
          throw std::runtime_error("could not create " + pending_path.string());
        out << "R,gamma\n" << std::setprecision(17);
        for (const auto &sample : self->state().samples)
          if (!sample.completed)
            out << sample.R << "," << sample.gamma << "\n";
      }
      {
        std::ofstream out(threshold_path.string(), std::ios::out);
        if (!out)
          throw std::runtime_error("could not create " + threshold_path.string());
        out << "sample_id,R,gamma,critical_Av,Av_lo,Av_hi,bracket_width,status,run_dir\n"
            << std::setprecision(17);
        for (const auto &sample : self->state().samples)
          if (sample.completed && sample.threshold_found && !sample.failed &&
              !sample.cancelled)
            out << sample.id << "," << sample.R << "," << sample.gamma << ","
                << sample.critical_Av << "," << sample.Av_lo << "," << sample.Av_hi
                << "," << sample.Av_hi - sample.Av_lo << ",converged,"
                << csv_escape(sample.run_dir) << "\n";
        for (const auto &sample : self->state().samples)
          if (salvageable_bracket(sample, cfg.gp_salvage_max_relative_half_width))
            out << sample.id << "," << sample.R << "," << sample.gamma << ","
                << 0.5 * (sample.Av_lo + sample.Av_hi) << "," << sample.Av_lo << ","
                << sample.Av_hi << "," << sample.Av_hi - sample.Av_lo
                << ",salvaged_partial," << csv_escape(sample.run_dir) << "\n";
      }
    }
  catch (const std::exception &error)
    {
      self->println("adaptive surface GP: bracket fit skipped: {}", error.what());
      surface_state.last_bracket_fit_valid_count = n_valid;
      return;
    }
  const std::string input_transform_R =
    cfg.gp_surface_input_transform_R == "auto" &&
        !surface_state.selected_input_transform_R.empty()
      ? surface_state.selected_input_transform_R
      : cfg.gp_surface_input_transform_R;
  const std::string input_transform_gamma =
    cfg.gp_surface_input_transform_gamma == "auto" &&
        !surface_state.selected_input_transform_gamma.empty()
      ? surface_state.selected_input_transform_gamma
      : cfg.gp_surface_input_transform_gamma;
  const std::string input_warp = !surface_state.selected_input_warp.empty()
                                   ? surface_state.selected_input_warp
                                   : "auto";
  const std::string command =
    shell_quote(cfg.gp_python) + " " + shell_quote(cfg.gp_surface_script) +
    " " + shell_quote(threshold_path.string()) + " --out-dir " +
    shell_quote(surface_state.gp_output_dir.string()) + " --prefix " +
    shell_quote(prefix.str()) + " --r-min " + double_value(cfg.R_min) +
    " --r-max " + double_value(cfg.R_max) + " --gamma-min " +
    double_value(cfg.gamma_min) + " --gamma-max " +
    double_value(cfg.gamma_max) + " --target-transform " +
    shell_quote(cfg.gp_target_transform) + " --noise-floor " +
    double_value(cfg.gp_noise_floor) + " --prediction-r-count " +
    std::to_string(cfg.gp_surface_prediction_r_count) +
    " --prediction-gamma-count " +
    std::to_string(cfg.gp_surface_prediction_gamma_count) +
    " --integration-count " + std::to_string(cfg.gp_integration_grid_size) +
    " --candidate-count 1 --nipv-backend " +
    shell_quote(cfg.gp_surface_nipv_backend) + " --batch-size 1" +
    " --acquisition " + shell_quote(cfg.gp_acquisition) +
    " --variation-weight " + double_value(cfg.gp_variation_weight) +
    " --acquisition-chunk-size " +
    std::to_string(cfg.gp_acquisition_chunk_size) +
    " --acquisition-value-target 0 --stop-max-relative-half-width 0"
    " --stop-rms-relative-half-width 0 --input-transform-r " +
    shell_quote(input_transform_R) + " --input-transform-gamma " +
    shell_quote(input_transform_gamma) + " --input-warp " +
    shell_quote(input_warp) + " --interior-margin-normalized " +
    double_value(cfg.gp_surface_interior_margin) +
    " --suggest-min-distance-lengthscales " +
    double_value(cfg.gp_surface_suggest_min_distance_lengthscales) +
    " --suggest-min-distance-normalized " +
    double_value(cfg.gp_surface_suggest_min_distance) +
    " --boundary-tolerance-normalized " +
    double_value(cfg.gp_surface_boundary_tolerance) +
    " --attempted-locations-csv " + shell_quote(attempted_path.string()) +
    " --pending-locations-csv " + shell_quote(pending_path.string()) +
    " --seed " + std::to_string(104729 + 31 * next_fit) +
    (cfg.gp_no_plot ? " --no-plot" : "") + " --defer-acquisition";
  self->println(
    "adaptive surface GP: bracket fit {} at {} valid thresholds ({} queued, {} "
    "unfinished); application elapsed {:.3f}s",
    next_fit, n_valid, self->state().pending_jobs.size(),
    unfinished_surface_count(self->state()),
    std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                  self->state().start_time).count());
  const auto advisor_start = std::chrono::steady_clock::now();
  const int rc = run_surface_advisor(self, cfg, command);
  const double advisor_runtime_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - advisor_start).count();
  // The deferred model is never resumed: release it and its checkpoint.
  if (self->state().surface_advisor)
    self->state().surface_advisor->discard();
  {
    std::error_code error;
    std::filesystem::remove(
      surface_state.gp_output_dir / (prefix.str() + "_acquisition.pt"), error);
  }
  surface_state.bracket_fits_started = next_fit;
  surface_state.last_bracket_fit_valid_count = n_valid;
  if (rc != 0)
    {
      self->println("adaptive surface GP: bracket fit {} failed with return "
                    "code {}; brackets stay global until the next fit",
                    next_fit, rc);
      return;
    }
  int refreshed = 0;
  try
    {
      refreshed = refresh_bracket_forecasts(
        self, read_pending_forecasts(surface_state.gp_output_dir /
                                     (prefix.str() + "_pending_predictions.csv")));
    }
  catch (const std::exception &error)
    {
      self->println("adaptive surface GP: bracket fit {}: pending forecasts "
                    "unavailable ({}); keeping previous forecasts",
                    next_fit, error.what());
      return;
    }
  self->println("adaptive surface GP: bracket fit {} done in {:.3f}s; forecasts "
                "refreshed for {} unfinished sample(s)",
                next_fit, advisor_runtime_s, refreshed);
}

// Called on every result before the freed worker takes the next queued job.
void maybe_fit_surface_for_brackets(stateful_actor<server_state> *self,
                                    const adaptive_gp_config_t  &cfg)
{
  if (!cfg.enabled || !cfg.surface || !cfg.gp_async_early_brackets ||
      cfg.mode != "asynchronous")
    return;
  auto &st = self->state();
  auto &surface_state = st.adaptive_surface_state;
  if (st.termination_initiated || surface_state.generation_stopped ||
      st.pending_jobs.empty() ||
      st.pending_jobs.front().sample_id >= cfg.initial_samples)
    return;
  const int n_valid = valid_surface_threshold_count(st);
  if (n_valid < std::max(3, async_first_fit_threshold(cfg)) ||
      n_valid <= surface_state.last_bracket_fit_valid_count)
    return;
  fit_surface_for_brackets(self, surface_state, cfg);
}

bool start_adaptive_gp_surface_round(
  stateful_actor<server_state> *self,
  adaptive_gamma_state        &surface_state,
  const adaptive_gp_config_t  &cfg,
  const int                     requested_capacity)
{
  const int n_attempted = static_cast<int>(self->state().samples.size());
  const int n_valid = valid_surface_threshold_count(self->state());
  if (n_valid < 3)
    {
      surface_state.stop_reason =
        "fewer than three valid ignition thresholds are available";
      self->println("adaptive surface GP: stopping because {}",
                    surface_state.stop_reason);
      return false;
    }

  const bool sample_budget_exhausted =
    cfg.max_samples > 0 && n_attempted >= cfg.max_samples;
  const bool update_budget_exhausted =
    surface_state.rounds_started >= cfg.max_rounds;
  const int next_round = surface_state.rounds_started + 1;
  const int next_fit = surface_state.fits_started + 1;
  std::filesystem::create_directories(surface_state.gp_output_dir);
  std::ostringstream prefix;
  prefix << "adaptive_surface_round_" << std::setw(3) << std::setfill('0')
         << next_fit;

  const auto attempted_path = surface_state.gp_output_dir /
                              (prefix.str() + "_attempted_locations.csv");
  const auto pending_path = surface_state.gp_output_dir /
                            (prefix.str() + "_pending_locations.csv");
  const auto threshold_path = surface_state.gp_output_dir /
                              (prefix.str() + "_threshold_input.csv");
  {
    std::ofstream out(attempted_path.string(), std::ios::out);
    if (!out)
      throw std::runtime_error("could not create " + attempted_path.string());
    out << "R,gamma\n" << std::setprecision(17);
    for (const auto &sample : self->state().samples)
      out << sample.R << "," << sample.gamma << "\n";
  }
  {
    std::ofstream out(pending_path.string(), std::ios::out);
    if (!out)
      throw std::runtime_error("could not create " + pending_path.string());
    out << "R,gamma\n" << std::setprecision(17);
    for (const auto &sample : self->state().samples)
      if (!sample.completed)
        out << sample.R << "," << sample.gamma << "\n";
  }
  {
    std::ofstream out(threshold_path.string(), std::ios::out);
    if (!out)
      throw std::runtime_error("could not create " + threshold_path.string());
    out << "sample_id,R,gamma,critical_Av,Av_lo,Av_hi,bracket_width,status,run_dir\n"
        << std::setprecision(17);
    for (const auto &sample : self->state().samples)
      if (sample.completed && sample.threshold_found && !sample.failed &&
          !sample.cancelled)
        out << sample.id << "," << sample.R << "," << sample.gamma << ","
            << sample.critical_Av << "," << sample.Av_lo << "," << sample.Av_hi
            << "," << sample.Av_hi - sample.Av_lo << ",converged,"
            << csv_escape(sample.run_dir) << "\n";
    for (const auto &sample : self->state().samples)
      if (salvageable_bracket(sample, cfg.gp_salvage_max_relative_half_width))
        out << sample.id << "," << sample.R << "," << sample.gamma << ","
            << 0.5 * (sample.Av_lo + sample.Av_hi) << "," << sample.Av_lo << ","
            << sample.Av_hi << "," << sample.Av_hi - sample.Av_lo
            << ",salvaged_partial," << csv_escape(sample.run_dir) << "\n";
  }
  const int remaining_budget = cfg.max_samples > 0
                                 ? std::max(0, cfg.max_samples - n_attempted)
                                 : requested_capacity;
  const int adaptive_dispatched = std::max(0, n_attempted - cfg.initial_samples);
  const int requested_suggestions =
    sample_budget_exhausted || update_budget_exhausted
      ? 0
      : std::min(requested_capacity, remaining_budget);

  // Bootstrap the idle capacity;
  // only then can each completion create exactly one vacancy.  Applying the
  // value-based truncation to this first refill leaves most workers idle and
  // takes several expensive fits merely to reach steady state.
  const bool asynchronous_bootstrap =
    cfg.mode == "asynchronous" && surface_state.rounds_started == 0 &&
    (cfg.gp_async_early_start || unfinished_surface_count(self->state()) == 0);

  int requested_probes = 0;
  if (cfg.gp_probe_every > 0 && !cfg.gp_background_probes_off)
    requested_probes =
      ((adaptive_dispatched + requested_suggestions) / cfg.gp_probe_every) -
      (adaptive_dispatched / cfg.gp_probe_every);

  // During a certification sprint, use every vacancy for exactly the missing
  // probe evidence.  The flag is latched by the first fit whose ordinary
  // uncertainty/shape/boundary diagnostics pass.  Retain the older scored-
  // probe trigger as a fallback for runs whose diagnostics are disabled.
  if (cfg.mode == "asynchronous" && probe_stop_enabled(cfg) &&
      !cfg.gp_async_postfit_certification)
    {
      accumulate_prequential(self, surface_state);
      const int scored =
        static_cast<int>(surface_state.probe_relative_errors.size());
      const double rms = probe_rms_relative_error(surface_state);
      const int eligible_unfinished = eligible_unfinished_probe_count(
        self->state(), surface_state, cfg);
      const bool scored_probe_trigger =
        scored > 0 && scored < cfg.gp_probe_min_samples &&
        std::isfinite(rms) && rms <= cfg.gp_probe_rms_tolerance;
      if (surface_state.probe_certification_sprinting ||
          scored_probe_trigger)
        requested_probes = std::max(
          requested_probes,
          std::min(requested_suggestions,
                   std::max(0, cfg.gp_probe_min_samples - scored -
                                 eligible_unfinished)));
    }

  // Coordinate and warp selection fits up to five alternative GPs. In async
  // mode retain the last structural choice for the configured interval.
  // Hyperparameters of the selected model are still refitted on every
  // decision. The optional eligibility refresh uses a threshold supplied by
  // the advisor, which owns the candidate-availability rules. No training
  // thresholds are duplicated here. Interval 0 still locks the first choice.
  const bool eligibility_refresh_enabled =
    cfg.mode == "asynchronous" &&
    cfg.gp_surface_model_eligibility_refresh &&
    cfg.gp_surface_model_reselect_every > 0;
  const bool eligibility_refresh_due =
    eligibility_refresh_enabled &&
    surface_state.next_input_model_selection_training_count > 0 &&
    n_valid >= surface_state.next_input_model_selection_training_count;
  const bool reselect_input_model =
    cfg.mode != "asynchronous" ||
    surface_state.selected_input_transform_R.empty() ||
    eligibility_refresh_due ||
    (cfg.gp_surface_model_reselect_every > 0 &&
     n_valid - surface_state.last_input_model_selection_training_count >=
       cfg.gp_surface_model_reselect_every);
  if (eligibility_refresh_due)
    self->println(
      "adaptive surface GP: model eligibility refresh at {} valid thresholds "
      "(new options available from {})",
      n_valid, surface_state.next_input_model_selection_training_count);
  const std::string input_transform_R =
    !reselect_input_model && cfg.gp_surface_input_transform_R == "auto"
      ? surface_state.selected_input_transform_R
      : cfg.gp_surface_input_transform_R;
  const std::string input_transform_gamma =
    !reselect_input_model && cfg.gp_surface_input_transform_gamma == "auto"
      ? surface_state.selected_input_transform_gamma
      : cfg.gp_surface_input_transform_gamma;
  const std::string input_warp =
    !reselect_input_model && !surface_state.selected_input_warp.empty()
      ? surface_state.selected_input_warp
      : "auto";
  std::string command =
    shell_quote(cfg.gp_python) + " " + shell_quote(cfg.gp_surface_script) +
    " " + shell_quote(threshold_path.string()) + " --out-dir " +
    shell_quote(surface_state.gp_output_dir.string()) + " --prefix " +
    shell_quote(prefix.str()) + " --r-min " + double_value(cfg.R_min) +
    " --r-max " + double_value(cfg.R_max) + " --gamma-min " +
    double_value(cfg.gamma_min) + " --gamma-max " +
    double_value(cfg.gamma_max) + " --target-transform " +
    shell_quote(cfg.gp_target_transform) + " --noise-floor " +
    double_value(cfg.gp_noise_floor) + " --prediction-r-count " +
    std::to_string(cfg.gp_surface_prediction_r_count) +
    " --prediction-gamma-count " +
    std::to_string(cfg.gp_surface_prediction_gamma_count) +
    " --integration-count " + std::to_string(cfg.gp_integration_grid_size) +
    " --candidate-count " + std::to_string(cfg.gp_surface_candidate_count) +
    " --nipv-backend " + shell_quote(cfg.gp_surface_nipv_backend) +
    " --batch-size " + std::to_string(requested_suggestions) +
    " --acquisition " + shell_quote(cfg.gp_acquisition) +
    " --variation-weight " + double_value(cfg.gp_variation_weight) +
    " --acquisition-chunk-size " +
    std::to_string(cfg.gp_acquisition_chunk_size) +
    // Value-based truncation remains Async-only. Ordinary Sync acquisition
    // requests a full batch; the opt-in certification rule below may instead
    // request a smaller probe-only batch after inspecting fresh diagnostics.
    " --acquisition-value-target " +
    double_value(cfg.mode == "asynchronous" && !asynchronous_bootstrap
                   ? cfg.gp_acquisition_value_target
                   : 0.0) +
    // Give the advisor the run's own accuracy gate so it can stop dispatching
    // work that will be cancelled. Sync does not use this look-ahead check;
    // its certification decision belongs to the coordinator, after the fit.
    " --stop-max-relative-half-width " +
    double_value(cfg.mode == "asynchronous" && !asynchronous_bootstrap
                   ? cfg.gp_relative_uncertainty_max_tolerance
                   : 0.0) +
    " --stop-rms-relative-half-width " +
    double_value(cfg.mode == "asynchronous" && !asynchronous_bootstrap
                   ? cfg.gp_relative_uncertainty_rms_tolerance
                   : 0.0) +
    " --input-transform-r " +
    shell_quote(input_transform_R) +
    " --input-transform-gamma " +
    shell_quote(input_transform_gamma) +
    " --input-warp " + shell_quote(input_warp) +
    " --interior-margin-normalized " +
    double_value(cfg.gp_surface_interior_margin) +
    " --suggest-min-distance-lengthscales " +
    double_value(cfg.gp_surface_suggest_min_distance_lengthscales) +
    // Probe RATE is per dispatched sample, not per round.  A fixed count per
    // round would give the synchronous scheduler one probe per 20 points and
    // the asynchronous one a probe per point, so the two schedulers would be
    // measured against different designs and the comparison would be void.
    (cfg.gp_probe_every > 0
       ? " --probe-count " + std::to_string(requested_probes) +
           " --probe-index " + std::to_string(surface_state.probe_index)
       : std::string()) +
    " --suggest-min-distance-normalized " +
    double_value(cfg.gp_surface_suggest_min_distance) +
    " --boundary-tolerance-normalized " +
    double_value(cfg.gp_surface_boundary_tolerance) +
    " --attempted-locations-csv " + shell_quote(attempted_path.string()) +
    " --pending-locations-csv " + shell_quote(pending_path.string()) +
    " --seed " + std::to_string(7919 * next_fit + 17);
  if (cfg.gp_no_plot)
    command += " --no-plot";

  self->state().results_file.flush();
  self->println(
    "adaptive surface GP: round {}, {} valid thresholds, requesting {} "
    "point(s), {} pending",
    next_fit,
    n_valid,
    requested_suggestions,
    unfinished_surface_count(self->state()));
  // Keep the fitted model alive in a private checkpoint until the coordinator
  // has evaluated its full stopping rule. Neither policy may select an unused
  // batch on its terminal decision. The continuation reuses the exact model.
  const auto checkpoint_path = surface_state.gp_output_dir /
                               (prefix.str() + "_acquisition.pt");
  struct checkpoint_cleanup_t
  {
    std::filesystem::path path;
    std::unique_ptr<gp_advisor_process> &advisor;
    ~checkpoint_cleanup_t()
    {
      if (advisor) advisor->discard();
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } checkpoint_cleanup{checkpoint_path, self->state().surface_advisor};
  const std::string fit_command = command + " --defer-acquisition";
  self->println("adaptive surface GP command: {}", fit_command);
  const auto advisor_start = std::chrono::steady_clock::now();
  const int rc = run_surface_advisor(self, cfg, fit_command);
  const double advisor_runtime_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - advisor_start).count();
  self->println("adaptive surface GP: fit/diagnostics completed in {:.3f}s",
                advisor_runtime_s);
  if (rc != 0)
    {
      surface_state.stop_reason = "surface GP advisor command failed";
      self->println("adaptive surface GP: command failed with return code {}",
                    rc);
      return false;
    }
  surface_state.fits_started = next_fit;

  gp_metrics_t metrics;
  try
    {
      metrics = read_gp_metrics(
        surface_state.gp_output_dir / (prefix.str() + "_metrics.csv"));
    }
  catch (const std::exception &error)
    {
      surface_state.stop_reason = "surface GP metrics could not be read";
      self->println("adaptive surface GP: {}", error.what());
      return false;
    }
  metrics.advisor_runtime_s = advisor_runtime_s;
  if (eligibility_refresh_enabled && reselect_input_model &&
      (metrics.input_model_next_selection_n < 0 ||
       (metrics.input_model_next_selection_n > 0 &&
        metrics.input_model_next_selection_n <= metrics.n_training)))
    {
      surface_state.stop_reason = "invalid or missing model eligibility metadata";
      self->println(
        "adaptive surface GP: invalid or missing model eligibility metadata; "
        "expected input_model_next_selection_n=0 or a future training count");
      return false;
    }
  surface_state.model_log_R = metrics.input_transform_R == "log";
  if (reselect_input_model)
    {
      surface_state.selected_input_transform_R = metrics.input_transform_R;
      surface_state.selected_input_transform_gamma =
        metrics.input_transform_gamma;
      surface_state.selected_input_warp = metrics.input_warp;
      surface_state.last_input_model_selection_training_count = n_valid;
      surface_state.next_input_model_selection_training_count =
        eligibility_refresh_enabled ? metrics.input_model_next_selection_n : 0;
    }
  gp_stop_evaluation_t evaluation =
    evaluate_gp_stopping(self, surface_state, cfg, metrics);
  const bool readiness_guard = certification_readiness_guard_enabled(cfg);
  const bool fresh_readiness = probe_sprint_diagnostics_ready(evaluation, cfg);
  if (!async_certification_readiness_permits(readiness_guard, fresh_readiness))
    {
      // A small error on earlier probes, or readiness at an earlier fit, must
      // not turn a presently unresolved model into a probe-only/drain policy.
      surface_state.probe_certification_sprinting = false;
      surface_state.probe_certification_draining = false;
      surface_state.admission_freeze_announced = false;
      if (!evaluation.stop)
        self->println(
          "adaptive surface GP: Async readiness guard: current diagnostics "
          "not ready; probe-only admissions and certification drains "
          "disabled for this fit");
    }
  if (cfg.mode == "asynchronous" && probe_stop_enabled(cfg) &&
      metrics.n_training >= probe_eligibility_warmup(cfg, surface_state) &&
      evaluation.probe_n < cfg.gp_probe_min_samples &&
      !evaluation.admission_frozen &&
      probe_sprint_diagnostics_ready(evaluation, cfg))
    {
      const int eligible_unfinished = eligible_unfinished_probe_count(
        self->state(), surface_state, cfg);
      if (!surface_state.probe_certification_sprinting)
        self->println(
          "adaptive surface GP: accuracy diagnostics passed; starting "
          "certification sprint ({} scored, {} eligible in flight, {} "
          "required)",
          evaluation.probe_n,
          eligible_unfinished,
          cfg.gp_probe_min_samples);
      surface_state.probe_certification_sprinting = true;
      evaluation.reason =
        "certification sprint: accuracy diagnostics passed; assigning "
        "upcoming vacancies to probes (" +
        std::to_string(evaluation.probe_n) + " scored, " +
        std::to_string(eligible_unfinished) + " eligible in flight, " +
        std::to_string(cfg.gp_probe_min_samples) + " required)";
    }
  const auto record_decision = [&](const int acquired_capacity) {
    auto recorded_evaluation = evaluation;
    if (cfg.gp_probe_readiness_transition)
      recorded_evaluation.reason +=
        "; readiness-driven probe eligibility: " +
        std::string(fresh_readiness ? "ready" : "not ready") +
        "; no count-based probe warm-up";
    if (readiness_guard)
      recorded_evaluation.reason +=
        std::string("; Async readiness guard: current diagnostics ") +
        (fresh_readiness ? "ready" : "not ready");
    record_surface_gp_round_metrics(self, surface_state, next_fit, cfg,
                                    acquired_capacity, metrics,
                                    recorded_evaluation);
  };
  surface_state.last_gp_fit_valid_threshold_count = n_valid;
  if (evaluation.stop)
    {
      record_decision(0);
      self->println("adaptive surface GP: acquisition skipped (stop)");
      surface_state.probe_certification_sprinting = false;
      surface_state.stop_reason = evaluation.reason;
      surface_state.stop_criterion_reached = true;
      self->println("adaptive surface GP: stopping because {}",
                    surface_state.stop_reason);
      return false;
    }
  // Early GP brackets: refresh the bracket forecasts of every unfinished
  // sample (queued seeds included) from THIS fit; a failure keeps the
  // previous forecasts and never stops the run.
  if (cfg.gp_async_early_brackets && cfg.mode == "asynchronous")
    {
      try
        {
          const int refreshed = refresh_bracket_forecasts(
            self, read_pending_forecasts(surface_state.gp_output_dir /
                                         (prefix.str() + "_pending_predictions.csv")));
          self->println("adaptive surface GP: early brackets: forecasts refreshed "
                        "for {} unfinished sample(s)", refreshed);
        }
      catch (const std::exception &error)
        {
          self->println("adaptive surface GP: early brackets: pending forecasts "
                        "unavailable ({}); keeping previous forecasts",
                        error.what());
        }
    }

  // Opt-in extensions of readiness-driven certification. Both read forecasts
  // written by the advisor from THIS fit, which used completed thresholds
  // only; no in-flight outcome or partial bracket can enter them. A Sync
  // barrier has no unfinished work at a fit, so both are inert in Sync.
  int anticipation_probes = 0;
  if ((cfg.gp_probe_inflight_promotion || cfg.gp_readiness_anticipation) &&
      probe_stop_enabled(cfg) &&
      cfg.gp_probe_readiness_transition)
    {
      const double elapsed_s = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() -
        self->state().start_time).count();
      std::vector<pending_forecast_t> forecasts;
      readiness_forecast_t forecast;
      try
        {
          forecasts = read_pending_forecasts(
            surface_state.gp_output_dir / (prefix.str() + "_pending_predictions.csv"));
          forecast = read_readiness_forecast(
            surface_state.gp_output_dir / (prefix.str() + "_readiness_forecast.csv"));
        }
      catch (const std::exception &error)
        {
          surface_state.stop_reason = "in-flight forecast files could not be read";
          self->println("adaptive surface GP: {}: {}", surface_state.stop_reason,
                        error.what());
          return false;
        }
      if (forecast.n_training != metrics.n_training ||
          forecast.n_pending != unfinished_surface_count(self->state()) ||
          static_cast<int>(forecasts.size()) != forecast.n_pending ||
          forecast.n_grid != metrics.n_grid)
        {
          surface_state.stop_reason = "in-flight forecast provenance mismatch";
          self->println("adaptive surface GP: {}", surface_state.stop_reason);
          return false;
        }
      accumulate_prequential(self, surface_state);
      const int scored =
        static_cast<int>(surface_state.probe_relative_errors.size());
      const auto find_forecast = [&](const sample_t &sample)
        -> const pending_forecast_t * {
        for (const auto &row : forecasts)
          if (same_location(row.R, sample.R) &&
              same_location(row.gamma, sample.gamma))
            return &row;
        return nullptr;
      };
      const auto promotable = [&](const sample_t &sample) {
        return sample.gp_is_probe && !sample.completed && !sample.cancelled &&
               !sample.gp_probe_certification_eligible;
      };

      if (cfg.gp_probe_inflight_promotion && fresh_readiness)
        {
          int quota = inflight_promotion_quota(
            cfg.gp_probe_min_samples, scored,
            eligible_unfinished_probe_count(self->state(), surface_state, cfg));
          // Samples are stored in dispatch (id) order: earliest first.
          for (auto &sample : self->state().samples)
            {
              if (quota <= 0)
                break;
              if (!promotable(sample))
                continue;
              const auto *row = find_forecast(sample);
              if (row == nullptr)
                {
                  surface_state.stop_reason =
                    "missing forecast for an in-flight probe";
                  self->println("adaptive surface GP: {} (sample {})",
                                surface_state.stop_reason, sample.id);
                  return false;
                }
              const bool usable = std::isfinite(row->pred_mean) &&
                row->pred_mean > 0.0 && std::isfinite(row->pred_lower95) &&
                row->pred_lower95 > 0.0 && std::isfinite(row->pred_upper95) &&
                row->pred_lower95 <= row->pred_mean &&
                row->pred_mean <= row->pred_upper95 &&
                row->n_training == metrics.n_training;
              if (!usable)
                continue;
              // Persist before the state change; the admission audit keeps the
              // superseded admission-time forecast unchanged.
              auto &audit = surface_state.gp_probe_promotions_file;
              audit << sample.id << "," << sample.gp_round << "," << next_fit
                    << "," << sample.R << "," << sample.gamma << ","
                    << sample.gp_prediction_n_training << ","
                    << sample.gp_pred_mean << ","
                    << sample.gp_pred_lower95 << "," << sample.gp_pred_upper95
                    << "," << metrics.n_training << "," << row->pred_mean << ","
                    << row->pred_lower95 << "," << row->pred_upper95 << ","
                    << row->posterior_std << "," << elapsed_s << "\n";
              audit.flush();
              if (!audit)
                throw std::runtime_error("could not record probe promotion");
              sample.gp_prediction_available = true;
              sample.gp_prediction_n_training = metrics.n_training;
              sample.gp_prediction_readiness = true;
              sample.gp_probe_certification_eligible = true;
              sample.gp_probe_epoch = surface_state.certification_epoch;
              sample.gp_pred_mean = row->pred_mean;
              sample.gp_pred_lower95 = row->pred_lower95;
              sample.gp_pred_upper95 = row->pred_upper95;
              sample.gp_posterior_std = row->posterior_std;
              ++surface_state.probes_promoted;
              --quota;
              self->println(
                "adaptive surface GP: promoted in-flight probe sample={} "
                "(R={:.6f}, gamma={:.6g}) to certification eligibility with "
                "ready forecast {:.8f} from n={} (admission forecast n={})",
                sample.id, sample.R, sample.gamma, row->pred_mean,
                metrics.n_training, row->n_training);
            }
        }

      if (cfg.gp_readiness_anticipation)
        {
          const int eligible_in_flight =
            eligible_unfinished_probe_count(self->state(), surface_state, cfg);
          const int promotable_in_flight = static_cast<int>(std::count_if(
            self->state().samples.begin(), self->state().samples.end(),
            promotable));
          const bool anticipated = !fresh_readiness &&
            readiness_from_metrics(metrics, cfg, forecast.max_after,
                                   forecast.rms_after);
          anticipation_probes = anticipation_probe_count(
            anticipated, cfg.gp_probe_min_samples, scored, eligible_in_flight,
            promotable_in_flight, requested_suggestions);
          // The forecast holds the hyperparameters fixed; when refits keep
          // moving the uncertainty, anticipatory probes can complete before
          // readiness and become ineligible. Admit at most one certification
          // set per run this way; the post-readiness sprint still covers any
          // remaining deficit with fresh probes.
          const int anticipation_budget =
            cfg.gp_probe_min_samples - surface_state.anticipation_probes_admitted;
          if (anticipation_probes > std::max(0, anticipation_budget))
            {
              self->println(
                "adaptive surface GP: anticipation budget exhausted ({} of {} "
                "admitted); not admitting further anticipatory probes",
                surface_state.anticipation_probes_admitted, cfg.gp_probe_min_samples);
              anticipation_probes = std::max(0, anticipation_budget);
            }
          auto &audit = surface_state.gp_readiness_anticipation_file;
          audit << next_fit << "," << metrics.n_training << ","
                << forecast.n_pending << "," << (fresh_readiness ? "true" : "false")
                << "," << forecast.max_now << "," << forecast.rms_now << ","
                << forecast.max_after << "," << forecast.rms_after << ","
                << (anticipated ? "true" : "false") << "," << scored << ","
                << eligible_in_flight << "," << promotable_in_flight << ","
                << requested_suggestions << "," << anticipation_probes << ","
                << elapsed_s << "\n";
          audit.flush();
          if (!audit)
            throw std::runtime_error("could not record readiness anticipation");
          if (anticipated)
            self->println(
              "adaptive surface GP: readiness anticipated after {} in-flight "
              "search(es) (forecast max/RMS {:.3f}%/{:.3f}%); {} scored, {} "
              "eligible and {} promotable probe(s) in flight; admitting {} "
              "anticipatory probe(s)",
              forecast.n_pending, 100.0 * forecast.max_after,
              100.0 * forecast.rms_after, scored, eligible_in_flight,
              promotable_in_flight, anticipation_probes);
        }
    }

  if (evaluation.admission_frozen)
    {
      record_decision(0);
      self->println("adaptive surface GP: acquisition skipped (admissions frozen)");
      // Admit nothing, but the run is not over: the in-flight searches are the
      // evidence being waited on, and every completion re-enters here.
      if (!surface_state.admission_freeze_announced)
        {
          surface_state.admission_freeze_announced = true;
          self->println("adaptive surface GP: {}", evaluation.reason);
        }
      if (probe_stop_enabled(cfg))
        {
          surface_state.probe_certification_sprinting = true;
          surface_state.probe_certification_draining = true;
        }
      return true;
    }
  surface_state.admission_freeze_announced = false;
  if (sample_budget_exhausted || update_budget_exhausted)
    {
      record_decision(0);
      self->println("adaptive surface GP: acquisition skipped (safety budget)");
      if (cfg.mode == "asynchronous" && self->state().active_jobs > 0)
        return true;
      surface_state.stop_reason = sample_budget_exhausted
        ? "maximum total surface sample count reached (" +
            std::to_string(cfg.max_samples) + ")"
        : "maximum adaptive surface GP update count reached (" +
            std::to_string(cfg.max_rounds) + ")";
      return false;
    }

  // The legacy path sets its probe allocation before fitting. Opt-in Async
  // instead uses the just-evaluated sprint trigger, without waiting for any
  // other worker or changing the stopping test. Existing latched-sprint and
  // scored-probe fallback semantics are preserved.
  const int async_eligible_in_flight =
    cfg.gp_async_postfit_certification && cfg.mode == "asynchronous"
      ? eligible_unfinished_probe_count(self->state(), surface_state, cfg) : 0;
  const bool scored_probe_trigger =
    evaluation.probe_n > 0 && evaluation.probe_n < cfg.gp_probe_min_samples &&
    std::isfinite(evaluation.probe_rms_relative_error) &&
    evaluation.probe_rms_relative_error <= cfg.gp_probe_rms_tolerance;
  const auto async_plan = async_postfit_certification(
    cfg.gp_async_postfit_certification, cfg.mode == "asynchronous",
    probe_stop_enabled(cfg),
    async_certification_sprint_trigger(
      readiness_guard, fresh_readiness,
      surface_state.probe_certification_sprinting, scored_probe_trigger),
    metrics.n_training, probe_eligibility_warmup(cfg, surface_state),
    evaluation.probe_n, cfg.gp_probe_min_samples, async_eligible_in_flight,
    requested_suggestions);
  if (async_plan.drain)
    {
      surface_state.probe_certification_sprinting = true;
      surface_state.probe_certification_draining = true;
      surface_state.admission_freeze_announced = true;
      evaluation.admission_frozen = true;
      evaluation.reason = "Async post-fit certification: enough eligible "
        "probes are already in flight; admitting no new work";
      record_decision(0);
      self->println("adaptive surface GP: {}", evaluation.reason);
      return true;
    }
  int certification_count = sync_certification_probe_batch(
    cfg.gp_sync_certification_batches, cfg.mode == "synchronous",
    probe_stop_enabled(cfg), probe_sprint_diagnostics_ready(evaluation, cfg),
    unfinished_surface_count(self->state()), metrics.n_training,
    probe_eligibility_warmup(cfg, surface_state), evaluation.probe_n,
    cfg.gp_probe_min_samples, evaluation.probe_rms_relative_error,
    cfg.gp_probe_rms_tolerance, requested_suggestions);
  if (certification_count > 0)
    {
      evaluation.reason = "Sync certification batch: " +
        std::to_string(evaluation.probe_n) + " scored; requesting " +
        std::to_string(certification_count) + " missing probe(s)";
      self->println("adaptive surface GP: {} (ordinary capacity {})",
                    evaluation.reason, requested_suggestions);
    }
  if (async_plan.probes > 0)
    {
      certification_count = async_plan.probes;
      surface_state.probe_certification_sprinting = true;
      evaluation.reason = "Async post-fit certification: " +
        std::to_string(evaluation.probe_n) + " scored, " +
        std::to_string(async_eligible_in_flight) +
        " eligible in flight; requesting " +
        std::to_string(certification_count) + " missing probe(s)";
      self->println("adaptive surface GP: {} (ordinary capacity {})",
                    evaluation.reason, requested_suggestions);
    }
  if (certification_count == 0 && anticipation_probes > 0)
    {
      certification_count = anticipation_probes;
      evaluation.reason = "readiness anticipation: requesting " +
        std::to_string(certification_count) +
        " probe(s) to be promoted at the forecast ready fit";
      self->println("adaptive surface GP: {} (ordinary capacity {})",
                    evaluation.reason, requested_suggestions);
      surface_state.anticipation_probes_admitted += certification_count;
    }
  const int acquired_capacity = certification_count > 0
                                  ? certification_count : requested_suggestions;
  const auto acquisition_start = std::chrono::steady_clock::now();
  // Keep the original deferred request intact. The explicit, bounded resume
  // control changes only probe admissions, reusing the exact fitted posterior.
  const std::string acquisition_command = command + " --resume-acquisition" +
    (certification_count > 0
       ? " --resume-probe-only-count " + std::to_string(certification_count)
       : "");
  self->println("adaptive surface GP acquisition command: {}", acquisition_command);
  const int acquisition_rc = run_surface_advisor(self, cfg, acquisition_command);
  const double acquisition_runtime_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - acquisition_start).count();
  metrics.advisor_runtime_s += acquisition_runtime_s;
  self->println("adaptive surface GP: acquisition completed in {:.3f}s; "
                "total advisor {:.3f}s", acquisition_runtime_s,
                metrics.advisor_runtime_s);
  record_decision(acquired_capacity);
  if (acquisition_rc != 0)
    {
      surface_state.stop_reason = "surface GP acquisition command failed";
      self->println("adaptive surface GP: acquisition failed with return code {}",
                    acquisition_rc);
      return false;
    }

  std::vector<gp_suggestion_t> suggestions;
  try
    {
      suggestions = read_gp_suggestions(
        surface_state.gp_output_dir / (prefix.str() + "_suggestions.csv"));
    }
  catch (const std::exception &error)
    {
      surface_state.stop_reason = "surface GP suggestions could not be read";
      self->println("adaptive surface GP: {}", error.what());
      return false;
    }
  if (suggestions.empty())
    {
      surface_state.stop_reason =
        "surface GP advisor returned no eligible locations";
      return false;
    }
  if (suggestions.size() > static_cast<std::size_t>(acquired_capacity))
    {
      surface_state.stop_reason =
        "surface GP advisor returned more locations than requested";
      return false;
    }

  // The advisor inverts normalized coordinates, so a location on a domain
  // edge can come back a few ulps outside it (e.g. gamma = 0.01 + 2e-18).
  // Snap such values onto the bound instead of aborting the run.
  {
    const double r_tol = 1.0e-9 * std::abs(cfg.R_max - cfg.R_min);
    const double g_tol = 1.0e-9 * std::abs(cfg.gamma_max - cfg.gamma_min);
    for (auto &s : suggestions)
      {
        if (std::isfinite(s.R) && s.R < cfg.R_min && s.R >= cfg.R_min - r_tol)
          s.R = cfg.R_min;
        if (std::isfinite(s.R) && s.R > cfg.R_max && s.R <= cfg.R_max + r_tol)
          s.R = cfg.R_max;
        if (std::isfinite(s.gamma) && s.gamma < cfg.gamma_min &&
            s.gamma >= cfg.gamma_min - g_tol)
          s.gamma = cfg.gamma_min;
        if (std::isfinite(s.gamma) && s.gamma > cfg.gamma_max &&
            s.gamma <= cfg.gamma_max + g_tol)
          s.gamma = cfg.gamma_max;
      }
  }
  for (std::size_t i = 0; i < suggestions.size(); ++i)
    {
      const auto &suggestion = suggestions[i];
      if (certification_count > 0 && suggestion.selection_method != "probe")
        {
          surface_state.stop_reason =
            "surface GP certification batch returned a non-probe location";
          return false;
        }
      if (!std::isfinite(suggestion.R) ||
          !std::isfinite(suggestion.gamma) || suggestion.R < cfg.R_min ||
          suggestion.R > cfg.R_max || suggestion.gamma < cfg.gamma_min ||
          suggestion.gamma > cfg.gamma_max)
        {
          surface_state.stop_reason =
            "surface GP advisor returned an invalid or out-of-bounds location";
          return false;
        }
      if (has_nearby_attempted_surface_location(
            self->state(), suggestion.R, suggestion.gamma, cfg,
            surface_state.model_log_R, metrics.input_transform_gamma == "log"))
        {
          surface_state.stop_reason =
            "surface GP advisor returned a location too close to an attempt";
          return false;
        }
      for (std::size_t j = 0; j < i; ++j)
        if (normalized_surface_distance(suggestion.R,
                                        suggestion.gamma,
                                        suggestions[j].R,
                                        suggestions[j].gamma,
                                        cfg,
                                        surface_state.model_log_R,
                                        metrics.input_transform_gamma == "log") <
            cfg.gp_surface_suggest_min_distance)
          {
            surface_state.stop_reason =
              "surface GP batch violates normalized minimum spacing";
            return false;
          }
    }

  // Decide early, bracket conservatively (see --gp-async-start-min-valid):
  // until the legacy corner/half-design gate passes, or the seed design has
  // fully drained, new searches keep the global bracket. Probe forecasts and
  // their provenance are unaffected; only the dispatched bracket changes.
  // Early GP brackets replace the corner gate by a per-dispatch expected-solve
  // rule (early_brackets.h); the gate below is then skipped.
  const bool early_brackets_active = cfg.gp_async_early_brackets &&
    cfg.mode == "asynchronous";
  bool bracket_gate_ready = true;
  if (!early_brackets_active &&
      cfg.mode == "asynchronous" && cfg.gp_async_start_min_valid > 0)
    {
      int valid_seed = 0, unfinished_seed = 0;
      unsigned corner_mask = 0;
      for (const auto &sample : self->state().samples)
        if (sample.id < cfg.initial_samples && sample.gp_round == 0)
          {
            if (!sample.completed)
              ++unfinished_seed;
            else if (sample.threshold_found && !sample.failed && !sample.cancelled)
              {
                ++valid_seed;
                corner_mask |= async_initial_corner_bit(
                  sample.R, sample.gamma, cfg.R_min, cfg.R_max,
                  cfg.gamma_min, cfg.gamma_max);
              }
          }
      bracket_gate_ready = unfinished_seed == 0 ||
        async_early_start_ready(cfg.initial_samples, valid_seed, corner_mask);
      if (!bracket_gate_ready)
        self->println(
          "adaptive surface GP: early decision at {} valid thresholds ({} seeds "
          "unfinished); GP-seeded brackets deferred until the corner/half-design "
          "gate passes, dispatching global brackets",
          n_valid, unfinished_seed);
    }

  for (const auto &suggestion : suggestions)
    {
      auto bracket_forecast = suggestion;
      if (!bracket_gate_ready)
        bracket_forecast.prediction_available = false;
      auto bracket = seeded_bracket(cfg, bracket_forecast);
      if (early_brackets_active && bracket_forecast.prediction_available)
        {
          const early_bracket_t candidate{true, bracket.first, bracket.second};
          const double rho = cfg.gp_bracket_quench_cost_ratio;
          const double global_cost = global_bracket_cost(
            cfg.Av_lo, cfg.Av_hi, bracket_forecast.pred_mean,
            cfg.bisection_tolerance, cfg.bisection_relative_tolerance, rho);
          const double gp_cost = gp_bracket_cost(
            candidate, bracket_forecast.pred_mean, cfg.Av_lo,
            cfg.bisection_tolerance, cfg.bisection_relative_tolerance, rho);
          const bool preferred = gp_bracket_preferred(
            candidate, bracket_forecast.pred_mean, cfg.Av_lo, cfg.Av_hi,
            cfg.bisection_tolerance, cfg.bisection_relative_tolerance, rho);
          record_early_bracket(surface_state,
                               static_cast<int>(self->state().samples.size()),
                               "enqueue", metrics.n_training,
                               bracket_forecast.pred_mean,
                               bracket_forecast.posterior_std, candidate,
                               gp_cost, global_cost, preferred);
          if (!preferred)
            bracket = {cfg.Av_lo, cfg.Av_hi};
        }
      auto sample = make_sample(static_cast<int>(self->state().samples.size()),
                                suggestion.R,
                                cfg.Aw,
                                suggestion.gamma,
                                wave_speed_for_gamma(suggestion.gamma, cfg),
                                cfg.steepness,
                                cfg.ic_type,
                                bracket.first,
                                bracket.second);
      sample.gp_seeded_bracket =
        bracket.first != cfg.Av_lo || bracket.second != cfg.Av_hi;
      sample.bracket_source = sample.gp_seeded_bracket ? "gp" : "global";
      sample.seed_Av_lo = bracket.first;
      sample.seed_Av_hi = bracket.second;
      sample.gp_round = next_round;
      sample.gp_prediction_available = suggestion.prediction_available;
      sample.gp_bracket_prediction_available = bracket_forecast.prediction_available;
      sample.gp_bracket_pred_mean = bracket_forecast.pred_mean;
      sample.gp_bracket_posterior_std = bracket_forecast.posterior_std;
      sample.gp_prediction_n_training = metrics.n_training;
      sample.gp_is_probe = suggestion.selection_method == "probe";
      sample.gp_prediction_readiness = fresh_readiness;
      const bool usable_forecast = suggestion.prediction_available &&
        std::isfinite(suggestion.pred_mean) && suggestion.pred_mean > 0.0 &&
        std::isfinite(suggestion.pred_lower95) && suggestion.pred_lower95 > 0.0 &&
        std::isfinite(suggestion.pred_upper95) &&
        suggestion.pred_lower95 <= suggestion.pred_mean &&
        suggestion.pred_mean <= suggestion.pred_upper95;
      sample.gp_probe_certification_eligible =
        (!cfg.gp_probe_readiness_transition || usable_forecast) &&
        probe_certification_eligible_at_admission(
          sample.gp_is_probe, sample.gp_prediction_n_training,
          probe_eligibility_warmup(cfg, surface_state),
          probe_readiness_eligibility_active(
            cfg.gp_probe_require_readiness, cfg.surface),
          sample.gp_prediction_readiness);
      sample.gp_probe_epoch = surface_state.certification_epoch;
      if (sample.gp_is_probe)
        {
          ++surface_state.probes_dispatched;
          ++surface_state.probe_index;
        }
      sample.gp_pred_mean = suggestion.pred_mean;
      sample.gp_pred_lower95 = suggestion.pred_lower95;
      sample.gp_pred_upper95 = suggestion.pred_upper95;
      sample.gp_posterior_std = suggestion.posterior_std;
      if (sample.gp_is_probe)
        {
          // Persist the prospective decision before the sample can be sent to
          // a worker. All probes are recorded, including ineligible ones.
          auto &audit = surface_state.gp_probe_admissions_file;
          audit << sample.id << "," << sample.gp_round << "," << next_fit
                << "," << sample.R << "," << sample.gamma << ","
                << sample.gp_prediction_n_training << ","
                << (sample.gp_prediction_readiness ? "true" : "false") << ","
                << (sample.gp_probe_certification_eligible ? "true" : "false")
                << "," << (sample.gp_prediction_available ? "true" : "false")
                << "," << sample.gp_pred_mean << "," << sample.gp_pred_lower95
                << "," << sample.gp_pred_upper95 << "\n";
          audit.flush();
          if (!audit)
            throw std::runtime_error("could not record prospective probe admission");
        }
      enqueue_sample(self, std::move(sample));
    }

  if (cfg.mode == "asynchronous" && probe_stop_enabled(cfg) &&
      async_certification_readiness_permits(readiness_guard, fresh_readiness) &&
      surface_state.probe_certification_sprinting &&
      evaluation.probe_n < cfg.gp_probe_min_samples &&
      evaluation.probe_n + eligible_unfinished_probe_count(
                               self->state(), surface_state, cfg) >=
        cfg.gp_probe_min_samples)
    {
      surface_state.probe_certification_draining = true;
      if (!surface_state.admission_freeze_announced)
        {
          surface_state.admission_freeze_announced = true;
          self->println(
            "adaptive surface GP: certification probes are in flight; "
            "freezing further admissions");
        }
    }
  surface_state.rounds_started = next_round;
  self->println("adaptive surface GP: enqueued {} locations for round {}",
                suggestions.size(),
                next_round);
  wake_idle_workers(self);
  return true;
}

bool adaptive_generation_stopped(const server_state &state)
{
  return state.adaptive_surface_state.generation_stopped;
}

void fit_final_adaptive_surface_gp(
  stateful_actor<server_state> *self,
  adaptive_gamma_state        &surface_state,
  const adaptive_gp_config_t  &cfg)
{
  const int n_valid = valid_surface_threshold_count(self->state());
  if (n_valid < 3 ||
      n_valid == surface_state.last_gp_fit_valid_threshold_count)
    return;
  const auto input_path = surface_state.gp_output_dir /
                          "adaptive_surface_final_threshold_input.csv";
  {
    std::ofstream out(input_path.string(), std::ios::out);
    if (!out)
      throw std::runtime_error("could not create " + input_path.string());
    out << "sample_id,R,gamma,critical_Av,Av_lo,Av_hi,bracket_width,status,run_dir\n"
        << std::setprecision(17);
    for (const auto &sample : self->state().samples)
      if (sample.completed && sample.threshold_found && !sample.failed &&
          !sample.cancelled)
        out << sample.id << "," << sample.R << "," << sample.gamma << ","
            << sample.critical_Av << "," << sample.Av_lo << "," << sample.Av_hi
            << "," << sample.Av_hi - sample.Av_lo << ",converged,"
            << csv_escape(sample.run_dir) << "\n";
    for (const auto &sample : self->state().samples)
      if (salvageable_bracket(sample, cfg.gp_salvage_max_relative_half_width))
        out << sample.id << "," << sample.R << "," << sample.gamma << ","
            << 0.5 * (sample.Av_lo + sample.Av_hi) << "," << sample.Av_lo << ","
            << sample.Av_hi << "," << sample.Av_hi - sample.Av_lo
            << ",salvaged_partial," << csv_escape(sample.run_dir) << "\n";
  }
  std::string command =
    shell_quote(cfg.gp_python) + " " + shell_quote(cfg.gp_surface_script) +
    " " + shell_quote(input_path.string()) + " --out-dir " +
    shell_quote(surface_state.gp_output_dir.string()) +
    " --prefix adaptive_surface_final --r-min " + double_value(cfg.R_min) +
    " --r-max " + double_value(cfg.R_max) + " --gamma-min " +
    double_value(cfg.gamma_min) + " --gamma-max " +
    double_value(cfg.gamma_max) + " --target-transform " +
    shell_quote(cfg.gp_target_transform) + " --noise-floor " +
    double_value(cfg.gp_noise_floor) + " --prediction-r-count " +
    std::to_string(cfg.gp_surface_prediction_r_count) +
    " --prediction-gamma-count " +
    std::to_string(cfg.gp_surface_prediction_gamma_count) +
    " --integration-count " + std::to_string(cfg.gp_integration_grid_size) +
    " --candidate-count 1 --batch-size 0 --seed 104729";
  if (cfg.gp_no_plot)
    command += " --no-plot";
  {
    int n_salvaged = 0, n_rejected = 0;
    for (const auto &sample : self->state().samples)
      if (sample.completed && sample.cancelled)
        ++(salvageable_bracket(sample, cfg.gp_salvage_max_relative_half_width)
             ? n_salvaged : n_rejected);
    self->println("adaptive surface GP: final fit input: {} converged thresholds, "
                  "{} salvaged partial brackets, {} cancelled brackets rejected "
                  "(unobserved end, zero lower end or too wide)",
                  n_valid, n_salvaged, n_rejected);
  }
  self->println("adaptive surface GP: fitting final surrogate using {} thresholds",
                n_valid);
  const int rc = run_surface_advisor(self, cfg, command);
  if (rc == 0)
    surface_state.last_gp_fit_valid_threshold_count = n_valid;
  else
    self->println("adaptive surface GP: final fit failed with return code {}", rc);
}

void finish_adaptive_workflow(stateful_actor<server_state> *self,
                              const adaptive_gp_config_t  &adaptive_cfg)
{
  if (self->state().termination_initiated ||
      !self->state().pending_jobs.empty() || self->state().active_jobs != 0)
    return;
  if (adaptive_cfg.enabled && !adaptive_generation_stopped(self->state()))
    return;

  if (adaptive_cfg.surface)
    {
      auto &surface_state = self->state().adaptive_surface_state;
      if (adaptive_cfg.mode == "asynchronous")
        for (int round = 1; round <= surface_state.rounds_started; ++round)
          report_surface_prequential_validation(self, surface_state, round);
      else
        report_surface_prequential_validation(
          self, surface_state, surface_state.rounds_started);
      fit_final_adaptive_surface_gp(self, surface_state, adaptive_cfg);
    }

  if (adaptive_cfg.enabled)
    {
      const auto summary_path = self->state().gp_output_dir /
                                "surface_summary.csv";
      std::ofstream summary(summary_path.string(), std::ios::out);
      if (summary)
        summary
          << "R_min,R_max,gamma_min,gamma_max,completed,cancelled,attempted,"
             "gp_suggestion_rounds_started,gp_fits_started,stop_reason\n"
          << std::setprecision(17) << adaptive_cfg.R_min << ","
          << adaptive_cfg.R_max << "," << adaptive_cfg.gamma_min << ","
          << adaptive_cfg.gamma_max << ","
          << self->state().total_jobs_completed << ","
          << self->state().total_jobs_cancelled << ","
          << self->state().samples.size() << ","
          << self->state().adaptive_surface_state.rounds_started << ","
          << self->state().adaptive_surface_state.fits_started << ","
          << csv_escape(self->state().adaptive_surface_state.stop_reason)
          << "\n";
    }

  self->state().termination_initiated = true;
  const auto end_time = std::chrono::high_resolution_clock::now();
  const double runtime_s =
    std::chrono::duration<double>(end_time - self->state().start_time).count();
  self->println("=== DONE: {} jobs completed, {} cancelled, {:.1f}s total ===",
                self->state().total_jobs_completed,
                self->state().total_jobs_cancelled,
                runtime_s);
  if (adaptive_cfg.surface)
    self->println("Adaptive surface GP stop reason: {} ({} completed, {} attempted)",
                  self->state().adaptive_surface_state.stop_reason,
                  self->state().total_jobs_completed,
                  self->state().samples.size());

  for (auto &worker : self->state().active_workers)
    anon_mail("quit").send(worker);
  for (auto &controller : self->state().cancellation_controllers)
    anon_mail("quit").send(controller);
}

void request_surface_cancellation(stateful_actor<server_state> *self,
                                  adaptive_gamma_state        &surface_state,
                                  const adaptive_gp_config_t  &cfg)
{
  if (!cfg.cancel_running_on_stop ||
      !surface_state.stop_criterion_reached ||
      surface_state.cancellation_requested)
    return;
  surface_state.cancellation_requested = true;
  int cancelled_before_start = 0;
  while (!self->state().pending_jobs.empty())
    {
      const auto job = self->state().pending_jobs.front();
      self->state().pending_jobs.pop_front();
      auto &sample = self->state().samples.at(job.sample_id);
      sample.completed = true;
      sample.cancelled = true;
      ++cancelled_before_start;
      ++self->state().total_jobs_cancelled;
    }
  int active_cancel_requests = 0;
  for (const auto &[worker, sample_id] : self->state().worker_active_samples)
    {
      static_cast<void>(sample_id);
      const auto controller =
        self->state().worker_cancellation_controllers.find(worker);
      if (controller != self->state().worker_cancellation_controllers.end())
        {
          anon_mail("cancel").send(controller->second);
          ++active_cancel_requests;
        }
    }
  self->println(
    "adaptive surface GP: cancellation requested for {} active and {} "
    "reserved simulations",
    active_cancel_requests,
    cancelled_before_start);
}

// Returns true when the caller must not fit or admit new work on this pass.
// A passing probe score remains independent of the Python advisor. The opt-in
// readiness guard additionally revisits fresh diagnostics after a new accepted
// observation, so a drain cannot conceal a regression in model readiness.
bool service_async_probe_certification(
  stateful_actor<server_state> *self,
  adaptive_gamma_state        &gamma_state,
  const adaptive_gp_config_t  &cfg)
{
  if (!gamma_state.probe_certification_draining ||
      !probe_stop_enabled(cfg))
    return false;

  accumulate_prequential(self, gamma_state);
  const int scored =
    static_cast<int>(gamma_state.probe_relative_errors.size());
  const double rms = probe_rms_relative_error(gamma_state);
  if (scored >= cfg.gp_probe_min_samples)
    {
      gamma_state.probe_certification_draining = false;
      gamma_state.admission_freeze_announced = false;
      if (std::isfinite(rms) && rms <= cfg.gp_probe_rms_tolerance)
        {
          gamma_state.probe_certification_sprinting = false;
          gamma_state.stop_reason =
            "unbiased probe RMS relative error over " +
            std::to_string(scored) +
            " space-filling predictions is within the accuracy target" +
            certification_set_suffix(gamma_state);
          record_certification_set(self, gamma_state,
                                   gamma_state.certification_epoch + 1,
                                   "passed", gamma_state.probe_relative_errors,
                                   gamma_state.probe_error_sample_ids, "drain");
          gamma_state.stop_criterion_reached = true;
          gamma_state.generation_stopped = true;
          self->println("adaptive surface GP: stopping because {}",
                        gamma_state.stop_reason);
          return true;
        }
      self->println(
        "adaptive surface GP: probe RMS rose above the target; resuming "
        "acquisition");
      archive_failed_certification_set(self, gamma_state, cfg, "drain");
      gamma_state.probe_certification_sprinting = false;
      return false;
    }

  if (async_certification_drain_needs_refit(
        certification_readiness_guard_enabled(cfg),
        valid_surface_threshold_count(self->state()),
        gamma_state.last_gp_fit_valid_threshold_count))
    {
      gamma_state.probe_certification_draining = false;
      gamma_state.admission_freeze_announced = false;
      self->println(
        "adaptive surface GP: Async readiness guard: new observation during "
        "probe drain; refitting before further admission decisions");
      return false;
    }

  const int eligible_unfinished = eligible_unfinished_probe_count(
    self->state(), gamma_state, cfg);
  if (scored + eligible_unfinished >= cfg.gp_probe_min_samples)
    return true;

  // A probe failed or was cancelled, so the existing pipeline can no longer
  // complete certification. Resume fitting but keep the sprint latched: the
  // forced-probe calculation will replace the missing evidence before
  // admitting acquisitions.
  gamma_state.probe_certification_draining = false;
  gamma_state.admission_freeze_announced = false;
  self->println(
    "adaptive surface GP: only {} scored and {} eligible in-flight probe(s); "
    "resuming admissions to replace missing certification evidence",
    scored,
    eligible_unfinished);
  return false;
}

void drive_synchronous_adaptive_surface_gp(
  stateful_actor<server_state> *self,
  const adaptive_gp_config_t  &cfg)
{
  auto &surface_state = self->state().adaptive_surface_state;
  if (self->state().termination_initiated)
    return;
  if (!surface_state.generation_stopped &&
      unfinished_surface_count(self->state()) == 0)
    {
      report_surface_prequential_validation(
        self, surface_state, surface_state.rounds_started);
      if (!start_adaptive_gp_surface_round(
            self, surface_state, cfg, cfg.batch_size))
        surface_state.generation_stopped = true;
    }
  wake_idle_workers(self);
  finish_adaptive_workflow(self, cfg);
}

void drive_asynchronous_adaptive_surface_gp(
  stateful_actor<server_state> *self,
  const adaptive_gp_config_t  &cfg)
{
  auto &surface_state = self->state().adaptive_surface_state;
  if (self->state().termination_initiated)
    return;
  wake_idle_workers(self);
  if (surface_state.generation_stopped)
    {
      request_surface_cancellation(self, surface_state, cfg);
      finish_adaptive_workflow(self, cfg);
      return;
    }
  if (service_async_probe_certification(self, surface_state, cfg))
    {
      if (surface_state.generation_stopped)
        request_surface_cancellation(self, surface_state, cfg);
      finish_adaptive_workflow(self, cfg);
      return;
    }
  if (!self->state().pending_jobs.empty() || self->state().idle_workers.empty())
    return;
  const int n_valid = valid_surface_threshold_count(self->state());
  if (n_valid < 3)
    {
      if (unfinished_surface_count(self->state()) == 0)
        {
          surface_state.stop_reason =
            "fewer than three valid ignition thresholds are available";
          surface_state.generation_stopped = true;
          finish_adaptive_workflow(self, cfg);
        }
      return;
    }
  // Opt-in early adaptation overlaps the seed drain, never the queued seed
  // dispatch above. Only completed valid seeds count toward this coverage
  // gate. A failed/missing corner cannot deadlock the run: once the seed
  // drains, the existing valid-count, budget and boundary logic takes over.
  // This gate is not a convergence or calibration test.
  if (cfg.gp_async_early_start &&
      cfg.gp_async_start_min_valid <= 0 &&
      surface_state.rounds_started == 0 &&
      unfinished_surface_count(self->state()) > 0)
    {
      int valid_seed = 0;
      unsigned corner_mask = 0;
      for (const auto &sample : self->state().samples)
        if (sample.id < cfg.initial_samples && sample.gp_round == 0 &&
            sample.completed && sample.threshold_found &&
            !sample.failed && !sample.cancelled)
          {
            ++valid_seed;
            corner_mask |= async_initial_corner_bit(
              sample.R, sample.gamma, cfg.R_min, cfg.R_max,
              cfg.gamma_min, cfg.gamma_max);
          }
      if (!async_early_start_ready(cfg.initial_samples, valid_seed, corner_mask))
        return;
    }
  // The legacy count-only option is unchanged unless the guarded option is
  // explicitly enabled. Sync's full-batch driver never enters this path.
  const int first_fit_threshold =
    cfg.gp_async_start_min_valid > 0
      ? std::min(std::max(3, cfg.gp_async_start_min_valid), cfg.initial_samples) :
    cfg.gp_async_early_start ? async_early_start_minimum(cfg.initial_samples) :
    (cfg.mode == "asynchronous" && cfg.adaptive_min_fit_samples > 0)
      ? std::min(cfg.adaptive_min_fit_samples, cfg.initial_samples)
      : cfg.initial_samples;
  if (surface_state.rounds_started == 0 && n_valid < first_fit_threshold &&
      unfinished_surface_count(self->state()) > 0)
    return;
  if (cfg.gp_async_early_start &&
      surface_state.fits_started == 0)
    self->println(
      "adaptive surface GP: guarded first fit at {} valid thresholds, {} "
      "unfinished, seed size {}; application elapsed {:.3f}s",
      n_valid, unfinished_surface_count(self->state()), cfg.initial_samples,
      std::chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    self->state().start_time).count());
  // Refit only when a newly completed valid threshold is available. Pending
  // simulations are passed to the acquisition so replacements do not cluster
  // around work that is already in flight.
  const bool readiness_work_shortfall =
    cfg.gp_probe_readiness_transition &&
    unfinished_surface_count(self->state()) <
      surface_state.last_gp_decision_unfinished_count;
  if (n_valid <= surface_state.last_gp_fit_valid_threshold_count &&
      !readiness_work_shortfall)
    return;
  int request = static_cast<int>(self->state().idle_workers.size());
  if (cfg.async_target_active > 0)
    {
      const int unfinished = unfinished_surface_count(self->state());
      request = std::min(
        request, std::max(0, cfg.async_target_active - unfinished));
      if (request == 0)
        return;
    }
  if (cfg.async_fill_cap > 0)
    request = std::min(request, cfg.async_fill_cap);
  request = std::max(1, request);
  report_surface_prequential_validation(
    self, surface_state, surface_state.rounds_started);
  if (!start_adaptive_gp_surface_round(self, surface_state, cfg, request))
    {
      surface_state.generation_stopped = true;
      request_surface_cancellation(self, surface_state, cfg);
    }
  surface_state.last_gp_decision_unfinished_count =
    unfinished_surface_count(self->state());
  wake_idle_workers(self);
  finish_adaptive_workflow(self, cfg);
}

void check_termination(stateful_actor<server_state> *self,
                       const adaptive_gp_config_t  &adaptive_cfg)
{
  if (!adaptive_cfg.enabled)
    {
      if (self->state().pending_jobs.empty() && self->state().active_jobs == 0)
        finish_adaptive_workflow(self, adaptive_cfg);
      return;
    }
  if (adaptive_cfg.surface)
    {
      if (adaptive_cfg.mode == "asynchronous")
        drive_asynchronous_adaptive_surface_gp(self, adaptive_cfg);
      else
        drive_synchronous_adaptive_surface_gp(self, adaptive_cfg);
      return;
    }
}
} // namespace

caf::behavior server(stateful_actor<server_state> *self,
                     uint16_t port,
                     const config &cfg)
{
  self->state().start_time = std::chrono::high_resolution_clock::now();
  self->state().global_Av_lo = cfg.sim.Av_lo;
  self->state().global_Av_hi = cfg.sim.Av_hi;
  if (port != 0 && !self->system().middleman().publish(self, port))
    {
      self->println("Failed to publish server on port {}", port);
      self->quit();
      return {};
    }

  std::filesystem::create_directories(cfg.sim.output_dir);
  self->state().results_path =
    std::filesystem::path(cfg.sim.output_dir) / "critical_amplitudes.csv";
  self->state().results_file.open(self->state().results_path.string(), std::ios::out);
  if (!self->state().results_file)
    throw std::runtime_error("could not create " + self->state().results_path.string());
  self->state().results_file
    << "sample_id,R,Aw,gamma,fast_wave_speed,planned_final_time,steepness,ic_type,critical_Av,Av_lo,Av_hi,"
       "bracket_width,bisection_iterations,solver_runs,converged,cancelled,status,"
       "runtime_s,run_dir,error,gp_round,gp_pred_mean,gp_pred_lower95,"
       "gp_pred_upper95,gp_posterior_std,gp_abs_error,gp_relative_error,"
       "gp_within_95ci,gp_seeded_bracket,seed_Av_lo,seed_Av_hi,"
       "gp_is_probe,gp_prediction_n_training,gp_prediction_readiness,"
       "gp_probe_certification_eligible";
  self->state().results_file
    << ",bracket_source";
  self->state().results_file << "\n";
  self->state().results_file << std::setprecision(17);

  std::vector<double> gamma_list = parse_gamma_values(cfg.gamma_values);
  if (gamma_list.empty())
    for (int i = 0; i < std::max(1, cfg.n_gam); ++i)
      gamma_list.push_back(linspace_value(cfg.gam_min, cfg.gam_max,
                                          std::max(1, cfg.n_gam), i));
  for (std::size_t i = 0; i < gamma_list.size(); ++i)
    for (std::size_t j = i + 1; j < gamma_list.size(); ++j)
      if (same_gamma(gamma_list[i], gamma_list[j]))
        throw std::runtime_error("--gamma-values contains a duplicate gamma");
  const auto gamma_wave_speeds =
    parse_gamma_wave_speeds(cfg.gamma_wave_speeds);

  if (cfg.adaptive_gp_surface && !cfg.adaptive_gp)
    throw std::runtime_error(
      "--adaptive-gp-surface requires --adaptive-gp");
  if (cfg.adaptive_gp_surface && !(0.0 < cfg.gam_min && cfg.gam_min < cfg.gam_max))
    throw std::runtime_error(
      "surface gamma bounds must satisfy 0 < gamma-min < gamma-max");
  std::shared_ptr<const fhn_fast_wave_speed_model> wave_speed_model;
  if (!cfg.wave_speed_continuation_file.empty())
    wave_speed_model = std::make_shared<const fhn_fast_wave_speed_model>(
      cfg.wave_speed_continuation_file,
      cfg.wave_speed_min_reliable_gamma,
      cfg.wave_speed_low_gamma_plateau,
      cfg.wave_speed_round_up);
  if (cfg.adaptive_gp_surface && !cfg.fixed_final_time && !wave_speed_model)
    throw std::runtime_error(
      "continuous-gamma radial surface mode requires "
      "--wave-speed-continuation-file (or --fixed-final-time for a "
      "non-radial observation window)");

  std::vector<double> initial_R_values = parse_R_values(cfg.R_values);
  std::vector<std::pair<double, double>> initial_surface_values;
  if (cfg.adaptive_gp_surface)
    {
      if (!initial_R_values.empty())
        throw std::runtime_error(
          "--R-values is ambiguous in surface mode; use the generated "
          "space-filling (R,gamma) initial design");
      if (!cfg.gp_surface_sample_list.empty())
        {
          initial_surface_values =
            read_surface_sample_list(cfg.gp_surface_sample_list);
          for (const auto &[R, gamma] : initial_surface_values)
            if (R < cfg.R_min || R > cfg.R_max || gamma < cfg.gam_min ||
                gamma > cfg.gam_max)
              throw std::runtime_error(
                "--gp-surface-sample-list contains a point outside "
                "[R_min,R_max] x [gamma_min,gamma_max]");
        }
      else
        initial_surface_values = initial_surface_design(
          std::max(1, cfg.adaptive_initial_samples),
          cfg.R_min,
          cfg.R_max,
          cfg.gam_min,
          cfg.gam_max,
          cfg.gp_surface_design_ladder_R,
          cfg.gp_surface_design_ladder_gamma);
    }
  else if (initial_R_values.empty())
    {
      const int initial_R_count =
        cfg.adaptive_gp ? std::max(1, cfg.adaptive_initial_samples)
                        : std::max(1, cfg.n_R);
      const double initial_R_min =
        cfg.adaptive_gp && cfg.gp_convergence_r_min > 0.0
          ? cfg.gp_convergence_r_min
          : cfg.R_min;
      initial_R_values.reserve(initial_R_count);
      for (int i = 0; i < initial_R_count; ++i)
        initial_R_values.push_back(
          linspace_value(initial_R_min, cfg.R_max, initial_R_count, i));
    }

  const int initial_R_count = static_cast<int>(initial_R_values.size());
  const int initial_total_count = cfg.adaptive_gp_surface
                                    ? static_cast<int>(initial_surface_values.size())
                                    : initial_R_count *
                                        static_cast<int>(gamma_list.size());

  if (cfg.adaptive_gp && cfg.adaptive_max_samples > 0 &&
      cfg.adaptive_max_samples <
        (cfg.adaptive_gp_surface ? initial_total_count : initial_R_count))
    throw std::runtime_error(
      "--adaptive-max-samples cannot be smaller than the initial sample count");

  adaptive_gp_config_t adaptive_cfg;
  adaptive_cfg.enabled = cfg.adaptive_gp;
  adaptive_cfg.surface = cfg.adaptive_gp_surface;
  adaptive_cfg.mode = cfg.adaptive_gp_mode;
  adaptive_cfg.batch_size = std::max(1, cfg.adaptive_batch_size);
  adaptive_cfg.initial_samples = std::max(1, cfg.adaptive_initial_samples);
  adaptive_cfg.max_rounds = std::max(0, cfg.adaptive_rounds);
  adaptive_cfg.max_samples = std::max(0, cfg.adaptive_max_samples);
  adaptive_cfg.cancel_running_on_stop =
    cfg.adaptive_cancel_running_on_stop;
  adaptive_cfg.async_fill_cap = std::max(0, cfg.adaptive_async_fill_cap);
  adaptive_cfg.async_target_active =
    std::max(0, cfg.adaptive_async_target_active);
  adaptive_cfg.R_min = cfg.R_min;
  adaptive_cfg.R_max = cfg.R_max;
  adaptive_cfg.gamma = gamma_list.front();
  adaptive_cfg.gamma_min = cfg.gam_min;
  adaptive_cfg.gamma_max = cfg.gam_max;
  adaptive_cfg.steepness = cfg.steepness;
  adaptive_cfg.ic_type = cfg.ic_type;
  adaptive_cfg.Av_lo = cfg.sim.Av_lo;
  adaptive_cfg.Av_hi = cfg.sim.Av_hi;
  adaptive_cfg.gp_python = cfg.gp_python;
  adaptive_cfg.gp_script = cfg.gp_script;
  adaptive_cfg.gp_surface_script = cfg.gp_surface_script;
  adaptive_cfg.gp_persistent_advisor = cfg.gp_persistent_advisor;
  adaptive_cfg.gp_grid_size = std::max(2, cfg.gp_grid_size);
  adaptive_cfg.gp_acquisition = cfg.gp_acquisition;
  if (adaptive_cfg.gp_acquisition != "max-std" &&
      adaptive_cfg.gp_acquisition != "greedy-nipv" &&
      adaptive_cfg.gp_acquisition != "variation-nipv" &&
      adaptive_cfg.gp_acquisition != "joint-nipv")
    throw std::runtime_error(
      "--gp-acquisition must be max-std, greedy-nipv, variation-nipv, or "
      "joint-nipv");
  adaptive_cfg.gp_integration_grid_size =
    std::max(2, cfg.gp_integration_grid_size);
  adaptive_cfg.gp_variation_weight = cfg.gp_variation_weight;
  adaptive_cfg.gp_focus_relative_uncertainty_target =
    cfg.gp_focus_relative_uncertainty_target;
  adaptive_cfg.gp_focus_variation_coverage =
    cfg.gp_focus_variation_coverage;
  adaptive_cfg.gp_acquisition_chunk_size =
    std::max(1, cfg.gp_acquisition_chunk_size);
  adaptive_cfg.gp_surface_nipv_backend = cfg.gp_surface_nipv_backend;
  if (adaptive_cfg.gp_surface_nipv_backend != "botorch" &&
      adaptive_cfg.gp_surface_nipv_backend != "analytic")
    throw std::runtime_error(
      "--gp-surface-nipv-backend must be botorch or analytic");
  adaptive_cfg.gp_joint_num_restarts =
    std::max(1, cfg.gp_joint_num_restarts);
  adaptive_cfg.gp_joint_raw_samples =
    std::max(adaptive_cfg.gp_joint_num_restarts, cfg.gp_joint_raw_samples);
  adaptive_cfg.gp_joint_maxiter = std::max(1, cfg.gp_joint_maxiter);
  adaptive_cfg.gp_joint_retries = std::max(1, cfg.gp_joint_retries);
  adaptive_cfg.gp_joint_fallback = cfg.gp_joint_fallback;
  adaptive_cfg.gp_surface_suggest_min_distance =
    std::max(0.0, cfg.gp_surface_suggest_min_distance);
  adaptive_cfg.gp_surface_suggest_min_distance_lengthscales =
    std::max(0.0, cfg.gp_surface_suggest_min_distance_lengthscales);
  adaptive_cfg.gp_surface_input_transform_R =
    cfg.gp_surface_input_transform_R;
  adaptive_cfg.gp_surface_input_transform_gamma =
    cfg.gp_surface_input_transform_gamma;
  adaptive_cfg.gp_surface_model_reselect_every =
    std::max(0, cfg.gp_surface_model_reselect_every);
  adaptive_cfg.gp_surface_model_eligibility_refresh =
    cfg.gp_surface_model_eligibility_refresh;
  adaptive_cfg.gp_acquisition_value_target =
    std::max(0.0, cfg.gp_acquisition_value_target);
  adaptive_cfg.adaptive_min_fit_samples =
    std::max(0, cfg.adaptive_min_fit_samples);
  adaptive_cfg.gp_async_early_start = cfg.gp_async_early_start;
  adaptive_cfg.gp_salvage_max_relative_half_width =
    std::max(0.0, cfg.gp_salvage_max_relative_half_width);
  adaptive_cfg.gp_surface_sample_list = cfg.gp_surface_sample_list;
  adaptive_cfg.gp_surface_design_ladder_R = cfg.gp_surface_design_ladder_R;
  adaptive_cfg.gp_surface_design_ladder_gamma =
    cfg.gp_surface_design_ladder_gamma;
  adaptive_cfg.gp_surface_interior_margin =
    std::max(0.0, cfg.gp_surface_interior_margin);
  adaptive_cfg.gp_surface_candidate_count =
    std::max(1, cfg.gp_surface_candidate_count);
  adaptive_cfg.gp_surface_prediction_r_count =
    std::max(2, cfg.gp_surface_prediction_r_count);
  adaptive_cfg.gp_surface_prediction_gamma_count =
    std::max(2, cfg.gp_surface_prediction_gamma_count);
  adaptive_cfg.gp_surface_boundary_tolerance =
    std::max(0.0, cfg.gp_surface_boundary_tolerance);
  adaptive_cfg.gp_relative_uncertainty_max_tolerance =
    std::max(0.0, cfg.gp_relative_uncertainty_max_tolerance);
  adaptive_cfg.gp_relative_uncertainty_rms_tolerance =
    std::max(0.0, cfg.gp_relative_uncertainty_rms_tolerance);
  adaptive_cfg.gp_shape_loo_max_tolerance =
    std::max(0.0, cfg.gp_shape_loo_max_tolerance);
  adaptive_cfg.gp_shape_loo_rms_tolerance =
    std::max(0.0, cfg.gp_shape_loo_rms_tolerance);
  adaptive_cfg.gp_integrated_variance_tolerance =
    std::max(0.0, cfg.gp_integrated_variance_tolerance);
  adaptive_cfg.gp_variance_min_relative_improvement =
    std::max(0.0, cfg.gp_variance_min_relative_improvement);
  adaptive_cfg.gp_variance_patience =
    std::max(0, cfg.gp_variance_patience);
  adaptive_cfg.gp_stop_hysteresis =
    std::max(1, cfg.gp_stop_hysteresis);
  adaptive_cfg.gp_probe_every = std::max(0, cfg.gp_probe_every);
  adaptive_cfg.gp_probe_min_samples = std::max(0, cfg.gp_probe_min_samples);
  adaptive_cfg.gp_probe_rms_tolerance = cfg.gp_probe_rms_tolerance;
  adaptive_cfg.gp_probe_require_readiness = cfg.gp_probe_require_readiness;
  adaptive_cfg.gp_probe_readiness_transition = cfg.gp_probe_readiness_transition;
  adaptive_cfg.gp_sync_certification_batches = cfg.gp_sync_certification_batches;
  adaptive_cfg.gp_async_postfit_certification = cfg.gp_async_postfit_certification;
  adaptive_cfg.gp_async_certification_require_readiness =
    cfg.gp_async_certification_require_readiness;
  adaptive_cfg.gp_async_start_min_valid = cfg.gp_async_start_min_valid;
  adaptive_cfg.gp_background_probes_off = cfg.gp_background_probes_off;
  adaptive_cfg.gp_probe_inflight_promotion = cfg.gp_probe_inflight_promotion;
  adaptive_cfg.gp_readiness_anticipation = cfg.gp_readiness_anticipation;
  adaptive_cfg.gp_async_early_brackets = cfg.gp_async_early_brackets;
  adaptive_cfg.bisection_tolerance = cfg.sim.bisection_tolerance;
  adaptive_cfg.bisection_relative_tolerance = cfg.sim.bisection_relative_tolerance;
  self->state().early_brackets_enabled =
    cfg.gp_async_early_brackets && cfg.adaptive_gp_mode == "asynchronous";
  self->state().early_bracket_sigma_k = cfg.gp_seed_sigma_k;
  self->state().early_bracket_abs_tol = cfg.sim.bisection_tolerance;
  self->state().early_bracket_rel_tol = cfg.sim.bisection_relative_tolerance;
  adaptive_cfg.gp_bracket_quench_cost_ratio = cfg.gp_bracket_quench_cost_ratio;
  adaptive_cfg.gp_probe_max_recertifications = cfg.gp_probe_max_recertifications;
  if (cfg.gp_probe_max_recertifications > 0)
    self->println(
      "bounded re-certification: after a failed certification set, at most {} "
      "fresh prospective probe set(s); failed sets are archived in "
      "gp_certification_sets.csv", cfg.gp_probe_max_recertifications);
  self->state().early_bracket_quench_cost_ratio = cfg.gp_bracket_quench_cost_ratio;
  if (cfg.gp_async_early_brackets)
    self->println(
      "Async extensions: early GP brackets={} (expected-cost rule at the GP "
      "median, k={}, tolerance max({}, {} relative), quench/ignite cost "
      "ratio {})",
      cfg.gp_async_early_brackets, cfg.gp_seed_sigma_k,
      cfg.sim.bisection_tolerance, cfg.sim.bisection_relative_tolerance,
      cfg.gp_bracket_quench_cost_ratio);
  if (adaptive_cfg.gp_probe_inflight_promotion ||
      adaptive_cfg.gp_readiness_anticipation)
    self->println(
      "adaptive surface GP: in-flight probe promotion={}, readiness "
      "anticipation={} ({}; Sync barriers have no in-flight work at a fit)",
      adaptive_cfg.gp_probe_inflight_promotion,
      adaptive_cfg.gp_readiness_anticipation, adaptive_cfg.mode);
  if (adaptive_cfg.gp_probe_readiness_transition)
    {
      // Normalize the complete opt-in policy in one place. Legacy flags in a
      // launcher cannot accidentally restore the old count-based rendezvous.
      adaptive_cfg.gp_probe_require_readiness = true;
      adaptive_cfg.gp_sync_certification_batches =
        adaptive_cfg.mode == "synchronous";
      adaptive_cfg.gp_async_postfit_certification =
        adaptive_cfg.mode == "asynchronous";
      adaptive_cfg.gp_async_certification_require_readiness =
        adaptive_cfg.mode == "asynchronous";
      self->println(
        "adaptive surface GP: readiness-driven certification enabled for {}; "
        "probe eligibility uses fresh ordinary-GP uncertainty/LOO diagnostics, "
        "not the coverage warm-up",
        adaptive_cfg.mode);
    }
  adaptive_cfg.gp_prequential_min_samples =
    std::max(0, cfg.gp_prequential_min_samples);
  adaptive_cfg.gp_prequential_coverage_lower_bound =
    cfg.gp_prequential_coverage_lower_bound;
  adaptive_cfg.gp_prequential_warmup_factor =
    cfg.gp_prequential_warmup_factor;
  adaptive_cfg.gp_convergence_r_min =
    cfg.gp_convergence_r_min > 0.0 ? cfg.gp_convergence_r_min : cfg.R_min;
  adaptive_cfg.gp_noise_floor = cfg.gp_noise_floor;
  adaptive_cfg.gp_target_transform = cfg.gp_target_transform;
  adaptive_cfg.gp_no_plot = cfg.gp_no_plot;
  adaptive_cfg.Aw = cfg.Aw;
  adaptive_cfg.gp_seeded_brackets = cfg.gp_seeded_brackets;
  adaptive_cfg.gp_seed_sigma_k = cfg.gp_seed_sigma_k;
  adaptive_cfg.gamma_wave_speeds = gamma_wave_speeds;
  adaptive_cfg.fallback_wave_speed = cfg.sim.fast_wave_speed;
  adaptive_cfg.allow_non_positive_wave_speed = cfg.fixed_final_time;
  adaptive_cfg.wave_speed_model = wave_speed_model;

  if (adaptive_cfg.surface && adaptive_cfg.gp_acquisition == "joint-nipv")
    throw std::runtime_error(
      "joint-nipv is not implemented for the two-input advisor; use "
      "max-std, greedy-nipv, or variation-nipv");

  self->state().gp_output_dir =
    cfg.gp_out_dir.empty()
      ? std::filesystem::path(cfg.sim.output_dir) / "gp_adaptive"
      : std::filesystem::path(cfg.gp_out_dir);
  if (cfg.adaptive_gp)
    {
      std::filesystem::create_directories(self->state().gp_output_dir);
      if (cfg.adaptive_gp_surface)
        {
          auto &surface_state = self->state().adaptive_surface_state;
          surface_state.gp_output_dir = self->state().gp_output_dir;
          surface_state.prequential_warmup_min = static_cast<int>(
            std::lround(cfg.gp_prequential_warmup_factor *
                        std::max(1, cfg.adaptive_initial_samples)));
          const auto probe_admissions_path = surface_state.gp_output_dir /
                                               "gp_probe_admissions.csv";
          surface_state.gp_probe_admissions_file.open(
            probe_admissions_path.string(), std::ios::out);
          if (!surface_state.gp_probe_admissions_file)
            throw std::runtime_error(
              "could not create " + probe_admissions_path.string());
          surface_state.gp_probe_admissions_file
            << "sample_id,gp_round,decision_round,R,gamma,"
               "gp_prediction_n_training,gp_prediction_readiness,"
               "gp_probe_certification_eligible,gp_prediction_available,"
               "gp_pred_mean,gp_pred_lower95,gp_pred_upper95\n"
            << std::setprecision(17);
          surface_state.gp_probe_admissions_file.flush();
          if (adaptive_cfg.gp_probe_max_recertifications > 0)
            {
              const auto path = surface_state.gp_output_dir /
                                "gp_certification_sets.csv";
              surface_state.certification_sets_file.open(path.string(),
                                                         std::ios::out);
              if (!surface_state.certification_sets_file)
                throw std::runtime_error("could not create " + path.string());
              surface_state.certification_sets_file
                << "set,outcome,n_scored,rms,max_error,max_error_sample_id,"
                   "sample_ids,application_elapsed_s,where\n"
                << std::setprecision(17);
              surface_state.certification_sets_file.flush();
            }
          if (adaptive_cfg.gp_async_early_brackets)
            {
              const auto early_bracket_path =
                surface_state.gp_output_dir / "gp_early_brackets.csv";
              surface_state.early_bracket_file.open(
                early_bracket_path.string(), std::ios::out);
              if (!surface_state.early_bracket_file)
                throw std::runtime_error(
                  "could not create " + early_bracket_path.string());
              surface_state.early_bracket_file
                << "sample_id,where,n_training,median,sigma,lo,hi,"
                   "gp_cost,global_cost,chosen\n"
                << std::setprecision(17);
              surface_state.early_bracket_file.flush();
            }
          if (adaptive_cfg.gp_probe_inflight_promotion)
            {
              const auto path = surface_state.gp_output_dir /
                                "gp_probe_promotions.csv";
              surface_state.gp_probe_promotions_file.open(path.string(),
                                                          std::ios::out);
              if (!surface_state.gp_probe_promotions_file)
                throw std::runtime_error("could not create " + path.string());
              surface_state.gp_probe_promotions_file
                << "sample_id,gp_round,decision_round,R,gamma,"
                   "admission_n_training,"
                   "admission_pred_mean,admission_pred_lower95,"
                   "admission_pred_upper95,promotion_n_training,"
                   "gp_pred_mean,gp_pred_lower95,gp_pred_upper95,"
                   "gp_posterior_std,application_elapsed_s\n"
                << std::setprecision(17);
              surface_state.gp_probe_promotions_file.flush();
            }
          if (adaptive_cfg.gp_readiness_anticipation)
            {
              const auto path = surface_state.gp_output_dir /
                                "gp_readiness_anticipation.csv";
              surface_state.gp_readiness_anticipation_file.open(path.string(),
                                                                std::ios::out);
              if (!surface_state.gp_readiness_anticipation_file)
                throw std::runtime_error("could not create " + path.string());
              surface_state.gp_readiness_anticipation_file
                << "decision_round,n_training,n_pending,fresh_ready,"
                   "max_now,rms_now,max_after_pending,rms_after_pending,"
                   "anticipated,scored,eligible_in_flight,promotable_in_flight,"
                   "capacity,probes_requested,application_elapsed_s\n"
                << std::setprecision(17);
              surface_state.gp_readiness_anticipation_file.flush();
            }
          const auto prequential_path = surface_state.gp_output_dir /
                                         "gp_surface_prequential_metrics.csv";
          surface_state.gp_prequential_file.open(
            prequential_path.string(), std::ios::out);
          if (!surface_state.gp_prequential_file)
            throw std::runtime_error(
              "could not create " + prequential_path.string());
          surface_state.gp_prequential_file
            << "round,n,rmse,mae,max_abs_error,mean_relative_error,"
               "max_relative_error,coverage_95\n"
            << std::setprecision(17);
          const auto metrics_path = surface_state.gp_output_dir /
                                    "gp_surface_round_metrics.csv";
          surface_state.gp_metrics_file.open(
            metrics_path.string(), std::ios::out);
          if (!surface_state.gp_metrics_file)
            throw std::runtime_error(
              "could not create " + metrics_path.string());
          surface_state.gp_metrics_file
            << "decision_round,scheduling_mode,n_completed,n_active,n_reserved,"
               "n_idle,n_pending_locations,n_requested,advisor_runtime_s,"
               "n_training,target_transform,"
               "input_transform_R,input_transform_gamma,input_warp,"
               "input_transform_selected_by,loo_log_probability,lengthscale_R,"
               "lengthscale_gamma,"
               "R_min,R_max,gamma_min,gamma_max,n_grid,max_model_std,"
               "rms_model_std,mean_model_variance,integrated_model_variance,"
               "max_relative_half_width95,rms_relative_half_width95,shape_loo_n,"
               "shape_loo_rms_relative_error,shape_loo_max_relative_error,"
               "shape_loo_worst_R,shape_loo_worst_gamma,"
               "interior_loo_n,interior_loo_rms_relative_error,"
               "interior_loo_max_relative_error,interior_loo_worst_R,"
               "interior_loo_worst_gamma,"
               "integrated_variance_relative_reduction,variance_stagnation_rounds,"
               "lower_R_boundary_supported,upper_R_boundary_supported,"
               "lower_gamma_boundary_supported,upper_gamma_boundary_supported,"
               "relative_uncertainty_ok,shape_resolution_ok,integrated_variance_ok,"
               "variance_stagnated,probe_n,probe_rms_relative_error,probe_accuracy_ok,"
               "prequential_n,prequential_coverage,"
               "prequential_coverage_lower95,prequential_calibration_ok,"
               "decision,decision_reason,probe_readiness_transition,"
               "probe_admission_ready\n"
            << std::setprecision(17);
        }
    }

  int id = 0;
  if (cfg.adaptive_gp_surface)
    {
      for (int i = 0; i < static_cast<int>(initial_surface_values.size()); ++i)
        {
          const auto [R, gamma] = initial_surface_values[i];
          auto s = make_sample(id++,
                               R,
                               cfg.Aw,
                               gamma,
                               wave_speed_for_gamma(
                                 gamma,
                                 gamma_wave_speeds,
                                 wave_speed_model.get(),
                                 cfg.sim.fast_wave_speed,
                                 cfg.fixed_final_time),
                               cfg.steepness,
                               cfg.ic_type,
                               cfg.sim.Av_lo,
                               cfg.sim.Av_hi);
          enqueue_sample(self, std::move(s));
        }
    }
  else
    for (const double R : initial_R_values)
      for (const double gamma : gamma_list)
        {
          auto s = make_sample(id++,
                               R,
                               cfg.Aw,
                               gamma,
                               wave_speed_for_gamma(
                                 gamma,
                                 gamma_wave_speeds,
                                 wave_speed_model.get(),
                                 cfg.sim.fast_wave_speed,
                                 cfg.fixed_final_time),
                               cfg.steepness,
                               cfg.ic_type,
                               cfg.sim.Av_lo,
                               cfg.sim.Av_hi);
          enqueue_sample(self, std::move(s));
        }

  if (port == 0)
    self->println("Server running in local-only mode");
  else
    self->println("Server running on port {}", port);
  if (cfg.adaptive_gp_surface)
    self->println(
      "Critical-amplitude surface: {} initial space-filling samples over "
      "R=[{}, {}], gamma=[{}, {}], Aw={}",
      self->state().samples.size(),
      adaptive_cfg.R_min,
      adaptive_cfg.R_max,
      adaptive_cfg.gamma_min,
      adaptive_cfg.gamma_max,
      adaptive_cfg.Aw);
  else
    self->println(
      "Critical-amplitude sweep: {} initial samples (R x gamma = {} x {}), Aw={}",
      self->state().samples.size(),
      initial_R_count,
      gamma_list.size(),
      adaptive_cfg.Aw);
  if (cfg.adaptive_gp)
    {
      self->println(
        "Adaptive GP: enabled, parameterization={}, mode={}, maximum updates={}, maximum samples={}, "
        "batch size={}, acquisition={}, variation weight={}, focus uncertainty={}, "
        "focus variation coverage={}, integration grid={}, acquisition "
        "chunk={}, joint restarts/raw/maxiter/retries={}/{}/{}/{}, "
        "joint fallback={}, stop hysteresis={}, GP outputs={}",
        "joint (R,gamma) surface",
        adaptive_cfg.mode,
        adaptive_cfg.max_rounds,
        adaptive_cfg.max_samples == 0
          ? std::string("unlimited")
          : std::to_string(adaptive_cfg.max_samples),
        adaptive_cfg.batch_size,
        adaptive_cfg.gp_acquisition,
        adaptive_cfg.gp_variation_weight,
        adaptive_cfg.gp_focus_relative_uncertainty_target,
        adaptive_cfg.gp_focus_variation_coverage,
        adaptive_cfg.gp_integration_grid_size,
        adaptive_cfg.gp_acquisition_chunk_size,
        adaptive_cfg.gp_joint_num_restarts,
        adaptive_cfg.gp_joint_raw_samples,
        adaptive_cfg.gp_joint_maxiter,
        adaptive_cfg.gp_joint_retries,
        adaptive_cfg.gp_joint_fallback,
        adaptive_cfg.gp_stop_hysteresis,
        self->state().gp_output_dir.string());
      self->println(
        "Adaptive GP convergence: R=[{}, {}], gamma=[{}, {}], maximum/RMS physical relative "
        "95% half-width <= {:.3f}%/{:.3f}%, local shape LOO max/RMS <= "
        "{:.3f}%/{:.3f}%, integrated model variance <= {:.6g}, "
        "boundary distance <= {}",
        adaptive_cfg.gp_convergence_r_min,
        adaptive_cfg.R_max,
        adaptive_cfg.gamma_min,
        adaptive_cfg.gamma_max,
        100.0 * adaptive_cfg.gp_relative_uncertainty_max_tolerance,
        100.0 * adaptive_cfg.gp_relative_uncertainty_rms_tolerance,
        100.0 * adaptive_cfg.gp_shape_loo_max_tolerance,
        100.0 * adaptive_cfg.gp_shape_loo_rms_tolerance,
        adaptive_cfg.gp_integrated_variance_tolerance,
        adaptive_cfg.gp_surface_suggest_min_distance);
      self->println(
        "Adaptive GP safety stops: maximum updates={}, maximum samples={}, "
        "integrated-variance minimum improvement={:.3f}%, patience={}; "
        "prequential RMSE is diagnostic only",
        adaptive_cfg.max_rounds,
        adaptive_cfg.max_samples == 0
          ? std::string("unlimited")
          : std::to_string(adaptive_cfg.max_samples),
        100.0 * adaptive_cfg.gp_variance_min_relative_improvement,
        adaptive_cfg.gp_variance_patience);
      self->println(
        "Adaptive surface GP: normalized spacing={}, candidates={}, "
        "prediction grid={}x{}, outputs={}",
        adaptive_cfg.gp_surface_suggest_min_distance,
        adaptive_cfg.gp_surface_candidate_count,
        adaptive_cfg.gp_surface_prediction_r_count,
        adaptive_cfg.gp_surface_prediction_gamma_count,
        self->state().gp_output_dir.string());
      if (adaptive_cfg.mode == "asynchronous")
        {
          self->println(
            "Adaptive surface GP refill policy: fit after new completed "
            "thresholds; target active/reserved={} (0 = fill idle workers), "
            "burst cap={} (0 = uncapped); adaptive batch size applies only "
            "to synchronous mode",
            adaptive_cfg.async_target_active,
            adaptive_cfg.async_fill_cap);
          self->println(
            "Adaptive GP cancellation policy: {} active simulations when a "
            "GP stopping criterion is reached",
            adaptive_cfg.cancel_running_on_stop ? "cancel" : "drain");
        }
    }
  self->println("Bisection: Av=[{:.6f}, {:.6f}], tolerance={}, max iterations={}",
                cfg.sim.Av_lo,
                cfg.sim.Av_hi,
                cfg.sim.bisection_tolerance,
                cfg.sim.bisection_max_iterations);
  self->println("Results: {}", self->state().results_path.string());
  self->println("Solver: {} backend, {} via {}",
                cfg.sim.solver_backend,
                cfg.sim.solver_binary,
                cfg.sim.solver_runner.empty() ? "direct" : cfg.sim.solver_runner);

  const auto sim_params = cfg.sim;
  const double finite_time_safety_factor =
    cfg.sim.finite_time_safety_factor;
  const double domain_half_length = cfg.sim.domain_half_length;
  const double configured_final_time = cfg.sim.final_time;

  return {
    [=](actor remote, int) {
      anon_mail(sim_params).send(remote);
    },

    [=](actor worker) {
      self->state().active_workers.insert(worker);
      if (!self->state().pending_jobs.empty())
        {
          auto job = self->state().pending_jobs.front();
          self->state().pending_jobs.pop_front();
          dispatch_job(self, worker, std::move(job));
        }
      else
        {
          self->state().idle_workers.insert(worker);
        }
    },

    [=](const std::string &msg,
        int sample_id,
        bool cancelled,
        bool threshold_found,
        bool converged,
        double critical_Av,
        double Av_lo,
        double Av_hi,
        unsigned int iterations,
        unsigned int solver_runs,
        bool bracket_ends_observed,
        double runtime_s,
        const std::string &run_dir,
        const std::string &error,
        actor worker) {
      if (msg != "result")
        return;

      --self->state().active_jobs;
      self->state().worker_active_samples.erase(worker);
      if (cancelled)
        ++self->state().total_jobs_cancelled;
      else
        ++self->state().total_jobs_completed;

      auto &s = self->state().samples.at(sample_id);
      s.completed = true;
      s.cancelled = cancelled;
      s.failed = !cancelled && !error.empty();
      s.threshold_found = !cancelled && threshold_found;
      s.converged = !cancelled && converged;
      s.critical_Av = critical_Av;
      s.Av_lo = Av_lo;
      s.Av_hi = Av_hi;
      s.iterations = iterations;
      s.solver_runs = solver_runs;
      s.bracket_ends_observed = bracket_ends_observed;
      s.error = error;
      s.run_dir = run_dir;

      const bool discard_upper_bound_not_igniting =
        !s.cancelled && !s.failed && !s.threshold_found;

      std::string status;
      if (s.cancelled)
        status = "cancelled_after_adaptive_stop";
      else if (s.failed)
        status = "failed";
      else if (discard_upper_bound_not_igniting)
        status = "discarded_upper_bound_not_igniting";
      else if (s.converged)
        status = "converged";
      else
        status = "max_iterations";

      if (s.threshold_found)
        self->println(
          "result: sample={} (R={:.4f} Aw={:.4f} gamma={:.5f}) -> "
          "critical Av={:.8f}, {} ({} runs, {:.1f}s)",
          sample_id,
          s.R,
          s.Aw,
          s.gamma,
          s.critical_Av,
          status,
          s.solver_runs,
          runtime_s);
      else
        self->println(
          "result: sample={} (R={:.4f} Aw={:.4f} gamma={:.5f}) -> {} "
          "({} runs, {:.1f}s)",
          sample_id,
          s.R,
          s.Aw,
          s.gamma,
          status,
          s.solver_runs,
          runtime_s);

      const bool has_gp_validation =
        s.gp_prediction_available && s.threshold_found && !s.failed &&
        !s.cancelled;
      const double gp_error =
        has_gp_validation ? s.gp_pred_mean - s.critical_Av : 0.0;
      const double gp_abs_error =
        has_gp_validation ? std::abs(gp_error) : 0.0;
      const double gp_relative_error =
        has_gp_validation
          ? gp_abs_error / std::max(std::abs(s.critical_Av), 1.0e-300)
          : 0.0;
      const bool gp_within_95ci =
        has_gp_validation && s.gp_pred_lower95 <= s.critical_Av &&
        s.critical_Av <= s.gp_pred_upper95;

      if (has_gp_validation)
        self->println(
          "adaptive GP validation: round={} sample={} R={:.6f} "
          "pred={:.8f}, actual={:.8f}, abs error={:.6g}, rel error={:.3f}%, "
          "inside 95%={}",
          s.gp_round,
          sample_id,
          s.R,
          s.gp_pred_mean,
          s.critical_Av,
          gp_abs_error,
          100.0 * gp_relative_error,
          gp_within_95ci ? "true" : "false");

      if (!discard_upper_bound_not_igniting)
        {
          const double planned_final_time = s.fast_wave_speed > 0.0
            ? finite_time_safety_factor *
                (domain_half_length - s.R) / s.fast_wave_speed
            : configured_final_time;
          self->state().results_file
            << sample_id << ","
            << s.R << ","
            << s.Aw << ","
            << s.gamma << ","
            << s.fast_wave_speed << ","
            << planned_final_time << ","
            << s.steepness << ","
            << csv_escape(s.ic_type) << ",";
          if (s.threshold_found)
            self->state().results_file << s.critical_Av;
          self->state().results_file
            << ","
            << s.Av_lo << ","
            << s.Av_hi << ","
            << (s.Av_hi - s.Av_lo) << ","
            << s.iterations << ","
              << s.solver_runs << ","
              << (s.converged ? "true" : "false") << ","
              << (s.cancelled ? "true" : "false") << ","
              << status << ","
            << runtime_s << ","
            << csv_escape(s.run_dir) << ","
            << csv_escape(s.error) << ","
            << s.gp_round << ",";
          if (s.gp_prediction_available)
            self->state().results_file
              << s.gp_pred_mean << ","
              << s.gp_pred_lower95 << ","
              << s.gp_pred_upper95 << ","
              << s.gp_posterior_std << ",";
          else
            self->state().results_file << ",,,,";
          if (has_gp_validation)
            self->state().results_file
              << gp_abs_error << ","
              << gp_relative_error << ","
              << (gp_within_95ci ? "true" : "false");
          else
            self->state().results_file << ",,";
          // Bracket the sample was dispatched with, for the seeded-vs-global
          // A/B: solver_runs above is the cost, these two say what was tried.
          self->state().results_file
            << "," << (s.gp_seeded_bracket ? "true" : "false")
            << "," << s.seed_Av_lo
            << "," << s.seed_Av_hi
            << "," << (s.gp_is_probe ? "true" : "false")
            << "," << s.gp_prediction_n_training
            << "," << (s.gp_prediction_readiness ? "true" : "false")
            << "," << (s.gp_probe_certification_eligible ? "true" : "false");
          self->state().results_file
            << "," << csv_escape(s.bracket_source);
          self->state().results_file << "\n";
          self->state().results_file.flush();
        }

      // Early GP brackets: refresh the model before this worker takes the
      // next queued seed, so that seed can be dispatched with a GP bracket.
      maybe_fit_surface_for_brackets(self, adaptive_cfg);
      self->state().idle_workers.insert(worker);
      wake_idle_workers(self);

      check_termination(self, adaptive_cfg);
    },

    [=](const std::string &msg, actor worker, actor controller) {
      if (msg == "cancellation_controller")
        {
          self->state().cancellation_controllers.insert(controller);
          self->state().worker_cancellation_controllers[worker] = controller;
          return;
        }
    },

    [=](const std::string &msg, actor worker) {
      if (msg != "quit")
        return;
      self->state().active_workers.erase(worker);
      self->state().idle_workers.erase(worker);
      self->state().worker_active_samples.erase(worker);
      self->state().worker_cancellation_controllers.erase(worker);
      if (self->state().active_workers.empty() &&
          self->state().idle_workers.empty())
        {
          self->println("All workers done. Shutting down.");
          self->quit();
        }
    }
  };
}
