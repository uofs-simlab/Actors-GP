#ifndef RADIAL_IGNITION_WAVE_SPEED_MODEL_H
#define RADIAL_IGNITION_WAVE_SPEED_MODEL_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Interpolates the fast (stable) FHN pulse branch from a numerical
// continuation file. The supplied continuation format has gamma in column 1
// and wave speed in column 6. It starts on the slow branch, turns at the
// maximum gamma, and then follows the fast branch. We therefore retain the
// records after the maximum-gamma row and sort them by gamma.
//
// Below min_reliable_gamma, finite-domain continuation data may bend away from
// the infinite-domain fast-wave plateau. A positive small_gamma_plateau is
// used there instead. A positive round_up_step rounds interpolated speeds
// upward, which is appropriate when the speed defines a pre-boundary final
// time and must be a conservative upper estimate.
class fhn_fast_wave_speed_model
{
public:
  fhn_fast_wave_speed_model() = default;

  fhn_fast_wave_speed_model(const std::string &path,
                            const double       min_reliable_gamma,
                            const double       small_gamma_plateau,
                            const double       round_up_step)
    : min_reliable_gamma_(min_reliable_gamma)
    , small_gamma_plateau_(small_gamma_plateau)
    , round_up_step_(round_up_step)
  {
    load(path);
  }

  bool empty() const
  {
    return fast_branch_.empty();
  }

  double min_gamma() const
  {
    ensure_loaded();
    return fast_branch_.front().first;
  }

  double max_gamma() const
  {
    ensure_loaded();
    return fast_branch_.back().first;
  }

  double speed(const double gamma) const
  {
    ensure_loaded();
    if (!std::isfinite(gamma) || gamma <= 0.0)
      throw std::runtime_error("wave-speed interpolation requires gamma > 0");

    if (min_reliable_gamma_ > 0.0 && gamma < min_reliable_gamma_)
      {
        if (!(small_gamma_plateau_ > 0.0) ||
            !std::isfinite(small_gamma_plateau_))
          throw std::runtime_error(
            "gamma is below the reliable continuation range and no positive "
            "small-gamma plateau was configured");
        return rounded_up(small_gamma_plateau_);
      }

    const double lower_limit = std::max(min_reliable_gamma_, min_gamma());
    if (gamma < lower_limit)
      {
        if (small_gamma_plateau_ > 0.0)
          return rounded_up(small_gamma_plateau_);
        throw std::runtime_error(
          "gamma is below the available fast-wave continuation range");
      }
    if (gamma > max_gamma())
      throw std::runtime_error(
        "gamma is above the available fast-wave continuation range");

    const auto upper = std::lower_bound(
      fast_branch_.begin(), fast_branch_.end(), gamma,
      [](const auto &entry, const double value) {
        return entry.first < value;
      });
    if (upper == fast_branch_.begin())
      return rounded_up(upper->second);
    if (upper == fast_branch_.end())
      return rounded_up(fast_branch_.back().second);
    if (nearly_equal(upper->first, gamma))
      return rounded_up(upper->second);

    const auto lower = std::prev(upper);
    const double fraction =
      (gamma - lower->first) / (upper->first - lower->first);
    const double interpolated =
      lower->second + fraction * (upper->second - lower->second);
    return rounded_up(interpolated);
  }

private:
  std::vector<std::pair<double, double>> fast_branch_;
  double min_reliable_gamma_ = 0.0;
  double small_gamma_plateau_ = 0.0;
  double round_up_step_ = 0.0;

  static bool nearly_equal(const double a, const double b)
  {
    return std::abs(a - b) <=
           1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
  }

  double rounded_up(const double value) const
  {
    if (!(round_up_step_ > 0.0))
      return value;
    // Subtract a few ulps before ceil so a value already on a rounding grid is
    // not spuriously promoted by its binary representation.
    const double scaled = value / round_up_step_;
    const double tolerance =
      8.0 * std::numeric_limits<double>::epsilon() * std::abs(scaled);
    return std::ceil(scaled - tolerance) * round_up_step_;
  }

  void ensure_loaded() const
  {
    if (fast_branch_.size() < 2)
      throw std::runtime_error("fast-wave continuation model is not loaded");
  }

  void load(const std::string &path)
  {
    std::ifstream stream(path);
    if (!stream)
      throw std::runtime_error("could not open wave-speed continuation: " + path);

    std::vector<std::pair<double, double>> full_curve;
    std::string line;
    while (std::getline(stream, line))
      {
        std::istringstream row(line);
        double columns[7] = {};
        bool valid = true;
        for (double &column : columns)
          if (!(row >> column))
            {
              valid = false;
              break;
            }
        if (!valid)
          continue;
        const double gamma = columns[0];
        const double speed = columns[5];
        if (std::isfinite(gamma) && gamma > 0.0 &&
            std::isfinite(speed) && speed > 0.0)
          full_curve.emplace_back(gamma, speed);
      }
    if (full_curve.size() < 4)
      throw std::runtime_error(
        "wave-speed continuation has too few valid numeric rows: " + path);

    const auto fold = std::max_element(
      full_curve.begin(), full_curve.end(),
      [](const auto &a, const auto &b) { return a.first < b.first; });
    if (fold == full_curve.end() || std::next(fold) == full_curve.end())
      throw std::runtime_error(
        "wave-speed continuation does not contain a post-fold fast branch");

    fast_branch_.assign(std::next(fold), full_curve.end());
    std::sort(fast_branch_.begin(), fast_branch_.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    fast_branch_.erase(
      std::unique(
        fast_branch_.begin(), fast_branch_.end(),
        [](const auto &a, const auto &b) { return nearly_equal(a.first, b.first); }),
      fast_branch_.end());
    ensure_loaded();
  }
};

#endif
