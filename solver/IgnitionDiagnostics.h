#ifndef TOSTII_MONODOMAIN_IGNITION_DIAGNOSTICS_H
#define TOSTII_MONODOMAIN_IGNITION_DIAGNOSTICS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>

namespace IgnitionDiagnostics
{
  inline constexpr double area_epsilon = 1e-14;

  struct Criteria
  {
    double diagnosis_start_time;
    double sustain_window;
    double area_decay_tolerance;
    double max_v_threshold;
    double excited_fraction_threshold;
  };

  /**
   * Tracks whether excitation is sustained over a physical time window.
   *
   * A hit requires every sampled area in the window to avoid a decrease
   * larger than the configured tolerance, and the newest area to remain
   * within that tolerance of the largest area observed in the window. This
   * rejects both abrupt and gradual collapse without tying the decision to a
   * solver-dependent number of diagnostic samples.
   */
  class State
  {
  public:
    void update(const double   time,
                const double   excited_area,
                const double   excited_fraction,
                const double   max_v,
                const Criteria criteria)
    {
      const double time_tolerance =
        1e-12 * std::max(1.0, std::abs(time));
      if (time + time_tolerance < criteria.diagnosis_start_time)
        return;

      ++samples_after_start;
      max_v_after_start = std::max(max_v_after_start, max_v);
      max_excited_fraction_after_start =
        std::max(max_excited_fraction_after_start, excited_fraction);

      area_history.push_back({time, excited_area});
      const double cutoff = time - criteria.sustain_window;

      // Retain the newest sample at or before the window boundary. This
      // supplies a real baseline spanning at least sustain_window seconds.
      while (area_history.size() > 1 &&
             area_history[1].time <= cutoff + time_tolerance)
        area_history.pop_front();

      window_ready_at_latest =
        !area_history.empty() &&
        time - area_history.front().time + time_tolerance >=
          criteria.sustain_window;

      bool area_is_noncollapsing = window_ready_at_latest;
      if (area_is_noncollapsing)
        {
          double previous_area = area_history.front().area;
          double peak_area     = previous_area;

          for (std::size_t i = 1; i < area_history.size(); ++i)
            {
              const double area = area_history[i].area;
              if (area < previous_area *
                           (1.0 - criteria.area_decay_tolerance))
                area_is_noncollapsing = false;
              previous_area = area;
              peak_area     = std::max(peak_area, area);
            }

          if (excited_area <
              peak_area * (1.0 - criteria.area_decay_tolerance))
            area_is_noncollapsing = false;
        }

      sustained_at_latest =
        area_is_noncollapsing &&
        excited_fraction >= criteria.excited_fraction_threshold &&
        max_v >= criteria.max_v_threshold;
      ignition_occurred = ignition_occurred || sustained_at_latest;
    }

    bool diagnosis_window_reached() const
    {
      return samples_after_start > 0;
    }

    unsigned int samples_after_start = 0;
    double max_v_after_start = -std::numeric_limits<double>::infinity();
    double max_excited_fraction_after_start = 0.0;
    bool   window_ready_at_latest = false;
    bool   sustained_at_latest    = false;
    bool   ignition_occurred      = false;

  private:
    struct AreaSample
    {
      double time;
      double area;
    };

    std::deque<AreaSample> area_history;
  };
} // namespace IgnitionDiagnostics

#endif
