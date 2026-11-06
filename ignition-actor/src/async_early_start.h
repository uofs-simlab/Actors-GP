#pragma once

#include <algorithm>
#include <cmath>

// Conservative, dimension-specific gate for the two-input surface model.
// Twelve is a conservative pilot minimum above the advisor's eight-point
// representation-selection floor, not a theorem about GP accuracy. Requiring
// at least half the original design and all four corners limits extrapolation
// from a runtime-biased completed subset. No validation responses set this gate.
inline int async_early_start_minimum(const int initial)
{
  if (initial <= 0)
    return 0;
  return std::min(initial, std::max(12, initial / 2 + initial % 2));
}

inline unsigned async_initial_corner_bit(
  const double R, const double gamma, const double Rmin, const double Rmax,
  const double gmin, const double gmax)
{
  if (!std::isfinite(R) || !std::isfinite(gamma) ||
      !std::isfinite(Rmin) || !std::isfinite(Rmax) ||
      !std::isfinite(gmin) || !std::isfinite(gmax) ||
      !(Rmax > Rmin) || !(gmax > gmin))
    return 0;
  const double Rspan = Rmax - Rmin;
  const double gspan = gmax - gmin;
  if (!std::isfinite(Rspan) || !std::isfinite(gspan))
    return 0;
  const double u = (R - Rmin) / Rspan;
  const double v = (gamma - gmin) / gspan;
  // Round-off tolerance only; this is not the broader stopping edge radius.
  constexpr double eps = 1e-10;
  const int i = std::abs(u) <= eps ? 0 : (std::abs(u - 1) <= eps ? 1 : -1);
  const int j = std::abs(v) <= eps ? 0 : (std::abs(v - 1) <= eps ? 1 : -1);
  return i < 0 || j < 0 ? 0u : (1u << (2 * i + j));
}

inline bool async_early_start_ready(
  const int initial, const int valid_seed, const unsigned corner_mask)
{
  return initial > 0 && valid_seed >= async_early_start_minimum(initial) &&
         (corner_mask & 15u) == 15u;
}
