#pragma once

#include <cmath>

// Match InputSpace/DuplicateGuard: physical inputs mapped to the selected
// linear/log coordinates and normalized by domain spans, before input warp.
inline double surface_model_distance(
  const double Ra, const double ga, const double Rb, const double gb,
  const double Rmin, const double Rmax, const double gmin, const double gmax,
  const bool log_R, const bool log_gamma)
{
  const double dR = log_R && Ra > 0 && Rb > 0 && Rmin > 0
    ? (std::log(Ra) - std::log(Rb)) / (std::log(Rmax) - std::log(Rmin))
    : (Ra - Rb) / (Rmax - Rmin);
  const double dg = log_gamma && ga > 0 && gb > 0 && gmin > 0
    ? (std::log(ga) - std::log(gb)) / (std::log(gmax) - std::log(gmin))
    : (ga - gb) / (gmax - gmin);
  return std::hypot(dR, dg);
}
