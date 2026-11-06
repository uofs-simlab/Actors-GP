#ifndef TOSTII_MONODOMAIN_INITIAL_CONDITION_FACTORY_H
#define TOSTII_MONODOMAIN_INITIAL_CONDITION_FACTORY_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace InitialConditions
{
  using namespace dealii;

  struct Parameters
  {
    std::string type = "disk";
    double      center_x = 0.0;
    double      center_y = 0.0;
    double      amplitude_v = 0.2;
    double      amplitude_w = 0.0;
    double      radius = 2.0;
    double      gaussian_width = 1.0;
    double      steepness_x = 2.0;
    double      steepness_y = 2.0;
    // wave_segment geometry (see WaveSegmentInitialCondition)
    double      segment_length = 20.0;
    double      tail_width     = 20.0;

    static void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial condition");
      {
        prm.declare_entry("Type",
                          "disk",
                          Patterns::Selection(
                            "zero|constant|disk|smooth_disk|stripe|smooth_stripe|band|smooth_band|gaussian|ring|broken_wave|spiral|wave_segment"),
                          "Initial-condition type.");
        prm.declare_entry("Center x",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition center x-coordinate.");
        prm.declare_entry("Center y",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition center y-coordinate.");
        prm.declare_entry("Amplitude v",
                          "0.2",
                          Patterns::Double(),
                          "Initial-condition amplitude for v.");
        prm.declare_entry("Amplitude w",
                          "0.0",
                          Patterns::Double(),
                          "Initial-condition amplitude for w.");
        prm.declare_entry("Radius",
                          "2.0",
                          Patterns::Double(0.0),
                          "Disk radius or stripe width.");
        prm.declare_entry("Gaussian width",
                          "1.0",
                          Patterns::Double(0.0),
                          "Gaussian width parameter.");
        prm.declare_entry("Steepness x",
                          "2.0",
                          Patterns::Double(0.0),
                          "Steepness of the x-directed broken-wave interface.");
        prm.declare_entry("Steepness y",
                          "2.0",
                          Patterns::Double(0.0),
                          "Steepness of the y-directed refractory interface.");
        prm.declare_entry("Segment length",
                          "20.0",
                          Patterns::Double(0.0),
                          "wave_segment: length of the excited segment along y "
                          "(the free-end separation).");
        prm.declare_entry("Tail width",
                          "20.0",
                          Patterns::Double(0.0),
                          "wave_segment: x-extent of the refractory (w) tail "
                          "behind the excited band.");
      }
      prm.leave_subsection();
    }

    void parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial condition");
      {
        type           = prm.get("Type");
        center_x       = prm.get_double("Center x");
        center_y       = prm.get_double("Center y");
        amplitude_v    = prm.get_double("Amplitude v");
        amplitude_w    = prm.get_double("Amplitude w");
        radius         = prm.get_double("Radius");
        gaussian_width = prm.get_double("Gaussian width");
        steepness_x    = prm.get_double("Steepness x");
        steepness_y    = prm.get_double("Steepness y");
        segment_length = prm.get_double("Segment length");
        tail_width     = prm.get_double("Tail width");
      }
      prm.leave_subsection();
    }
  };

  template <int dim>
  class ZeroInitialCondition : public Function<dim>
  {
  public:
    ZeroInitialCondition()
      : Function<dim>(2)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/) const override
    {
      return 0.0;
    }
  };

  template <int dim>
  class ConstantInitialCondition : public Function<dim>
  {
  public:
    explicit ConstantInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component) const override
    {
      if (component == 0)
        return prm.amplitude_v;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  class DiskInitialCondition : public Function<dim>
  {
  public:
    explicit DiskInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double dx = p[0] - prm.center_x;
      const double dy = p[1] - prm.center_y;
      const double r  = std::sqrt(dx * dx + dy * dy);

      const double profile = (r <= prm.radius) ? 1.0 : 0.0;

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  class GaussianInitialCondition : public Function<dim>
  {
  public:
    explicit GaussianInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double dx = p[0] - prm.center_x;
      const double dy = p[1] - prm.center_y;
      const double width = std::max(prm.gaussian_width, 1.0e-12);
      const double profile =
        std::exp(-(dx * dx + dy * dy) / (width * width));

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  // Smooth disk: v(r) = amplitude_v * 0.5 * (1 - tanh(k * (r - R)))
  // Plateau at amplitude_v inside r=R, smooth rolloff over width ~2/k outside.
  // Steepness x as k (larger = sharper edge, smaller = more gradual).
  template <int dim>
  class SmoothDiskInitialCondition : public Function<dim>
  {
  public:
    explicit SmoothDiskInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double dx = p[0] - prm.center_x;
      const double dy = p[1] - prm.center_y;
      const double r  = std::sqrt(dx * dx + dy * dy);
      const double k  = std::max(prm.steepness_x, 1.0e-12);
      const double profile = 0.5 * (1.0 - std::tanh(k * (r - prm.radius)));

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  // One-sided planar stripe:
  //   sharp:  v = amplitude_v for x <= center_x + radius, else 0
  //   smooth: v = amplitude_v * 0.5 * (1 - tanh(k * (x - edge_x)))
  // Set center_x to the left boundary and radius to the stripe width to
  template <int dim>
  class StripeInitialCondition : public Function<dim>
  {
  public:
    explicit StripeInitialCondition(const Parameters &parameters,
                                    const bool        smooth_profile)
      : Function<dim>(2)
      , prm(parameters)
      , smooth(smooth_profile)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double edge_x = prm.center_x + prm.radius;
      double       profile = 0.0;

      if (smooth)
        {
          const double k = std::max(prm.steepness_x, 1.0e-12);
          profile = 0.5 * (1.0 - std::tanh(k * (p[0] - edge_x)));
        }
      else
        profile = (p[0] <= edge_x) ? 1.0 : 0.0;

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
    const bool        smooth;
  };

  // Two-sided localized band:
  //   sharp:  v = amplitude_v for |x - center_x| <= radius, else 0
  //   smooth: v = amplitude_v * 0.5*(1+tanh(k*(x-left))) * 0.5*(1-tanh(k*(x-right)))
  // radius is the half-width of the band. Narrow bands at large amplitude
  template <int dim>
  class BandInitialCondition : public Function<dim>
  {
  public:
    explicit BandInitialCondition(const Parameters &parameters,
                                  const bool        smooth_profile)
      : Function<dim>(2)
      , prm(parameters)
      , smooth(smooth_profile)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double left_edge  = prm.center_x - prm.radius;
      const double right_edge = prm.center_x + prm.radius;
      double       profile    = 0.0;

      if (smooth)
        {
          const double k = std::max(prm.steepness_x, 1.0e-12);
          profile = 0.5 * (1.0 + std::tanh(k * (p[0] - left_edge))) *
                    0.5 * (1.0 - std::tanh(k * (p[0] - right_edge)));
        }
      else
        profile = (std::abs(p[0] - prm.center_x) <= prm.radius) ? 1.0 : 0.0;

      if (component == 0)
        return prm.amplitude_v * profile;
      if (component == 1)
        return prm.amplitude_w;

      return 0.0;
    }

  private:
    const Parameters &prm;
    const bool        smooth;
  };

  template <int dim>
  class BrokenWaveInitialCondition : public Function<dim>
  {
  public:
    explicit BrokenWaveInitialCondition(const Parameters &parameters)
      : Function<dim>(2)
      , prm(parameters)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component) const override
    {
      const double x = p[0] - prm.center_x;
      const double y = p[1] - prm.center_y;

      const double kx = std::max(prm.steepness_x, 1.0e-12);
      const double ky = std::max(prm.steepness_y, 1.0e-12);

      const double v_profile = 0.5 * (1.0 - std::tanh(kx * x));
      const double w_profile = 0.5 * (1.0 - std::tanh(ky * y));

      if (component == 0)
        return prm.amplitude_v * v_profile;
      if (component == 1)
        return prm.amplitude_w * w_profile;

      return 0.0;
    }

  private:
    const Parameters &prm;
  };


  // ---------------------------------------------------------------------
  // wave_segment: a FINITE excited wave segment with two free ends, plus a
  // refractory tail behind it -- the canonical spiral-initiation geometry.
  //
  //   v = amplitude_v  inside  x in [cx - radius, cx],  |y - cy| <= L/2
  //   w = amplitude_w  inside  x in [cx - radius - tail, cx - radius],
  //                            |y - cy| <= L/2
  //
  // The excited band (width `radius`) is the wave; the `w` block immediately
  // behind it is refractory, so the wave can only propagate in +x. Truncating
  // both blocks at |y - cy| = L/2 (L = `segment_length`) leaves two FREE ENDS.
  //
  // Edges are smoothed with tanh: steepness_x in x, steepness_y in y.
  template <int dim>
  class WaveSegmentInitialCondition : public Function<dim>
  {
  public:
    explicit WaveSegmentInitialCondition(const Parameters &prm)
      : Function<dim>(2)
      , prm(prm)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double kx = std::max(prm.steepness_x, 1.0e-12);
      const double ky = std::max(prm.steepness_y, 1.0e-12);
      const double x  = p[0] - prm.center_x;
      const double y  = (dim > 1) ? (p[1] - prm.center_y) : 0.0;

      // smooth indicator of the interval [a, b]
      const auto box = [](const double t, const double a, const double b,
                          const double k) {
        return 0.5 * (std::tanh(k * (t - a)) - std::tanh(k * (t - b)));
      };

      const double half_len = 0.5 * prm.segment_length;
      const double along_y  = box(y, -half_len, half_len, ky);

      if (component == 0)
        return prm.amplitude_v * box(x, -prm.radius, 0.0, kx) * along_y;
      if (component == 1)
        return prm.amplitude_w *
               box(x, -prm.radius - prm.tail_width, -prm.radius, kx) * along_y;
      return 0.0;
    }

  private:
    const Parameters &prm;
  };

  template <int dim>
  std::unique_ptr<Function<dim>>
  make_initial_condition(const Parameters &prm)
  {
    if (prm.type == "zero")
      return std::make_unique<ZeroInitialCondition<dim>>();
    if (prm.type == "constant")
      return std::make_unique<ConstantInitialCondition<dim>>(prm);
    if (prm.type == "disk")
      return std::make_unique<DiskInitialCondition<dim>>(prm);
    if (prm.type == "smooth_disk")
      return std::make_unique<SmoothDiskInitialCondition<dim>>(prm);
    if (prm.type == "stripe")
      return std::make_unique<StripeInitialCondition<dim>>(prm, false);
    if (prm.type == "smooth_stripe")
      return std::make_unique<StripeInitialCondition<dim>>(prm, true);
    if (prm.type == "band")
      return std::make_unique<BandInitialCondition<dim>>(prm, false);
    if (prm.type == "smooth_band")
      return std::make_unique<BandInitialCondition<dim>>(prm, true);
    if (prm.type == "gaussian")
      return std::make_unique<GaussianInitialCondition<dim>>(prm);
    if (prm.type == "broken_wave" || prm.type == "spiral")
      return std::make_unique<BrokenWaveInitialCondition<dim>>(prm);
    if (prm.type == "wave_segment")
      return std::make_unique<WaveSegmentInitialCondition<dim>>(prm);

    throw std::runtime_error("Unsupported initial-condition type: " + prm.type);
  }
} // namespace InitialConditions

#endif
