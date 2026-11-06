/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2025 by the authors listed below
 *
 * This file is NOT part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Mohammad Mahdi Moayeri, University of Saskatchewan, 2026
 *
 * ---------------------------------------------------------------------
 * Parallel fully implicit FitzHugh-Nagumo solver using PETSc TS BDF
 * Uses MPI parallel computing and PETSc BDF time integration
 * The IGNITION_RADIAL_BDF build uses the same BDF implementation on r in
 * [0,L], with the cylindrical r-weight in every weak-form integral.
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_ts.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "IgnitionDiagnostics.h"
#include "InitialConditionFactory.h"

using namespace dealii;

namespace
{
  using VectorType = PETScWrappers::MPI::Vector;
  using MatrixType = PETScWrappers::MPI::SparseMatrix;

  /**
   * parallel::distributed::Triangulation is not available in one space
   * dimension.  This small adapter lets the BDF implementation use a serial
   * triangulation for the radial (dim=1) executable and retain the distributed
   * triangulation for the Cartesian executable.
   */
  template <int dim>
  class BdfTriangulation : public parallel::distributed::Triangulation<dim>
  {
  public:
    explicit BdfTriangulation(const MPI_Comm communicator)
      : parallel::distributed::Triangulation<dim>(
          communicator,
          typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening))
    {}
  };

  template <>
  class BdfTriangulation<1> : public Triangulation<1>
  {
  public:
    explicit BdfTriangulation(const MPI_Comm /*communicator*/)
      : Triangulation<1>(
          typename Triangulation<1>::MeshSmoothing(
            Triangulation<1>::smoothing_on_refinement |
            Triangulation<1>::smoothing_on_coarsening))
    {}
  };

  class RadialInitialCondition : public Function<1>
  {
  public:
    explicit RadialInitialCondition(
      const InitialConditions::Parameters &parameters)
      : Function<1>(2)
      , parameters(parameters)
    {}

    double value(const Point<1> &point,
                 const unsigned int component) const override
    {
      if (component == 1)
        return parameters.amplitude_w;

      const double radius = point[0];
      double       profile = 0.0;
      if (parameters.type == "disk")
        profile = radius <= parameters.radius ? 1.0 : 0.0;
      else if (parameters.type == "smooth_disk")
        profile = 0.5 *
                  (1.0 - std::tanh(parameters.steepness_x *
                                   (radius - parameters.radius)));

      return parameters.amplitude_v * profile;
    }

  private:
    const InitialConditions::Parameters parameters;
  };
} // namespace

struct ExperimentParameters
{
  // Mesh and time
  double       domain_half_length           = 100.0;
  unsigned int n_refinements                = 5;
  unsigned int fe_degree                    = 2;
  double       final_time                   = 100.0;
  double       fast_wave_speed              = 0.0;
  double       finite_time_safety_factor    = 0.9;
  double       time_step                    = 5.0e-3;
  unsigned int output_stride                = 100;
  unsigned int mesh_adaptation_frequency    = 0;
  unsigned int max_delta_refinement_level   = 2;
  double       mesh_refinement_fraction     = 0.3;
  double       mesh_coarsening_fraction     = 0.03;
  bool         pre_refine_initial_condition = false;

  // PETSc TS / BDF
  std::string petsc_ts_type           = "bdf";
  std::string petsc_ts_adapt_type     = "none";
  int         petsc_max_steps         = -1;
  bool        petsc_match_step        = false;
  bool        petsc_restart_if_remesh = false;
  double      petsc_minimum_step_size = -1.0;
  double      petsc_maximum_step_size = -1.0;
  double      petsc_absolute_tolerance = -1.0;
  double      petsc_relative_tolerance = -1.0;
  bool        early_exit_on_ignition   = false;
  // Opt-in exits that stop a run once its ignite/quench outcome is decided.
  // Quench: max v over the domain stays below quench_max_v_threshold for a
  // full quench_window after the diagnosis start. With f_v = v(1-v)(v-beta)-w
  // and w >= 0, v < beta can only decay, so a sub-threshold field cannot
  // re-ignite; the threshold is set far below beta. Propagation: the detector
  // verdict is latched, so stopping once it is sustained is exact.
  bool        early_exit_on_quench      = false;
  double      quench_max_v_threshold    = 0.05;
  double      quench_window             = 10.0;
  bool        early_exit_on_propagation = false;

  // Diagnostics
  unsigned int diagnostic_stride                  = 20;
  double       excitation_threshold               = 0.5;
  double       diagnosis_start_time              = 0.5;
  double       ignition_max_v_threshold          = 0.5;
  double       ignition_excited_fraction_threshold = 1e-3;
  // Propagation detector: station at x = detector_x. Excitation is deemed to
  // have reached it when the excited area beyond it exceeds
  // detector_min_excited_area (a nonzero transverse extent), with
  // max v >= ignition_max_v_threshold there.
  double       detector_x                  = 0.0;
  double       detector_min_excited_area   = 0.0;
  // Hardened propagation verdict (opt-in): over the sustain window after the
  // crossing, v must stay >= the ignition max-v threshold BEYOND the station
  // and the excited area must not collapse (step tolerance = ignition area
  // decay tolerance) and must end at least as large as it started.
  bool         detector_require_area_growth = false;

  double       ignition_sustain_window           = 5.0;
  double       ignition_area_decay_tolerance     = 0.02;

  InitialConditions::Parameters initial_condition;

  double alpha_param = 1.0;
  double gamma       = 1.0;
  double diffusion_v = 1.0;
  double diffusion_w = 0.0;
  double beta_param    = 0.1;

  // Output configuration
  std::string output_directory = "outputs";
  std::string label_suffix     = "";
  bool        silent           = false;  // suppress all stdout when true

  static void declare_parameters(ParameterHandler &prm);
  void        parse_parameters(ParameterHandler &prm);
  std::string make_run_name() const;
  std::filesystem::path make_output_directory() const;

private:
  static std::string format_double_token(const double value);
  static std::string sanitize_token(const std::string &value);
};

struct DiagnosticsRecord
{
  double max_v                  = 0.0;
  double excited_area           = 0.0;
  double excited_fraction       = 0.0;
  double relative_excited_area  = 0.0;
  // Detector line: quantities restricted to the half-space x > detector_x.
  // Used for a propagation-based oracle (did excitation actually reach a
  // station downstream of the stimulus?) instead of "still alive at time T".
  double detector_max_v         = 0.0;
  double detector_excited_area  = 0.0;
  // Passive: outermost coordinate (r in the radial build, x in 2D) where
  // v exceeds the excitation threshold; 0 when nothing is excited. Used to
  // place and validate a radial propagation detector; changes no verdict.
  double outer_excited_radius   = 0.0;
};

void ExperimentParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    prm.declare_entry("Domain half length",
                      "100.0",
                      Patterns::Double(0.0),
                      "Half-length of the square domain.");
    prm.declare_entry("Global refinements",
                      "5",
                      Patterns::Integer(0),
                      "Number of global mesh refinements.");
    prm.declare_entry("Finite element degree",
                      "2",
                      Patterns::Integer(1),
                      "Polynomial degree for each component.");
    prm.declare_entry("Final time",
                      "100.0",
                      Patterns::Double(0.0),
                      "Final simulation time; overridden when Fast wave speed is positive.");
    prm.declare_entry("Fast wave speed",
                      "0.0",
                      Patterns::Double(0.0),
                      "If positive, use T=safety*(L/2-R)/C for a radial disk; "
                      "Domain half length supplies L/2.");
    prm.declare_entry("Finite time safety factor",
                      "0.9",
                      Patterns::Double(0.0, 1.0),
                      "Strictly sub-boundary factor in (0,1) for radial runs.");
    prm.declare_entry("Time step",
                      "5e-3",
                      Patterns::Double(0.0),
                      "Initial time-step size.");
    prm.declare_entry("Output stride",
                      "100",
                      Patterns::Integer(0),
                      "Write output every N accepted time steps.");
    prm.declare_entry("Mesh adaptation frequency",
                      "0",
                      Patterns::Integer(0),
                      "Adapt the mesh every N accepted time steps. Use 0 to disable.");
    prm.declare_entry("Maximum delta refinement level",
                      "2",
                      Patterns::Integer(0),
                      "Maximum adaptive refinement levels above the global base mesh.");
    prm.declare_entry("Mesh refinement fraction",
                      "0.3",
                      Patterns::Double(0.0, 1.0),
                      "Fraction of cells refined by the Kelly estimator.");
    prm.declare_entry("Mesh coarsening fraction",
                      "0.03",
                      Patterns::Double(0.0, 1.0),
                      "Fraction of cells coarsened by the Kelly estimator.");
    prm.declare_entry("Pre-refine initial condition",
                      "false",
                      Patterns::Bool(),
                      "Radial (dim=1) only: geometrically refine the mesh "
                      "around the initial disk up to the maximum adaptive "
                      "level before seeding it, so a coarse Global refinements "
                      "base can be used without smearing the initial "
                      "condition.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    prm.declare_entry("Type",
                      "bdf",
                      Patterns::Selection("bdf"),
                      "PETSc TS method to use.");
    prm.declare_entry("Adaptivity type",
                      "none",
                      Patterns::Anything(),
                      "PETSc TS adaptivity type. Use 'none' for fixed step.");
    prm.declare_entry("Maximum steps",
                      "-1",
                      Patterns::Integer(),
                      "Maximum number of TS steps. Negative uses PETSc defaults.");
    prm.declare_entry("Match step",
                      "false",
                      Patterns::Bool(),
                      "Whether PETSc should stop exactly at the final time.");
    prm.declare_entry("Restart if remesh",
                      "false",
                      Patterns::Bool(),
                      "Whether PETSc should restart the current step after remeshing.");
    prm.declare_entry("Minimum step size",
                      "-1.0",
                      Patterns::Double(),
                      "Minimum adaptive time step. Non-positive uses PETSc defaults.");
    prm.declare_entry("Maximum step size",
                      "-1.0",
                      Patterns::Double(),
                      "Maximum adaptive time step. Non-positive uses PETSc defaults.");
    prm.declare_entry("Absolute tolerance",
                      "-1.0",
                      Patterns::Double(),
                      "Absolute time-integration tolerance. Negative uses PETSc defaults.");
    prm.declare_entry("Relative tolerance",
                      "-1.0",
                      Patterns::Double(),
                      "Relative time-integration tolerance. Negative uses PETSc defaults.");
    prm.declare_entry("Early exit on ignition",
                      "false",
                      Patterns::Bool(),
                      "Stop PETSc TS after a sustained ignition is detected. "
                      "Leave false when survival to final time is required.");
    prm.declare_entry("Early exit on quench",
                      "false",
                      Patterns::Bool(),
                      "Stop once max v stays below 'Quench max v threshold' "
                      "for a full 'Quench window' after the diagnosis start.");
    prm.declare_entry("Quench max v threshold",
                      "0.05",
                      Patterns::Double(0.0),
                      "Domain maximum of v below which the field counts as "
                      "quenched; keep well below the FHN beta parameter.");
    prm.declare_entry("Quench window",
                      "10.0",
                      Patterns::Double(0.0),
                      "Time the field must stay below the quench threshold "
                      "before an early quench exit.");
    prm.declare_entry("Early exit on propagation",
                      "false",
                      Patterns::Bool(),
                      "Stop once the propagation detector verdict is "
                      "sustained (exact: the verdict is latched).");
  }
  prm.leave_subsection();

  InitialConditions::Parameters::declare_parameters(prm);

  prm.enter_subsection("Output");
  {
    prm.declare_entry("Output directory",
                      "outputs",
                      Patterns::Anything(),
                      "Parent directory for simulation outputs.");
    prm.declare_entry("Label suffix",
                      "",
                      Patterns::Anything(),
                      "Optional suffix appended to the auto-generated run name.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Model");
  {
    prm.declare_entry("Alpha parameter",
                      "1.0",
                      Patterns::Double(),
                      "Kinetic model alpha parameter.");
    prm.declare_entry("Gamma parameter",
                      "1.0",
                      Patterns::Double(),
                      "Kinetic model gamma parameter.");
    prm.declare_entry("Diffusion v",
                      "1.0",
                      Patterns::Double(),
                      "Diffusion coefficient for v.");
    prm.declare_entry("Diffusion w",
                      "0.0",
                      Patterns::Double(),
                      "Diffusion coefficient for w.");
    prm.declare_entry("Beta parameter",
                      "0.1",
                      Patterns::Double(),
                      "Excitation threshold beta: f_v = v(1-v)(v-beta)-w.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    prm.declare_entry("Diagnostic stride",
                      "20",
                      Patterns::Integer(1),
                      "Write diagnostics every N accepted time steps.");
    prm.declare_entry("Excitation threshold",
                      "0.5",
                      Patterns::Double(),
                      "Threshold on v used to define the excited set.");
    prm.declare_entry("Diagnosis start time",
                      "0.5",
                      Patterns::Double(0.0),
                      "Ignore early transients and begin ignition checks only after this time.");
    prm.declare_entry("Ignition max v threshold",
                      "0.5",
                      Patterns::Double(),
                      "Absolute lower bound on max(v) required for an ignition hit.");
    prm.declare_entry("Ignition excited fraction threshold",
                      "1e-3",
                      Patterns::Double(0.0),
                      "Minimum excited-domain fraction required for an ignition hit.");
    prm.declare_entry("Detector x",
                      "0.0",
                      Patterns::Double(),
                      "Propagation detector station: diagnostics are also "
                      "reported restricted to x > this value.");
    prm.declare_entry("Detector min excited area",
                      "0.0",
                      Patterns::Double(0.0),
                      "Minimum excited area beyond the detector required to "
                      "count as 'excitation reached the detector' (enforces a "
                      "nonzero transverse extent).");
    prm.declare_entry("Detector require area growth",
                      "false",
                      Patterns::Bool(),
                      "Hardened propagation verdict: after the crossing, require "
                      "over a full sustain window that max v beyond the station "
                      "stays >= the ignition max-v threshold, that the excited "
                      "area never drops by more than the ignition area decay "
                      "tolerance between samples, and that it ends at least as "
                      "large as at the window start. The window slides until "
                      "these hold. Default false keeps the original rule (global "
                      "max v one window after the crossing).");
    prm.declare_entry("Ignition sustain window",
                      "5.0",
                      Patterns::Double(0.0),
                      "Trailing time window over which the excited area must "
                      "be non-collapsing (not shrinking) for a diagnostic "
                      "sample to count as an ignition hit. This distinguishes "
                      "a self-propagating wave from a decaying transient pulse "
                      "that merely happens to be above the absolute thresholds "
                      "right now.");
    prm.declare_entry("Ignition area decay tolerance",
                      "0.02",
                      Patterns::Double(0.0, 1.0),
                      "Maximum fractional decrease in excited area allowed "
                      "over the sustain window before the excitation is "
                      "classified as a collapsing pulse rather than a "
                      "sustained wave (filters out numerical noise/oscillation).");
  }
  prm.leave_subsection();

}

void ExperimentParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    domain_half_length        = prm.get_double("Domain half length");
    n_refinements             = prm.get_integer("Global refinements");
    fe_degree                 = prm.get_integer("Finite element degree");
    final_time                = prm.get_double("Final time");
    fast_wave_speed           = prm.get_double("Fast wave speed");
    finite_time_safety_factor = prm.get_double("Finite time safety factor");
    time_step                 = prm.get_double("Time step");
    output_stride             = prm.get_integer("Output stride");
    mesh_adaptation_frequency = prm.get_integer("Mesh adaptation frequency");
    max_delta_refinement_level =
      prm.get_integer("Maximum delta refinement level");
    mesh_refinement_fraction = prm.get_double("Mesh refinement fraction");
    mesh_coarsening_fraction = prm.get_double("Mesh coarsening fraction");
    pre_refine_initial_condition =
      prm.get_bool("Pre-refine initial condition");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    petsc_ts_type            = prm.get("Type");
    petsc_ts_adapt_type      = prm.get("Adaptivity type");
    petsc_max_steps          = prm.get_integer("Maximum steps");
    petsc_match_step         = prm.get_bool("Match step");
    petsc_restart_if_remesh  = prm.get_bool("Restart if remesh");
    petsc_minimum_step_size  = prm.get_double("Minimum step size");
    petsc_maximum_step_size  = prm.get_double("Maximum step size");
    petsc_absolute_tolerance = prm.get_double("Absolute tolerance");
    petsc_relative_tolerance = prm.get_double("Relative tolerance");
    early_exit_on_ignition   = prm.get_bool("Early exit on ignition");
    early_exit_on_quench      = prm.get_bool("Early exit on quench");
    quench_max_v_threshold    = prm.get_double("Quench max v threshold");
    quench_window             = prm.get_double("Quench window");
    early_exit_on_propagation = prm.get_bool("Early exit on propagation");
  }
  prm.leave_subsection();

  initial_condition.parse_parameters(prm);

  prm.enter_subsection("Output");
  {
    output_directory = prm.get("Output directory");
    label_suffix     = prm.get("Label suffix");
  }
  prm.leave_subsection();

  prm.enter_subsection("Model");
  {
    alpha_param = prm.get_double("Alpha parameter");
    gamma       = prm.get_double("Gamma parameter");
    diffusion_v   = prm.get_double("Diffusion v");
    diffusion_w   = prm.get_double("Diffusion w");
    beta_param    = prm.get_double("Beta parameter");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    diagnostic_stride                   = prm.get_integer("Diagnostic stride");
    excitation_threshold                = prm.get_double("Excitation threshold");
    diagnosis_start_time                = prm.get_double("Diagnosis start time");
    ignition_max_v_threshold            = prm.get_double("Ignition max v threshold");
    detector_x                = prm.get_double("Detector x");
    detector_min_excited_area = prm.get_double("Detector min excited area");
    detector_require_area_growth = prm.get_bool("Detector require area growth");
    ignition_excited_fraction_threshold =
      prm.get_double("Ignition excited fraction threshold");
    ignition_sustain_window =
      prm.get_double("Ignition sustain window");
    ignition_area_decay_tolerance =
      prm.get_double("Ignition area decay tolerance");
  }
  prm.leave_subsection();

}

std::string ExperimentParameters::format_double_token(const double value)
{
  const double cleaned_value = (std::abs(value) < 1e-14 ? 0.0 : value);

  std::ostringstream out;
  out << std::defaultfloat << std::setprecision(4) << cleaned_value;
  std::string token = out.str();
  std::replace(token.begin(), token.end(), '-', 'm');
  std::replace(token.begin(), token.end(), '.', 'p');
  token.erase(std::remove(token.begin(), token.end(), '+'), token.end());
  return token;
}

std::string ExperimentParameters::sanitize_token(const std::string &value)
{
  std::string token;
  token.reserve(value.size());
  for (const char c : value)
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
      token.push_back(c);
    else
      token.push_back('_');
  return token;
}

std::string ExperimentParameters::make_run_name() const
{
  std::ostringstream name;
#ifdef IGNITION_RADIAL_BDF
  constexpr bool radial_problem = true;
#else
  constexpr bool radial_problem = false;
#endif

  if (silent)
    {
      // Short name for actor sweep runs: only the varying parameters
      name << "R"  << format_double_token(initial_condition.radius)
           << "_Aw" << format_double_token(initial_condition.amplitude_w)
           << "_Av" << format_double_token(initial_condition.amplitude_v)
           << "_g"  << format_double_token(gamma)
           << "_T"  << format_double_token(final_time);
      return name.str();
    }

  name << (radial_problem ? "fhn_radial" : "fhn")
       << "_L" << format_double_token(domain_half_length)
       << "_N" << n_refinements
       << "_p" << fe_degree
       << "_al" << format_double_token(alpha_param)
       << "_g" << format_double_token(gamma)
       << "_Dv" << format_double_token(diffusion_v)
       << "_Dw" << format_double_token(diffusion_w)
       << "_IC" << sanitize_token(initial_condition.type)
       << "_Av" << format_double_token(initial_condition.amplitude_v)
       << "_Aw" << format_double_token(initial_condition.amplitude_w);

  if (initial_condition.type == "disk")
    name << "_R" << format_double_token(initial_condition.radius);
  else if (initial_condition.type == "smooth_disk")
    name << "_R" << format_double_token(initial_condition.radius)
         << "_k" << format_double_token(initial_condition.steepness_x);
  else if (initial_condition.type == "stripe")
    name << "_W" << format_double_token(initial_condition.radius);
  else if (initial_condition.type == "smooth_stripe")
    name << "_W" << format_double_token(initial_condition.radius)
         << "_k" << format_double_token(initial_condition.steepness_x);
  else if (initial_condition.type == "gaussian")
    name << "_GW" << format_double_token(initial_condition.gaussian_width);
  else if (initial_condition.type == "broken_wave" ||
           initial_condition.type == "spiral")
    name << "_kx" << format_double_token(initial_condition.steepness_x)
         << "_ky" << format_double_token(initial_condition.steepness_y);

  name << "_T" << format_double_token(final_time)
       << (radial_problem ? "_BDF_radial" : "_BDF");

  const std::string suffix = sanitize_token(label_suffix);
  if (!suffix.empty())
    name << "_" << suffix;

  return name.str();
}

std::filesystem::path ExperimentParameters::make_output_directory() const
{
  return std::filesystem::path(output_directory) / make_run_name();
}

template <int dim>
class ParallelBdfFHN
{
public:
  explicit ParallelBdfFHN(const ExperimentParameters &parameters,
                          MPI_Comm                   comm = MPI_COMM_WORLD);
  bool run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices();
  void set_initial_conditions();
  void refine_around_initial_condition();

  void assemble_reaction_rhs(const VectorType &state,
                             VectorType       &reaction_rhs) const;
  void prepare_for_coarsening_and_refinement(const VectorType &state);
  void transfer_solution_vectors_to_new_mesh(
    const double                    time,
    const std::vector<VectorType>  &all_in,
    std::vector<VectorType>        &all_out);

  void implicit_function(const double      time,
                         const VectorType &y,
                         const VectorType &y_dot,
                         VectorType       &residual);
  void assemble_implicit_jacobian(const double      time,
                                  const VectorType &y,
                                  const VectorType &y_dot,
                                  const double      alpha);
  void solve_with_jacobian(const VectorType &src, VectorType &dst);
  void output_results(const double time, const unsigned int timestep_number) const;

  DiagnosticsRecord compute_diagnostics() const;
  void initialize_diagnostics_file() const;
  void append_diagnostics(const double             time,
                          const unsigned int       timestep_number,
                          const DiagnosticsRecord &diag);
  std::filesystem::path diagnostics_file_path() const;
  void update_ignition_state(const double             time,
                             const DiagnosticsRecord &diag);
  std::filesystem::path ignition_summary_file_path() const;
  void write_ignition_summary(const double             final_time,
                              const DiagnosticsRecord &final_diag) const;

  double                     initial_excited_area = -1.0;
  IgnitionDiagnostics::State ignition_state;
  // Propagation-detector tracking: the first time excitation reached the
  // detector station, and whether it was still alive one sustain-window later.
  double detector_crossing_time    = -1.0;
  bool   detector_sustained        = false;
  // Hardened verdict: (time, excited area) samples since the crossing, trimmed
  // to one sustain window plus a baseline sample, and when it was sustained.
  std::deque<std::pair<double, double>> detector_area_window;
  double detector_sustained_time   = -1.0;
  // Largest outer excited radius seen (passive, see DiagnosticsRecord).
  double max_outer_excited_radius  = 0.0;
  // Early-exit bookkeeping: start of the current sub-threshold stretch, and
  // the reason/time recorded in the summary when a run is stopped early.
  double      quench_below_since = -1.0;
  std::string early_exit_reason  = "none";
  double      early_exit_time    = -1.0;
  double       next_remesh_time                 =
    std::numeric_limits<double>::infinity();
  unsigned int last_output_step                 = numbers::invalid_unsigned_int;
  unsigned int last_diagnostic_step             = numbers::invalid_unsigned_int;

  ExperimentParameters prm;

  MPI_Comm mpi_communicator;

  BdfTriangulation<dim> triangulation;
  FESystem<dim>         fe;
  DoFHandler<dim>       dof_handler;

  IndexSet                  locally_owned_dofs;
  IndexSet                  locally_relevant_dofs;
  AffineConstraints<double> constraints;

  MatrixType mass_matrix;
  MatrixType laplace_matrix;
  MatrixType reaction_jacobian_matrix;
  MatrixType jacobian_matrix;

  VectorType solution;
  VectorType locally_relevant_solution;
  Vector<float> estimated_error_per_cell;

  mutable std::vector<std::pair<double, std::string>> pvd_times_and_names;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};

template <int dim>
ParallelBdfFHN<dim>::ParallelBdfFHN(const ExperimentParameters &parameters,
                                    MPI_Comm                   comm)
  : prm(parameters)
  , mpi_communicator(comm)
  , triangulation(mpi_communicator)
  , fe(FE_Q<dim>(parameters.fe_degree), 2)
  , dof_handler(triangulation)
  , pcout(std::cout, (!parameters.silent &&
                      Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    parameters.silent ? TimerOutput::never : TimerOutput::summary,
                    TimerOutput::wall_times)
{}

template <int dim>
std::filesystem::path ParallelBdfFHN<dim>::diagnostics_file_path() const
{
  return prm.make_output_directory() / "diagnostics.csv";
}

template <int dim>
std::filesystem::path ParallelBdfFHN<dim>::ignition_summary_file_path() const
{
  return prm.make_output_directory() / "ignition-summary.txt";
}

template <int dim>
void ParallelBdfFHN<dim>::initialize_diagnostics_file() const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::out);
  out << (dim == 1 ?
            "time,step,max_v,excited_area_2d,excited_fraction,relative_excited_area,"
            "detector_max_v,detector_excited_area,outer_excited_radius\n" :
            "time,step,max_v,excited_area,excited_fraction,relative_excited_area,"
            "detector_max_v,detector_excited_area,outer_excited_radius\n");
}

template <int dim>
DiagnosticsRecord ParallelBdfFHN<dim>::compute_diagnostics() const
{
  DiagnosticsRecord diag;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  local_solution_values(dofs_per_cell);

  VectorType solution_with_ghosts(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);
  solution_with_ghosts = solution;

  double local_max_v        = -std::numeric_limits<double>::infinity();
  double local_excited_area = 0.0;
  double local_det_max_v        = -std::numeric_limits<double>::infinity();
  double local_det_excited_area = 0.0;
  double local_outer_excited    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = solution_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double v_q = 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              if (fe.system_to_component_index(i).first == 0)
                v_q += local_solution_values[i] * fe_values.shape_value(i, q);

            local_max_v = std::max(local_max_v, v_q);

            const bool beyond_detector =
              fe_values.quadrature_point(q)[0] > prm.detector_x;
            if (beyond_detector)
              local_det_max_v = std::max(local_det_max_v, v_q);

            if (v_q > prm.excitation_threshold)
              {
                const double measure =
                  dim == 1 ? 2.0 * numbers::PI *
                               fe_values.quadrature_point(q)[0] *
                               fe_values.JxW(q) :
                             fe_values.JxW(q);
                local_excited_area += measure;
                if (beyond_detector)
                  local_det_excited_area += measure;
                local_outer_excited =
                  std::max(local_outer_excited, fe_values.quadrature_point(q)[0]);
              }
          }
      }

  diag.max_v = Utilities::MPI::max(local_max_v, mpi_communicator);
  diag.excited_area =
    Utilities::MPI::sum(local_excited_area, mpi_communicator);
  diag.detector_max_v =
    Utilities::MPI::max(local_det_max_v, mpi_communicator);
  diag.detector_excited_area =
    Utilities::MPI::sum(local_det_excited_area, mpi_communicator);
  diag.outer_excited_radius =
    Utilities::MPI::max(local_outer_excited, mpi_communicator);

  const double domain_measure =
    dim == 1 ? numbers::PI * prm.domain_half_length * prm.domain_half_length :
               std::pow(2.0 * prm.domain_half_length, static_cast<int>(dim));
  diag.excited_fraction = diag.excited_area / domain_measure;

  if (initial_excited_area > IgnitionDiagnostics::area_epsilon)
    diag.relative_excited_area = diag.excited_area / initial_excited_area;
  else
    diag.relative_excited_area =
      (diag.excited_area <= IgnitionDiagnostics::area_epsilon ?
         0.0 :
         std::numeric_limits<double>::quiet_NaN());

  return diag;
}

template <int dim>
void ParallelBdfFHN<dim>::append_diagnostics(const double             time,
                                             const unsigned int       timestep_number,
                                             const DiagnosticsRecord &diag)
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::app);
      out << std::setprecision(16)
          << time << ","
          << timestep_number << ","
          << diag.max_v << ","
          << diag.excited_area << ","
          << diag.excited_fraction << ","
          << diag.relative_excited_area << ","
          << diag.detector_max_v << ","
          << diag.detector_excited_area << ","
          << diag.outer_excited_radius << "\n";
    }
}

template <int dim>
void ParallelBdfFHN<dim>::update_ignition_state(const double             time,
                                                const DiagnosticsRecord &diag)
{
  max_outer_excited_radius =
    std::max(max_outer_excited_radius, diag.outer_excited_radius);
  ignition_state.update(
    time,
    diag.excited_area,
    diag.excited_fraction,
    diag.max_v,
    {prm.diagnosis_start_time,
     prm.ignition_sustain_window,
     prm.ignition_area_decay_tolerance,
     prm.ignition_max_v_threshold,
     prm.ignition_excited_fraction_threshold});

  // Propagation detector. "Reached" requires both a supra-threshold peak and a
  // nonzero excited extent beyond the station, so a decaying tail that merely
  // diffuses past it does not count. "Sustained" then requires the wave to be
  // still alive one sustain-window after the crossing, which rejects a wave
  // that grazes the station and immediately dies.
  if (time >= prm.diagnosis_start_time)
    {
      const bool reached =
        diag.detector_max_v >= prm.ignition_max_v_threshold &&
        diag.detector_excited_area > prm.detector_min_excited_area;
      if (reached && detector_crossing_time < 0.0)
        detector_crossing_time = time;
      if (detector_crossing_time >= 0.0 && !detector_sustained)
        {
          const double tol = 1e-12 * std::max(1.0, std::abs(time));
          if (!prm.detector_require_area_growth)
            {
              if (time >= detector_crossing_time + prm.ignition_sustain_window &&
                  diag.max_v >= prm.ignition_max_v_threshold)
                detector_sustained = true;
            }
          else
            {
              detector_area_window.emplace_back(time, diag.excited_area);
              const double cutoff = time - prm.ignition_sustain_window;
              // Keep the newest sample at or before the window start as the
              // baseline, so the window spans a full sustain window.
              while (detector_area_window.size() > 1 &&
                     detector_area_window[1].first <= cutoff + tol)
                detector_area_window.pop_front();
              const bool window_full =
                time + tol >= detector_crossing_time + prm.ignition_sustain_window &&
                time - detector_area_window.front().first + tol >=
                  prm.ignition_sustain_window;
              if (window_full &&
                  diag.detector_max_v >= prm.ignition_max_v_threshold)
                {
                  bool non_collapsing =
                    diag.excited_area >= detector_area_window.front().second;
                  for (std::size_t i = 1; i < detector_area_window.size(); ++i)
                    if (detector_area_window[i].second <
                        detector_area_window[i - 1].second *
                          (1.0 - prm.ignition_area_decay_tolerance))
                      non_collapsing = false;
                  detector_sustained = non_collapsing;
                }
            }
          if (detector_sustained)
            detector_sustained_time = time;
        }
    }
}

template <int dim>
void ParallelBdfFHN<dim>::write_ignition_summary(const double             final_time,
                                                 const DiagnosticsRecord &final_diag) const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  const bool diagnosis_window_reached =
    ignition_state.diagnosis_window_reached();
  const bool ignition_occurred = ignition_state.ignition_occurred;
  const bool sustained_to_final = ignition_state.sustained_at_latest;
  const bool wave_alive_at_final =
    final_diag.max_v >= prm.ignition_max_v_threshold &&
    final_diag.excited_fraction >= prm.ignition_excited_fraction_threshold;
  const bool ignited = wave_alive_at_final;

  const double reported_max_v_after_start =
    diagnosis_window_reached ? ignition_state.max_v_after_start
                             : std::numeric_limits<double>::quiet_NaN();
  const double reported_max_excited_fraction_after_start =
    diagnosis_window_reached ? ignition_state.max_excited_fraction_after_start
                             : std::numeric_limits<double>::quiet_NaN();

  std::ofstream out(ignition_summary_file_path().string().c_str(), std::ios::out);
  out << std::setprecision(16)
      << "diagnosis_window_reached = "
      << (diagnosis_window_reached ? "true" : "false") << "\n"
      << "ignited = " << (ignited ? "true" : "false") << "\n"
      << "ignition_occurred = " << (ignition_occurred ? "true" : "false") << "\n"
      << "sustained_to_final = " << (sustained_to_final ? "true" : "false") << "\n"
      << "wave_alive_at_final = " << (wave_alive_at_final ? "true" : "false") << "\n"
      << "sustain_window_reached_at_final = "
      << (ignition_state.window_ready_at_latest ? "true" : "false") << "\n"
      << "diagnosis_start_time = " << prm.diagnosis_start_time << "\n"
      << "ignition_max_v_threshold = " << prm.ignition_max_v_threshold << "\n"
      << "ignition_excited_fraction_threshold = "
      << prm.ignition_excited_fraction_threshold << "\n"
      << "ignition_sustain_window = " << prm.ignition_sustain_window << "\n"
      << "ignition_area_decay_tolerance = "
      << prm.ignition_area_decay_tolerance << "\n"
      << "diagnostic_samples_after_start = "
      << ignition_state.samples_after_start
      << "\n"
      << "max_v_after_start = " << reported_max_v_after_start << "\n"
      << "max_excited_fraction_after_start = "
      << reported_max_excited_fraction_after_start << "\n"
      << "final_time = " << final_time << "\n"
      << "requested_final_time = " << prm.final_time << "\n"
      << "early_exit_reason = " << early_exit_reason << "\n"
      << "early_exit_time = " << early_exit_time << "\n"
      << "final_max_v = " << final_diag.max_v << "\n"
      << "final_excited_area = " << final_diag.excited_area << "\n"
      << "final_excited_fraction = " << final_diag.excited_fraction << "\n"
      << "detector_x = " << prm.detector_x << "\n"
      << "detector_min_excited_area = " << prm.detector_min_excited_area << "\n"
      << "detector_reached = "
      << (detector_crossing_time >= 0.0 ? "true" : "false") << "\n"
      << "detector_crossing_time = " << detector_crossing_time << "\n"
      << "detector_sustained_after_crossing = "
      << (detector_sustained ? "true" : "false") << "\n"
      << "detector_require_area_growth = "
      << (prm.detector_require_area_growth ? "true" : "false") << "\n"
      << "detector_sustained_time = " << detector_sustained_time << "\n"
      << "propagated = "
      << ((detector_crossing_time >= 0.0 && detector_sustained) ? "true"
                                                                : "false")
      << "\n"
      << "final_detector_max_v = " << final_diag.detector_max_v << "\n"
      << "final_detector_excited_area = " << final_diag.detector_excited_area
      << "\n"
      << "max_outer_excited_radius = " << max_outer_excited_radius << "\n"
      << "final_outer_excited_radius = " << final_diag.outer_excited_radius
      << "\n";
  if constexpr (dim == 1)
    out << "note = excited_area is 2*pi*integral(r*dr)\n"
        << "note = domain_measure is pi*L^2\n";
}

template <int dim>
void ParallelBdfFHN<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "make grid");

  GridGenerator::hyper_cube(triangulation,
                            dim == 1 ? 0.0 : -prm.domain_half_length,
                            prm.domain_half_length);
  triangulation.refine_global(prm.n_refinements);
}


// Geometrically refine the mesh around the initial disk up to the maximum
// adaptive level *before* the initial condition is interpolated. The radial
// (dim=1) solver seeds the disk with VectorTools::interpolate on the base
// mesh, and the first solution-driven remesh only fires after one adaptation
// window, so a coarse Global refinements base would otherwise smear a small
// disk before any refinement could recover it. Refining the origin collar
// deterministically decouples the base level from the initial-condition
// resolution, letting the quiescent far field stay coarse.
template <int dim>
void ParallelBdfFHN<dim>::refine_around_initial_condition()
{
  TimerOutput::Scope t(computing_timer, "pre-refine initial condition");

  const unsigned int max_grid_level =
    prm.n_refinements + prm.max_delta_refinement_level;

  // --- wave_segment (dim >= 2): refine the bounding box of the excited band
  // AND the refractory tail before the IC is interpolated.
  bool   use_box = false;
  double bx_lo = 0.0, bx_hi = 0.0, by_lo = 0.0, by_hi = 0.0;
  if constexpr (dim >= 2)
    {
      if (prm.initial_condition.type == "wave_segment")
        {
          const auto  &ic = prm.initial_condition;
          const double h_base = 2.0 * prm.domain_half_length /
                                static_cast<double>(1u << prm.n_refinements);
          const double mx = 6.0 / std::max(ic.steepness_x, 1.0e-12) + h_base;
          const double my = 6.0 / std::max(ic.steepness_y, 1.0e-12) + h_base;
          bx_lo   = ic.center_x - ic.radius - ic.tail_width - mx;
          bx_hi   = ic.center_x + mx;
          by_lo   = ic.center_y - 0.5 * ic.segment_length - my;
          by_hi   = ic.center_y + 0.5 * ic.segment_length + my;
          use_box = true;
        }
    }

  // Radial extent that must start fully resolved: the disk radius, plus the
  // smooth edge (tanh saturates by |steepness*(r-radius)| ~ 6), plus one
  // coarse base cell so the transition cell is always bracketed.
  double collar_outer = prm.initial_condition.radius;
  if (prm.initial_condition.type == "smooth_disk" &&
      prm.initial_condition.steepness_x > 0.0)
    collar_outer += 6.0 / prm.initial_condition.steepness_x;
  collar_outer += prm.domain_half_length /
                  static_cast<double>(1u << prm.n_refinements);

  for (unsigned int round = 0; round < prm.max_delta_refinement_level; ++round)
    {
      bool flagged_any = false;
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          if (static_cast<unsigned int>(cell->level()) >= max_grid_level)
            continue;
          bool flag = false;
          if (use_box)
            {
              if constexpr (dim >= 2)
                {
                  double xlo = std::numeric_limits<double>::max();
                  double xhi = -std::numeric_limits<double>::max();
                  double ylo = xlo, yhi = xhi;
                  for (const unsigned int v : cell->vertex_indices())
                    {
                      xlo = std::min(xlo, cell->vertex(v)[0]);
                      xhi = std::max(xhi, cell->vertex(v)[0]);
                      ylo = std::min(ylo, cell->vertex(v)[1]);
                      yhi = std::max(yhi, cell->vertex(v)[1]);
                    }
                  // cell overlaps the segment bounding box
                  flag = (xhi >= bx_lo && xlo <= bx_hi && yhi >= by_lo &&
                          ylo <= by_hi);
                }
            }
          else
            {
              double r_min = std::numeric_limits<double>::max();
              for (const unsigned int v : cell->vertex_indices())
                r_min = std::min(r_min, cell->vertex(v)[0]);
              flag = (r_min <= collar_outer);
            }
          if (flag)
            {
              cell->set_refine_flag();
              flagged_any = true;
            }
        }
      if (!flagged_any)
        break;
      triangulation.prepare_coarsening_and_refinement();
      triangulation.execute_coarsening_and_refinement();
    }

  if (use_box)
    pcout << "Pre-refined wave_segment box: x in [" << bx_lo << ", " << bx_hi
          << "], y in [" << by_lo << ", " << by_hi << "] to level "
          << max_grid_level << " (" << triangulation.n_active_cells()
          << " active cells)" << std::endl;
  else
    pcout << "Pre-refined initial-condition collar: r <= " << collar_outer
          << " to level " << max_grid_level << " ("
          << triangulation.n_active_cells() << " active cells)" << std::endl;
}

template <int dim>
void ParallelBdfFHN<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup system");

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  Assert(locally_owned_dofs.is_contiguous(),
         ExcMessage("PETSc requires contiguous locally owned DoFs."));

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);

  pcout << "Number of active cells: "
        << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (" << dofs_per_component[0] << " + " << dofs_per_component[1]
        << ")" << std::endl;

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.make_consistent_in_parallel(locally_owned_dofs,
                                          locally_relevant_dofs,
                                          mpi_communicator);
  constraints.close();

  estimated_error_per_cell.reinit(triangulation.n_active_cells(), true);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
  reaction_jacobian_matrix.reinit(locally_owned_dofs,
                                  locally_owned_dofs,
                                  dsp,
                                  mpi_communicator);
  jacobian_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_matrices()
{
  TimerOutput::Scope t(computing_timer, "assemble matrices");

  mass_matrix    = 0;
  laplace_matrix = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_mass_matrix    = 0;
        cell_laplace_matrix = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double radial_weight =
              dim == 1 ? fe_values.quadrature_point(q)[0] : 1.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int component_j =
                    fe.system_to_component_index(j).first;

                  if (component_i == component_j)
                    {
                      cell_mass_matrix(i, j) +=
                        fe_values.shape_value(i, q) *
                        fe_values.shape_value(j, q) * radial_weight *
                        fe_values.JxW(q);

                      const double diffusion_coeff =
                        (component_i == 0) ? prm.diffusion_v : prm.diffusion_w;
                      cell_laplace_matrix(i, j) -=
                        diffusion_coeff * fe_values.shape_grad(i, q) *
                        fe_values.shape_grad(j, q) * radial_weight *
                        fe_values.JxW(q);
                    }
                }
            }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
        constraints.distribute_local_to_global(cell_laplace_matrix,
                                               local_dof_indices,
                                               laplace_matrix);
      }

  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}

template <int dim>
void ParallelBdfFHN<dim>::set_initial_conditions()
{
  TimerOutput::Scope t(computing_timer, "set initial conditions");

  if constexpr (dim == 1)
    {
      AssertThrow(prm.initial_condition.type == "disk" ||
                    prm.initial_condition.type == "smooth_disk",
                  ExcMessage("The radial BDF solver supports the 'disk' and "
                             "'smooth_disk' initial conditions."));
      const RadialInitialCondition initial_condition(prm.initial_condition);
      VectorTools::interpolate(dof_handler, initial_condition, solution);
    }
  else
    {
      const auto initial_condition =
        InitialConditions::make_initial_condition<dim>(prm.initial_condition);
      VectorTools::interpolate(dof_handler, *initial_condition, solution);
    }

  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  locally_relevant_solution = solution;

  pcout << "Initial condition:\n"
        << "  fe degree   = " << prm.fe_degree << "\n"
        << "  type        = " << prm.initial_condition.type << "\n"
        << "  center      = (" << prm.initial_condition.center_x << ", "
        << prm.initial_condition.center_y << ")\n"
        << "  amplitude_v = " << prm.initial_condition.amplitude_v << "\n"
        << "  amplitude_w = " << prm.initial_condition.amplitude_w << "\n";

  if (prm.initial_condition.type == "disk")
    pcout << "  radius      = " << prm.initial_condition.radius << "\n";
  else if (prm.initial_condition.type == "smooth_disk")
    pcout << "  radius      = " << prm.initial_condition.radius << "\n"
          << "  steepness   = " << prm.initial_condition.steepness_x << "\n";
  else if (prm.initial_condition.type == "stripe")
    pcout << "  width       = " << prm.initial_condition.radius << "\n"
          << "  edge_x      = "
          << prm.initial_condition.center_x + prm.initial_condition.radius
          << "\n";
  else if (prm.initial_condition.type == "smooth_stripe")
    pcout << "  width       = " << prm.initial_condition.radius << "\n"
          << "  edge_x      = "
          << prm.initial_condition.center_x + prm.initial_condition.radius
          << "\n"
          << "  steepness   = " << prm.initial_condition.steepness_x << "\n";
  else if (prm.initial_condition.type == "gaussian")
    pcout << "  gauss_width = " << prm.initial_condition.gaussian_width << "\n";
  else if (prm.initial_condition.type == "broken_wave" ||
           prm.initial_condition.type == "spiral")
    pcout << "  steepness_x = " << prm.initial_condition.steepness_x << "\n"
          << "  steepness_y = " << prm.initial_condition.steepness_y << "\n";

  pcout << "  output dir  = " << prm.make_output_directory().string()
        << std::endl;
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_reaction_rhs(const VectorType &state,
                                                VectorType       &reaction_rhs) const
{
  reaction_rhs.reinit(locally_owned_dofs, mpi_communicator);
  reaction_rhs = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>                        cell_reaction_rhs(dofs_per_cell);
  std::vector<types::global_dof_index>  local_dof_indices(dofs_per_cell);
  std::vector<double>                   local_solution_values(dofs_per_cell);

  VectorType state_with_ghosts(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
  state_with_ghosts = state;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_reaction_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = state_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double radial_weight =
              dim == 1 ? fe_values.quadrature_point(q)[0] : 1.0;
            double v_q = 0.0;
            double w_q = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                if (component_i == 0)
                  v_q += local_solution_values[i] * fe_values.shape_value(i, q);
                else
                  w_q += local_solution_values[i] * fe_values.shape_value(i, q);
              }

            const double f_v = v_q * (1.0 - v_q) * (v_q - prm.beta_param) - w_q;
            const double f_w = prm.gamma * (prm.alpha_param * v_q - w_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                const double reaction_value = (component_i == 0) ? f_v : f_w;
                cell_reaction_rhs[i] +=
                  reaction_value * fe_values.shape_value(i, q) * radial_weight *
                  fe_values.JxW(q);
              }
          }

        constraints.distribute_local_to_global(cell_reaction_rhs,
                                               local_dof_indices,
                                               reaction_rhs);
      }

  reaction_rhs.compress(VectorOperation::add);
}

template <int dim>
void ParallelBdfFHN<dim>::prepare_for_coarsening_and_refinement(const VectorType &state)
{
  VectorType locally_relevant_state(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);
  locally_relevant_state = state;

  estimated_error_per_cell.grow_or_shrink(triangulation.n_active_cells());

  ComponentMask voltage_mask(fe.n_components(), false);
  voltage_mask.set(0, true);

  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe.degree + 1),
    {},
    locally_relevant_state,
    estimated_error_per_cell,
    voltage_mask,
    nullptr,
    numbers::invalid_unsigned_int,
    numbers::invalid_subdomain_id,
    numbers::invalid_material_id,
    dim == 1 ? KellyErrorEstimator<dim>::Strategy::cell_diameter_over_24 :
               KellyErrorEstimator<dim>::Strategy::
                 face_diameter_over_twice_max_degree);

  if constexpr (dim == 1)
    GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation,
      estimated_error_per_cell,
      prm.mesh_refinement_fraction,
      prm.mesh_coarsening_fraction);
  else
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation,
      estimated_error_per_cell,
      prm.mesh_refinement_fraction,
      prm.mesh_coarsening_fraction);

  const unsigned int min_grid_level = prm.n_refinements;
  const unsigned int max_grid_level =
    prm.n_refinements + prm.max_delta_refinement_level;

  if (triangulation.n_levels() > max_grid_level)
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(max_grid_level))
      cell->clear_refine_flag();

  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(min_grid_level))
    cell->clear_coarsen_flag();
}

template <int dim>
void ParallelBdfFHN<dim>::transfer_solution_vectors_to_new_mesh(
  const double                    time,
  const std::vector<VectorType>  &all_in,
  std::vector<VectorType>        &all_out)
{
  using SolutionTransferType =
    std::conditional_t<dim == 1,
                       SolutionTransfer<dim, VectorType>,
                       parallel::distributed::SolutionTransfer<dim, VectorType>>;
  SolutionTransferType solution_transfer(dof_handler);

  std::vector<VectorType>        all_in_ghosted(all_in.size());
  std::vector<const VectorType *> all_in_ghosted_ptr(all_in.size());
  std::vector<VectorType *>       all_out_ptr(all_in.size());

  for (unsigned int i = 0; i < all_in.size(); ++i)
    {
      all_in_ghosted[i].reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
      all_in_ghosted[i] = all_in[i];
      all_in_ghosted_ptr[i] = &all_in_ghosted[i];
    }

  triangulation.prepare_coarsening_and_refinement();
  if constexpr (dim == 1)
    solution_transfer.prepare_for_coarsening_and_refinement(all_in_ghosted);
  else
    solution_transfer.prepare_for_coarsening_and_refinement(all_in_ghosted_ptr);
  triangulation.execute_coarsening_and_refinement();

  setup_system();
  assemble_matrices();
  estimated_error_per_cell.reinit(triangulation.n_active_cells(), false);

  all_out.resize(all_in.size());
  for (unsigned int i = 0; i < all_in.size(); ++i)
    {
      all_out[i].reinit(locally_owned_dofs, mpi_communicator);
      all_out_ptr[i] = &all_out[i];
    }

  if constexpr (dim == 1)
    solution_transfer.interpolate(all_in_ghosted, all_out);
  else
    solution_transfer.interpolate(all_out_ptr);

  // interpolate() writes into all locally ACTIVE dofs, not only locally owned
  // ones, so for a non-ghosted PETSc vector those writes to non-owned dofs are
  // off-process VecSetValues that sit in PETSc's stash until assembly. Assemble
  // every vector before constraints.distribute() reads it. Previously only
  // all_out[0] was ever compressed, so writes into all_out[1..] were never
  // flushed at all. No-op for dim == 1 (serial Vector).
  for (VectorType &v : all_out)
    v.compress(VectorOperation::insert);

  for (VectorType &v : all_out)
    constraints.distribute(v);

  if (!all_out.empty())
    {
      solution = all_out[0];
      solution.compress(VectorOperation::insert);
      locally_relevant_solution = solution;
    }

  (void)time;
}

template <int dim>
void ParallelBdfFHN<dim>::implicit_function(const double      /*time*/,
                                            const VectorType &y,
                                            const VectorType &y_dot,
                                            VectorType       &residual)
{
  TimerOutput::Scope t(computing_timer, "implicit function");

  VectorType tmp_solution(locally_owned_dofs, mpi_communicator);
  VectorType tmp_solution_dot(locally_owned_dofs, mpi_communicator);
  tmp_solution     = y;
  tmp_solution_dot = y_dot;

  constraints.distribute(tmp_solution);
  constraints.distribute(tmp_solution_dot);

  VectorType reaction_rhs(locally_owned_dofs, mpi_communicator);
  VectorType laplace_part(locally_owned_dofs, mpi_communicator);
  VectorType mass_part(locally_owned_dofs, mpi_communicator);

  assemble_reaction_rhs(tmp_solution, reaction_rhs);
  laplace_matrix.vmult(laplace_part, tmp_solution);
  mass_matrix.vmult(mass_part, tmp_solution_dot);

  residual = 0;
  residual += mass_part;
  residual -= laplace_part;
  residual -= reaction_rhs;
  residual.compress(VectorOperation::insert);

  for (const auto &c : constraints.get_lines())
    if (locally_owned_dofs.is_element(c.index))
      residual[c.index] = 0.0;

  residual.compress(VectorOperation::insert);
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_implicit_jacobian(const double      /*time*/,
                                                     const VectorType &y,
                                                     const VectorType & /*y_dot*/,
                                                     const double      alpha)
{
  TimerOutput::Scope t(computing_timer, "assemble implicit Jacobian");

  reaction_jacobian_matrix = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>                 cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  local_solution_values(dofs_per_cell);

  VectorType tmp_solution(locally_owned_dofs, mpi_communicator);
  tmp_solution = y;
  constraints.distribute(tmp_solution);

  VectorType solution_with_ghosts(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);
  solution_with_ghosts = tmp_solution;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = solution_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double radial_weight =
              dim == 1 ? fe_values.quadrature_point(q)[0] : 1.0;
            double v_q = 0.0;
            double w_q = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                if (component_i == 0)
                  v_q += local_solution_values[i] * fe_values.shape_value(i, q);
                else
                  w_q += local_solution_values[i] * fe_values.shape_value(i, q);
              }

            const double dfv_dv = (1.0 - 2.0 * v_q) * (v_q - prm.beta_param)
                                   + v_q * (1.0 - v_q);
            const double dfv_dw = -1.0;
            const double dfw_dv = prm.gamma * prm.alpha_param;
            const double dfw_dw = -prm.gamma;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const unsigned int component_j =
                      fe.system_to_component_index(j).first;

                    double jacobian_entry = 0.0;
                    if (component_i == 0 && component_j == 0)
                      jacobian_entry = dfv_dv;
                    else if (component_i == 0 && component_j == 1)
                      jacobian_entry = dfv_dw;
                    else if (component_i == 1 && component_j == 0)
                      jacobian_entry = dfw_dv;
                    else if (component_i == 1 && component_j == 1)
                      jacobian_entry = dfw_dw;

                    cell_matrix(i, j) +=
                      jacobian_entry * fe_values.shape_value(i, q) *
                      fe_values.shape_value(j, q) * radial_weight *
                      fe_values.JxW(q);
                  }
              }
          }

        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               reaction_jacobian_matrix);
      }

  reaction_jacobian_matrix.compress(VectorOperation::add);

  jacobian_matrix.copy_from(mass_matrix);
  jacobian_matrix *= alpha;
  jacobian_matrix.add(-1.0, laplace_matrix);
  jacobian_matrix.add(-1.0, reaction_jacobian_matrix);

  for (const auto &c : constraints.get_lines())
    if (locally_owned_dofs.is_element(c.index))
      jacobian_matrix.set(c.index, c.index, 1.0);

  jacobian_matrix.compress(VectorOperation::insert);
}

template <int dim>
void ParallelBdfFHN<dim>::solve_with_jacobian(const VectorType &src, VectorType &dst)
{
  TimerOutput::Scope t(computing_timer, "solve with Jacobian");

  SolverControl                solver_control(2000, 1e-8 * src.l2_norm());
  PETScWrappers::SolverGMRES   solver(solver_control);
  solver.set_prefix("user_");

#if defined(PETSC_HAVE_HYPRE)
  PETScWrappers::PreconditionBoomerAMG preconditioner;
  preconditioner.initialize(jacobian_matrix);
#else
  PETScWrappers::PreconditionBlockJacobi preconditioner;
  preconditioner.initialize(jacobian_matrix);
#endif

  dst = 0;
  solver.solve(jacobian_matrix, dst, src, preconditioner);
  constraints.distribute(dst);
}

template <int dim>
void ParallelBdfFHN<dim>::output_results(const double       time,
                                         const unsigned int timestep_number) const
{
  const std::filesystem::path output_dir = prm.make_output_directory();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> component_names(2);
  component_names[0] = "v";
  component_names[1] = "w";

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(2, DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(locally_relevant_solution,
                           component_names,
                           DataOut<dim>::type_dof_data,
                           component_interpretation);
  data_out.add_data_vector(estimated_error_per_cell, "error");

  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
  data_out.write_vtu_in_parallel((output_dir / filename).string(),
                                 mpi_communicator);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      pvd_times_and_names.emplace_back(time, filename);

      std::ofstream pvd_output((output_dir / "solution.pvd").string());
      DataOutBase::write_pvd_record(pvd_output, pvd_times_and_names);
    }
}

template <int dim>
bool ParallelBdfFHN<dim>::run()
{
  if constexpr (dim == 1)
    if (prm.fast_wave_speed > 0.0)
      {
        AssertThrow(prm.finite_time_safety_factor > 0.0 &&
                      prm.finite_time_safety_factor < 1.0,
                    ExcMessage("Finite time safety factor must be in (0, 1) "
                               "so T < (L/2-R)/C."));
        AssertThrow(prm.initial_condition.radius < prm.domain_half_length,
                    ExcMessage("Disk radius must be smaller than the radial domain."));
        prm.final_time =
          prm.finite_time_safety_factor *
          (prm.domain_half_length - prm.initial_condition.radius) /
          prm.fast_wave_speed;
        pcout << "Finite-time radial window: T = safety*(L/2-R)/C = "
              << prm.final_time << std::endl;
      }

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::filesystem::create_directories(prm.make_output_directory());
  MPI_Barrier(mpi_communicator);

  make_grid();

  // Pre-refinement is supported for the radial disk (dim==1) and for the 2D
  // wave_segment; other 2D initial conditions have no bespoke geometry here.
  const bool pre_refine_supported =
    (dim == 1) || (prm.initial_condition.type == "wave_segment");
  if (prm.pre_refine_initial_condition && prm.max_delta_refinement_level > 0 &&
      pre_refine_supported)
    refine_around_initial_condition();

  pcout << (dim == 1 ? "Setting up radial system..." :
                      "Setting up parallel system...")
        << std::endl;
  setup_system();

  pcout << (dim == 1 ? "Assembling radially weighted matrices..." :
                      "Assembling matrices...")
        << std::endl;
  assemble_matrices();

  pcout << "Setting initial conditions..." << std::endl;
  set_initial_conditions();

  initialize_diagnostics_file();

  DiagnosticsRecord diag0 = compute_diagnostics();
  initial_excited_area = diag0.excited_area;
  diag0.relative_excited_area =
    diag0.excited_area /
    std::max(initial_excited_area, IgnitionDiagnostics::area_epsilon);
  append_diagnostics(0.0, 0, diag0);
  update_ignition_state(0.0, diag0);

  if (prm.output_stride > 0)
    output_results(0.0, 0);
  last_output_step     = 0;
  last_diagnostic_step = 0;
  next_remesh_time =
    (prm.mesh_adaptation_frequency > 0
       ? prm.mesh_adaptation_frequency * prm.time_step
       : std::numeric_limits<double>::infinity());

  pcout << "Starting PETSc TS integration with type '" << prm.petsc_ts_type
        << "'..." << std::endl;

  PETScWrappers::TimeStepperData time_stepper_data(
    "",
    prm.petsc_ts_type,
    0.0,
    prm.final_time,
    prm.time_step,
    prm.petsc_max_steps,
    prm.petsc_match_step,
    prm.petsc_restart_if_remesh,
    prm.petsc_ts_adapt_type,
    prm.petsc_minimum_step_size,
    prm.petsc_maximum_step_size,
    prm.petsc_absolute_tolerance,
    prm.petsc_relative_tolerance,
    true);

  PETScWrappers::TimeStepper<VectorType, MatrixType> petsc_ts(time_stepper_data);
  petsc_ts.set_matrices(jacobian_matrix, jacobian_matrix);

  petsc_ts.implicit_function =
    [&](const double      time,
        const VectorType &y,
        const VectorType &y_dot,
        VectorType       &res)
  {
    this->implicit_function(time, y, y_dot, res);
  };

  petsc_ts.setup_jacobian =
    [&](const double      time,
        const VectorType &y,
        const VectorType &y_dot,
        const double      alpha)
  {
    this->assemble_implicit_jacobian(time, y, y_dot, alpha);
  };

  petsc_ts.solve_with_jacobian =
    [&](const VectorType &src, VectorType &dst)
  {
    this->solve_with_jacobian(src, dst);
  };

  petsc_ts.algebraic_components = [&]() {
    IndexSet algebraic_set(dof_handler.n_dofs());
    // deal.II consumes this set in setup_algebraic_constraints() as one
    // VecSetValues per index. After a mesh adaptation
    // extract_hanging_node_dofs() can return a MALFORMED IndexSet on some
    // ranks -- observed: size 64010 yet reporting an element of 13722368 --
    // and IndexSet::add_indices() only range-checks under Assert, so in a
    // release build the bad indices merged silently and PETSc aborted with
    // "Out of range index value" during mesh adaptation. Hanging nodes live on
    // refinement boundaries, so more MPI ranks means more partition boundaries
    // carrying them: 3 aborts in 958 solves at 31-32 ranks, 0 in 1048 at 8.
    // Intersecting with the locally owned range drops the out-of-range entries
    // and confines each rank to writing its own vector entries rather than
    // issuing off-process VecSetValues for the whole global set.
    IndexSet hanging = DoFTools::extract_hanging_node_dofs(dof_handler);
    hanging.compress();
    hanging = hanging & dof_handler.locally_owned_dofs();
    algebraic_set.add_indices(hanging);
    algebraic_set.compress();
    return algebraic_set;
  };

  petsc_ts.update_constrained_components =
    [&](const double /*time*/, VectorType &y)
  {
    constraints.distribute(y);
  };

  petsc_ts.decide_and_prepare_for_remeshing =
    [&](const double      time,
        const unsigned int step_number,
        const VectorType  &y) -> bool
  {
    if (prm.mesh_adaptation_frequency == 0)
      return false;

    const double tol = 1e-12 * std::max(1.0, std::abs(time));

    if (step_number > 0 && time + tol >= next_remesh_time &&
        time < prm.final_time - tol)
      {
        pcout << "Adapting mesh at time step " << step_number
              << " (t = " << time << ")..."
              << std::endl;
        this->prepare_for_coarsening_and_refinement(y);

        const double remesh_interval =
          prm.mesh_adaptation_frequency * prm.time_step;
        while (next_remesh_time <= time + tol)
          next_remesh_time += remesh_interval;

        return true;
      }

    return false;
  };

  petsc_ts.transfer_solution_vectors_to_new_mesh =
    [&](const double                   time,
        const std::vector<VectorType> &all_in,
        std::vector<VectorType>       &all_out)
  {
    this->transfer_solution_vectors_to_new_mesh(time, all_in, all_out);
  };

  petsc_ts.monitor =
    [&](const double      time,
        const VectorType &y,
        const unsigned int step_number)
  {
    solution = y;
    locally_relevant_solution = solution;

    if (prm.output_stride > 0 &&
        step_number % prm.output_stride == 0 &&
        step_number != last_output_step)
      {
        output_results(time, step_number);
        last_output_step = step_number;
      }

    if (step_number % prm.diagnostic_stride == 0 &&
        step_number != last_diagnostic_step)
      {
        const DiagnosticsRecord diag = compute_diagnostics();
        append_diagnostics(time, step_number, diag);
        update_ignition_state(time, diag);
        last_diagnostic_step = step_number;

        const double time_tolerance = 1e-12 * std::max(1.0, std::abs(time));
        if (time + time_tolerance >= prm.diagnosis_start_time)
          {
            if (diag.max_v < prm.quench_max_v_threshold)
              {
                if (quench_below_since < 0.0)
                  quench_below_since = time;
              }
            else
              quench_below_since = -1.0;
          }
        std::string exit_reason;
        if (prm.early_exit_on_ignition &&
            ignition_state.sustained_at_latest)
          exit_reason = "ignition";
        else if (prm.early_exit_on_propagation && detector_sustained)
          exit_reason = "propagation";
        else if (prm.early_exit_on_quench && quench_below_since >= 0.0 &&
                 !ignition_state.sustained_at_latest &&
                 time - quench_below_since + time_tolerance >= prm.quench_window)
          exit_reason = "quench";
        if (!exit_reason.empty() && early_exit_reason == "none")
          {
            early_exit_reason = exit_reason;
            early_exit_time   = time;
            // This monitor runs before TSStep, which resets the converged
            // reason, so a reason set here is discarded. Capping the step
            // count instead makes TSStep itself declare convergence
            // (TS_CONVERGED_ITS) once the step in progress completes.
            PetscErrorCode ierr =
              TSSetMaxSteps(petsc_ts.petsc_ts(), step_number);
            AssertThrow(ierr == 0, ExcPETScError(ierr));
            ierr = TSSetConvergedReason(petsc_ts.petsc_ts(), TS_CONVERGED_USER);
            AssertThrow(ierr == 0, ExcPETScError(ierr));
          }
      }
  };

  petsc_ts.solve(solution);
  locally_relevant_solution = solution;

  const double       final_time = petsc_ts.get_time();
  const unsigned int final_step = petsc_ts.get_step_number();

  if (prm.output_stride > 0 && final_step != last_output_step)
    output_results(final_time, final_step);

  const DiagnosticsRecord final_diag = compute_diagnostics();
  if (final_step != last_diagnostic_step)
    {
      append_diagnostics(final_time, final_step, final_diag);
      update_ignition_state(final_time, final_diag);
    }

  write_ignition_summary(final_time, final_diag);
  if (early_exit_reason != "none")
    pcout << "Early exit (" << early_exit_reason << ") at t = " << final_time
          << " of requested " << prm.final_time << std::endl;

  const bool diagnosis_window_reached =
    ignition_state.diagnosis_window_reached();
  const bool ignition_occurred = ignition_state.ignition_occurred;
  const bool sustained_to_final = ignition_state.sustained_at_latest;
  const bool wave_alive_at_final =
    final_diag.max_v >= prm.ignition_max_v_threshold &&
    final_diag.excited_fraction >= prm.ignition_excited_fraction_threshold;
  const bool ignited = wave_alive_at_final;

  if (diagnosis_window_reached)
    pcout << "Ignition diagnosis after t >= "
          << prm.diagnosis_start_time << ": "
          << (ignited ? "IGNITED" : "NOT IGNITED")
          << " (ignition occurred = "
          << (ignition_occurred ? "true" : "false")
          << ", sustained to final = " << (sustained_to_final ? "true" : "false")
          << ", wave alive at final = " << (wave_alive_at_final ? "true" : "false")
          << ")" << std::endl;
  else
    pcout << "Ignition diagnosis after t >= "
          << prm.diagnosis_start_time
          << " could not be performed because no diagnostic sample was "
             "recorded in that window."
          << std::endl;

  pcout << "Simulation completed successfully." << std::endl;
  return ignited;
}

#ifndef IGNITION_RADIAL_BDF
// Library interface for actor-based parameter studies


struct IgnitionSampleParams
{
  double       R                  = 2.0;    // disk radius (initial condition)
  double       Av                 = 0.2;    // amplitude_v (bisection variable)
  double       Aw                 = 0.0;    // amplitude_w (fixed parameter)
  double       gamma              = 1.0;    // kinetic model gamma
  double       domain_half_length = 30.0;
  unsigned int n_refinements      = 5;
  unsigned int fe_degree          = 2;
  double       final_time         = 100.0;
  double       time_step          = 5.0e-2;
  unsigned int output_stride      = 0;     // 0 = no VTU output
  std::string  output_directory   = "outputs";
  std::string  label_suffix       = "";
  unsigned int mesh_adaptation_frequency  = 50;
  unsigned int max_delta_refinement_level = 4;
  double       mesh_refinement_fraction   = 0.3;
  double       mesh_coarsening_fraction   = 0.15;
};

bool run_ignition_simulation(const IgnitionSampleParams &sample, MPI_Comm comm)
{
  // Zero amplitude_v → identically-zero initial condition → no excitation,
  // trivially not ignited.  Skip the expensive PETSc solve.
  if (sample.Av == 0.0)
    return false;

  ExperimentParameters prm;

  prm.initial_condition.type        = "disk";
  prm.initial_condition.radius      = sample.R;
  prm.initial_condition.amplitude_v = sample.Av;
  prm.initial_condition.amplitude_w = sample.Aw;

  prm.silent                   = true;
  prm.petsc_ts_adapt_type      = "basic";
  prm.petsc_match_step         = true;
  prm.petsc_minimum_step_size  = 1e-6;   // prevent dt collapse with large Av
  prm.petsc_maximum_step_size  = 2.0 * sample.time_step;
  prm.petsc_absolute_tolerance = 1e-6;
  prm.petsc_relative_tolerance = 1e-4;
  prm.alpha_param              = 0.37;
  prm.beta_param               = 0.131655;
  prm.excitation_threshold     = 0.3;
  prm.ignition_max_v_threshold = 0.3;
  prm.early_exit_on_ignition   = false;

  prm.gamma              = sample.gamma;
  prm.domain_half_length = sample.domain_half_length;
  prm.n_refinements      = sample.n_refinements;
  prm.fe_degree          = sample.fe_degree;
  prm.final_time         = sample.final_time;
  prm.time_step          = sample.time_step;
  prm.output_stride      = sample.output_stride;
  prm.output_directory   = sample.output_directory;
  prm.label_suffix       = sample.label_suffix;

  prm.mesh_adaptation_frequency    = sample.mesh_adaptation_frequency;
  prm.max_delta_refinement_level   = sample.max_delta_refinement_level;
  prm.mesh_refinement_fraction     = sample.mesh_refinement_fraction;
  prm.mesh_coarsening_fraction     = sample.mesh_coarsening_fraction;

  ParallelBdfFHN<2> solver(prm, comm);
  return solver.run();
}
#endif

#ifndef IGNITION_ACTOR_BUILD
int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef IGNITION_RADIAL_BDF
      AssertThrow(
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
        ExcMessage("radial_fhn_bdf uses a serial Triangulation<1> and must "
                   "be run with exactly one MPI rank."));
#endif

      const std::string parameter_file =
        (argc > 1) ? argv[1] :
#ifdef IGNITION_RADIAL_BDF
                     "./kII_disk_radial_bdf.prm";
#else
                     "./parallel_bdf_fhn.prm";
#endif

      std::ifstream parameter_stream(parameter_file);
      AssertThrow(parameter_stream.is_open(),
                  ExcMessage("Could not open parameter file: " + parameter_file));

      ParameterHandler prm;
      ExperimentParameters::declare_parameters(prm);
      prm.parse_input(parameter_file);

      ExperimentParameters params;
      params.parse_parameters(prm);

#ifdef IGNITION_RADIAL_BDF
      ParallelBdfFHN<1> fhn_solver(params);
#else
      ParallelBdfFHN<2> fhn_solver(params);
#endif
      fhn_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
#endif // IGNITION_ACTOR_BUILD
