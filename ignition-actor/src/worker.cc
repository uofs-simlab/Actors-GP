#include "worker.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

using namespace caf;

namespace
{
void request_process_cancellation(
  const std::shared_ptr<worker_cancellation_state> &cancellation)
{
  cancellation->requested.store(true, std::memory_order_release);
  const pid_t process_group =
    cancellation->active_process_group.load(std::memory_order_acquire);
  if (process_group <= 0)
    return;

  // Every solver command is placed in its own process group. This targets only
  // the active radial solve (including MPI children), never unrelated jobs on
  // the node as a global pkill would.
  ::kill(-process_group, SIGTERM);

  // Escalate only if the exact same process group is still registered after a
  // grace period. The worker clears the registration as soon as waitpid returns.
  std::thread([cancellation, process_group]() {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    if (cancellation->active_process_group.load(std::memory_order_acquire) ==
        process_group)
      ::kill(-process_group, SIGKILL);
  }).detach();
}

struct managed_command_result
{
  int  wait_status = 0;
  bool cancelled = false;
};

managed_command_result run_managed_command(
  const std::string &command,
  const std::shared_ptr<worker_cancellation_state> &cancellation)
{
  if (cancellation->requested.load(std::memory_order_acquire))
    return {0, true};

  const pid_t pid = ::fork();
  if (pid < 0)
    throw std::runtime_error("could not fork radial solver process");

  if (pid == 0)
    {
      ::setpgid(0, 0);
      ::execl("/bin/sh", "sh", "-c", command.c_str(),
              static_cast<char *>(nullptr));
      ::_exit(127);
    }

  // Also set the group in the parent to close the race before the child execs.
  // EACCES is harmless if the child has already completed its exec.
  if (::setpgid(pid, pid) != 0 && errno != EACCES && errno != ESRCH)
    {
      ::kill(pid, SIGTERM);
      ::waitpid(pid, nullptr, 0);
      throw std::runtime_error("could not create radial solver process group");
    }

  cancellation->active_process_group.store(pid, std::memory_order_release);
  if (cancellation->requested.load(std::memory_order_acquire))
    ::kill(-pid, SIGTERM);

  int wait_status = 0;
  pid_t waited = -1;
  do
    waited = ::waitpid(pid, &wait_status, 0);
  while (waited < 0 && errno == EINTR);

  pid_t expected = pid;
  cancellation->active_process_group.compare_exchange_strong(
    expected, 0, std::memory_order_acq_rel);

  if (waited < 0)
    throw std::runtime_error("waitpid failed for radial solver process");

  return {
    wait_status,
    cancellation->requested.load(std::memory_order_acquire)
  };
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

std::string token_from_double(const double value)
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

bool try_read_summary_value(const std::filesystem::path &summary,
                            const std::string           &key,
                            std::string                 &value)
{
  std::ifstream in(summary);
  std::string line;
  const std::string prefix = key + " = ";
  while (std::getline(in, line))
    if (line.rfind(prefix, 0) == 0)
      {
        value = line.substr(prefix.size());
        return true;
      }
  return false;
}

std::filesystem::path find_latest_summary(const std::filesystem::path &root)
{
  std::filesystem::path latest;
  std::filesystem::file_time_type latest_time{};

  if (!std::filesystem::exists(root))
    return latest;

  for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
    {
      const std::string filename = entry.path().filename().string();
      if (!entry.is_regular_file() ||
          filename.find("ignition-summary.txt") == std::string::npos)
        continue;

      const auto entry_time = entry.last_write_time();
      if (latest.empty() || entry_time > latest_time)
        {
          latest = entry.path();
          latest_time = entry_time;
        }
    }

  return latest;
}

bool parse_radial_summary(const std::filesystem::path &summary,
                          const bool use_detector_oracle = false)
{
  // Propagation oracle: success is "excitation reached the detector station and
  // stayed alive", which is independent of the final time. Preferred for the 2D
  // wave-segment problem, where "still alive at T" is T-dependent because the
  // wave eventually leaves the domain.
  if (use_detector_oracle)
    {
      std::string propagated;
      if (try_read_summary_value(summary, "propagated", propagated))
        return propagated == "true";
      throw std::runtime_error(
        "detector oracle requested but the solver summary has no "
        "'propagated' field: " + summary.string());
    }

  std::string ignited;
  if (try_read_summary_value(summary, "ignited", ignited))
    return ignited == "true";

  std::string sustained;
  if (try_read_summary_value(summary, "sustained_to_final", sustained))
    return sustained == "true";

  std::string fraction;
  std::string final_max_v;
  std::string peak_max_v;
  const bool has_fraction =
    try_read_summary_value(summary, "final_excited_fraction", fraction);
  const bool has_final_max_v =
    try_read_summary_value(summary, "final_max_v", final_max_v);
  const bool has_peak_max_v =
    try_read_summary_value(summary, "max_v_after_start", peak_max_v);

  if (has_fraction && has_final_max_v && has_peak_max_v)
    {
      const double excited_fraction = std::stod(fraction);
      const double amplitude_drop =
        std::stod(peak_max_v) - std::stod(final_max_v);
      return excited_fraction >= 0.01 && amplitude_drop < 0.6;
    }

  if (has_fraction && has_final_max_v)
    return std::stod(fraction) >= 0.01 && std::stod(final_max_v) >= 0.6;

  std::string wave_alive;
  if (try_read_summary_value(summary, "wave_alive_at_final", wave_alive))
    return wave_alive == "true";

  std::string ignition_occurred;
  if (try_read_summary_value(summary, "ignition_occurred", ignition_occurred))
    return ignition_occurred == "true";

  std::string diagnostic_samples;
  std::string maximum_hits;
  std::string required_hits;
  if (try_read_summary_value(summary,
                             "diagnostic_samples_after_start",
                             diagnostic_samples) &&
      try_read_summary_value(summary,
                             "max_consecutive_ignition_hits",
                             maximum_hits) &&
      try_read_summary_value(summary,
                             "required_ignition_hits",
                             required_hits))
    return std::stoi(diagnostic_samples) > 0 &&
           std::stoi(maximum_hits) >= std::stoi(required_hits);

  throw std::runtime_error("missing ignition result in " + summary.string());
}

void write_radial_prm(const std::filesystem::path &parameter_file,
                      const std::filesystem::path &solver_output_dir,
                      const radial_job_t          &job,
                      const radial_sim_params_t   &sp,
                      const double                 Av,
                      const unsigned int           run_id)
{
  std::ofstream prm(parameter_file);
  if (!prm)
    throw std::runtime_error("could not create " + parameter_file.string());

  prm << "subsection Mesh and Time\n"
      << "  set Domain half length           = "
      << double_value(sp.domain_half_length) << "\n"
      << "  set Global refinements           = "
      << sp.n_refinements << "\n"
      << "  set Finite element degree        = "
      << sp.fe_degree << "\n"
      << "  set Final time                   = "
      << double_value(sp.final_time) << "\n"
      << "  set Fast wave speed              = "
      << double_value(job.fast_wave_speed > 0.0
                        ? job.fast_wave_speed
                        : sp.fast_wave_speed) << "\n"
      << "  set Finite time safety factor    = "
      << double_value(sp.finite_time_safety_factor) << "\n"
      << "  set Time step                    = "
      << double_value(sp.time_step) << "\n"
      << "  set Output stride                = "
      << sp.output_stride << "\n"
      << "  set Mesh adaptation frequency    = "
      << sp.mesh_adaptation_frequency << "\n"
      << "  set Maximum delta refinement level = "
      << sp.max_delta_refinement_level << "\n"
      << "  set Mesh refinement fraction     = "
      << double_value(sp.mesh_refinement_fraction) << "\n"
      << "  set Mesh coarsening fraction     = "
      << double_value(sp.mesh_coarsening_fraction) << "\n";

  // Only the BDF radial solver declares this entry; emitting it for arkode
  // would make its ParameterHandler reject the file.
  if (sp.solver_backend == "bdf")
    prm << "  set Pre-refine initial condition = "
        << (sp.pre_refine_initial_condition ? "true" : "false") << "\n";

  prm << "end\n\n"
      << "subsection Time integrator\n";

  if (sp.solver_backend == "arkode")
    prm << "  set ARKode minimum step size                     = "
        << double_value(sp.arkode_minimum_step_size) << "\n"
        << "  set ARKode maximum order                         = "
        << sp.arkode_maximum_order << "\n"
        << "  set ARKode maximum nonlinear iterations          = "
        << sp.arkode_maximum_non_linear_iterations << "\n"
        << "  set ARKode absolute tolerance                    = "
        << double_value(sp.arkode_absolute_tolerance) << "\n"
        << "  set ARKode relative tolerance                    = "
        << double_value(sp.arkode_relative_tolerance) << "\n"
        << "  set ARKode implicit function is linear           = false\n"
        << "  set ARKode implicit function is time independent = false\n";
  else if (sp.solver_backend == "bdf")
    prm << "  set Type               = bdf\n"
        << "  set Adaptivity type    = basic\n"
        << "  set Maximum steps      = -1\n"
        << "  set Match step         = true\n"
        << "  set Restart if remesh  = true\n"
        << "  set Minimum step size  = "
        << double_value(sp.arkode_minimum_step_size) << "\n"
        << "  set Maximum step size  = "
        << double_value(sp.time_step) << "\n"
        << "  set Absolute tolerance = "
        << double_value(sp.arkode_absolute_tolerance) << "\n"
        << "  set Relative tolerance = "
        << double_value(sp.arkode_relative_tolerance) << "\n"
        << "  set Early exit on ignition = false\n";
  else
    throw std::runtime_error("unsupported solver backend '" +
                             sp.solver_backend + "'");
  if (sp.solver_backend == "bdf" &&
      (sp.early_exit_on_quench || sp.early_exit_on_propagation))
    prm << "  set Early exit on quench = "
        << (sp.early_exit_on_quench ? "true" : "false") << "\n"
        << "  set Quench max v threshold = "
        << double_value(sp.quench_max_v_threshold) << "\n"
        << "  set Quench window = " << double_value(sp.quench_window) << "\n"
        << "  set Early exit on propagation = "
        << (sp.early_exit_on_propagation ? "true" : "false") << "\n";

  prm << "end\n\n"
      << "subsection Initial condition\n"
      << "  set Type         = " << job.ic_type << "\n"
      << "  set Center x     = 0.0\n"
      << "  set Center y     = 0.0\n"
      << "  set Amplitude v  = " << double_value(Av) << "\n"
      << "  set Amplitude w  = " << double_value(job.Aw) << "\n";

  if (job.ic_type == "wave_segment")
    {
      // The swept variable is the SEGMENT LENGTH for this initial condition;
      // the solver's `Radius` is instead the excited band's x-extent.
      prm << "  set Radius         = "
          << double_value(sp.segment_wave_width) << "\n"
          << "  set Tail width     = "
          << double_value(sp.segment_tail_width) << "\n"
          << "  set Segment length = " << double_value(job.R) << "\n"
          << "  set Steepness x    = " << double_value(job.steepness) << "\n"
          << "  set Steepness y    = " << double_value(job.steepness) << "\n";
    }
  else
    {
      prm << "  set Radius       = " << double_value(job.R) << "\n"
          << "  set Steepness x  = " << double_value(job.steepness) << "\n"
          << "  set Steepness y  = " << double_value(job.steepness) << "\n";
    }

  prm << "end\n\n"
      << "subsection Output\n"
      << "  set Output directory = " << solver_output_dir.string() << "\n"
      << "  set Label suffix     = actor_sample" << job.sample_id
      << "_run" << run_id << "\n"
      << "end\n\n"
      << "subsection Model\n"
      << "  set Alpha parameter = " << double_value(sp.alpha_param) << "\n"
      << "  set Gamma parameter = " << double_value(job.gamma) << "\n"
      << "  set Diffusion v     = " << double_value(sp.diffusion_v) << "\n"
      << "  set Diffusion w     = " << double_value(sp.diffusion_w) << "\n"
      << "  set Beta parameter  = " << double_value(sp.beta_param) << "\n"
      << "end\n\n"
      << "subsection Diagnostics\n"
      << "  set Diagnostic stride                   = "
      << sp.diagnostic_stride << "\n"
      << "  set Excitation threshold                = "
      << double_value(sp.excitation_threshold) << "\n"
      << "  set Diagnosis start time                = "
      << double_value(sp.diagnosis_start_time) << "\n"
      << "  set Ignition max v threshold            = "
      << double_value(sp.ignition_max_v_threshold) << "\n"
      << "  set Ignition excited fraction threshold = "
      << double_value(sp.ignition_excited_fraction_threshold) << "\n"
      << "  set Ignition sustain window             = "
      << double_value(sp.ignition_sustain_window) << "\n"
      << "  set Ignition area decay tolerance       = "
      << double_value(sp.ignition_area_decay_tolerance) << "\n";

  // Radial detector: a fixed distance beyond this sample's own disk radius.
  if (sp.use_detector_oracle)
    prm << "  set Detector x                          = "
        << double_value(sp.detector_offset_from_radius > 0.0
                          ? job.R + sp.detector_offset_from_radius
                          : sp.detector_x) << "\n"
        << "  set Detector min excited area           = "
        << double_value(sp.detector_min_excited_area) << "\n";
  if (sp.use_detector_oracle && sp.detector_require_area_growth)
    prm << "  set Detector require area growth        = true\n";

  prm << "end\n";
}

struct radial_run_result
{
  bool ignited = false;
  bool cancelled = false;
  double runtime_s = 0.0;
  std::filesystem::path run_root;
};

radial_run_result run_radial_solver(const radial_job_t        &job,
                                    const radial_sim_params_t &sp,
                                    const double               Av,
                                    const unsigned int         run_id,
                                    const std::shared_ptr<worker_cancellation_state>
                                      &cancellation)
{
  if (sp.solver_binary.empty())
    throw std::runtime_error("--solver-binary is required");

  const std::filesystem::path sample_root =
    std::filesystem::path(sp.output_dir) / "external_runs" /
    ("sample" + std::to_string(job.sample_id) +
     "_R" + token_from_double(job.R) +
     "_Aw" + token_from_double(job.Aw) +
     "_g" + token_from_double(job.gamma));
  const std::filesystem::path run_root =
    sample_root /
    ("run" + std::to_string(run_id) + "_Av" + token_from_double(Av));
  const std::filesystem::path solver_output_dir = run_root / "solver_output";
  const std::filesystem::path parameter_file = run_root / "input.prm";
  const std::filesystem::path solver_log = run_root / "solver.log";

  std::filesystem::create_directories(solver_output_dir);
  write_radial_prm(parameter_file, solver_output_dir, job, sp, Av, run_id);

  std::string command;
  const bool direct_execution =
    sp.solver_runner.empty() || sp.solver_runner == "direct" ||
    sp.solver_runner == "none";

  if (direct_execution)
    {
      command = shell_quote(sp.solver_binary);
    }
  else
    {
      // mpirun reads the Slurm environment to size its slot pool, and a
      // client-node worker runs inside an `srun --ntasks=1` step.
      const std::string mpi_env_scrub =
        "env -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE -u SLURM_JOB_ID "
        "-u SLURM_JOBID -u SLURM_NODELIST -u SLURM_JOB_NODELIST "
        "-u SLURM_NPROCS ";
      command = sp.solver_runner;
      // Rank-count syntax differs by launcher.
      const int first_core = static_cast<int>(sp.solver_first_core);
      const int last_core =
        first_core + static_cast<int>(sp.solver_mpi_ranks) - 1;
      const auto runner_is_mpirun =
        sp.solver_runner.find("mpirun") != std::string::npos ||
        sp.solver_runner.find("mpiexec") != std::string::npos;
      if (runner_is_mpirun)
        {
          // mpirun creates no Slurm step, so nothing allocates cores for it:
          // the ABSOLUTE slice must be named explicitly or concurrent solves
          // land on the same cores.
          //
          // `--mca plm isolated` is essential, inside a Slurm
          // allocation OpenMPI otherwise selects its slurm launch module and
          // goes back through srun -- re-creating the nested-step problem this
          // path exists to avoid. The env scrubber clears SLURM_STEP_* but
          // leaves SLURM_JOB_ID/SLURM_NODELIST, which is all OpenMPI needs to
          // detect the resource manager. `isolated` forks the ranks locally.
          command = mpi_env_scrub + command;
          command += " -n " + std::to_string(sp.solver_mpi_ranks);
          command += " --mca plm isolated";
          command += " --bind-to core";
          if (sp.pin_solver_cores)
            command += " --cpu-set " + std::to_string(first_core) + "-" +
                       std::to_string(last_core);
        }
      else
        {
          command += " --ntasks=" + std::to_string(sp.solver_mpi_ranks);
          command += " --cpus-per-task=1";
          command += " --distribution=block:block";
          command += " --cpu-bind=cores";
        }
      command += " " + shell_quote(sp.solver_binary);
    }
  command += " " + shell_quote(parameter_file.string());
  {
    std::string header = "# " + command + "\n";
    command = "{ printf '%s' " + shell_quote(header) + "; " + command + "; }";
  }
  command += " > " + shell_quote(solver_log.string()) + " 2>&1";

  const auto start = std::chrono::high_resolution_clock::now();
  const auto command_result = run_managed_command(command, cancellation);
  const double runtime_s = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - start).count();

  radial_run_result result;
  result.runtime_s = runtime_s;
  result.run_root = run_root;
  if (command_result.cancelled)
    {
      result.cancelled = true;
      return result;
    }

  if (!WIFEXITED(command_result.wait_status) ||
      WEXITSTATUS(command_result.wait_status) != 0)
    throw std::runtime_error("radial solver failed with status " +
                             std::to_string(command_result.wait_status) + "; see " +
                             solver_log.string());

  const std::filesystem::path summary = find_latest_summary(solver_output_dir);
  if (summary.empty())
    throw std::runtime_error("radial solver did not write ignition summary; see " +
                             solver_log.string());

  result.ignited = parse_radial_summary(summary, sp.use_detector_oracle);
  return result;
}

struct bisection_result
{
  bool threshold_found = false;
  bool converged = false;
  bool cancelled = false;
  double critical_Av = 0.0;
  double Av_lo = 0.0;
  double Av_hi = 0.0;
  unsigned int iterations = 0;
  unsigned int solver_runs = 0;
  // Whether the current Av_lo / Av_hi were set by an observed solve (quench /
  // ignition). The server salvages a cancelled bracket only when both were.
  bool lower_end_observed = false;
  bool upper_end_observed = false;
  std::filesystem::path sample_root;
};

bisection_result run_bisection(const radial_job_t        &job,
                               const radial_sim_params_t &sp,
                               const std::shared_ptr<worker_cancellation_state>
                                 &cancellation)
{
  bisection_result result;
  // The bracket travels with the job so the server can narrow it per sample
  // from the GP posterior. With seeding disabled these are the global bounds
  // and everything below reduces to the original behaviour.
  result.Av_lo = job.Av_lo;
  result.Av_hi = job.Av_hi;
  // The global lower end (Av_lo = 0, nothing to verify) is trivially
  // non-igniting; a GP-seeded bracket starts with neither end observed.
  result.lower_end_observed = !job.verify_lower_endpoint;
  result.upper_end_observed = false;
  result.sample_root =
    std::filesystem::path(sp.output_dir) / "external_runs" /
    ("sample" + std::to_string(job.sample_id) +
     "_R" + token_from_double(job.R) +
     "_Aw" + token_from_double(job.Aw) +
     "_g" + token_from_double(job.gamma));

  std::filesystem::create_directories(result.sample_root);
  std::ofstream trace(result.sample_root / "bisection.csv", std::ios::out);
  if (!trace)
    throw std::runtime_error("could not create bisection trace in " +
                             result.sample_root.string());
  trace << "run,Av_lo_before,Av_tested,Av_hi_before,ignited,runtime_s,"
           "run_dir,cancelled\n";
  trace << std::setprecision(17);

  auto run_and_record = [&](const double Av) {
    if (cancellation->requested.load(std::memory_order_acquire))
      {
        result.cancelled = true;
        return std::optional<bool>{};
      }

    const double lo_before = result.Av_lo;
    const double hi_before = result.Av_hi;
    const auto run =
      run_radial_solver(job, sp, Av, result.solver_runs, cancellation);
    trace << result.solver_runs << ","
          << lo_before << ","
          << Av << ","
          << hi_before << ",";
    if (!run.cancelled)
      trace << (run.ignited ? "true" : "false");
    trace << ","
          << run.runtime_s << ","
          << csv_escape(run.run_root.string()) << ","
          << (run.cancelled ? "true" : "false") << "\n";
    trace.flush();
    if (run.cancelled)
      {
        result.cancelled = true;
        return std::optional<bool>{};
      }
    ++result.solver_runs;
    return std::optional<bool>{run.ignited};
  };

  const auto climb = [&](double from, const double top) -> std::optional<bool> {
    double probe = std::min(from, top);
    while (true)
      {
        const auto ignited = run_and_record(probe);
        if (!ignited.has_value())
          return std::nullopt;
        if (ignited.value())
          {
            result.Av_hi = probe;
            result.upper_end_observed = true;
            return true;
          }
        result.Av_lo = probe;
        result.lower_end_observed = true;
        if (probe >= top)
          return false;
        probe = std::min(probe * sp.bisection_probe_ratio, top);
      }
  };
  const bool ladder_enabled =
    sp.bisection_probe_ladder && sp.bisection_probe_ratio > 1.0;

  if (ladder_enabled && !job.verify_lower_endpoint &&
      sp.bisection_probe_start > result.Av_lo &&
      sp.bisection_probe_start < result.Av_hi)
    {
      const auto climbed = climb(sp.bisection_probe_start, result.Av_hi);
      if (!climbed.has_value())
        return result;
      // Nothing ignited all the way up to Av_hi: no threshold in range, which
      // is the same verdict the single-probe path returns.
      if (!climbed.value())
        return result;
    }
  else
    {
      // Single-probe path. With the global bracket, Av_lo = 0 is trivially
      // non-igniting and only the upper end needs testing -- the original
      // behaviour. A GP-seeded bracket carries no such guarantee at EITHER end,
      // so both are tested.
    auto upper_ignited = run_and_record(result.Av_hi);
    if (!upper_ignited.has_value())
      return result;
    result.upper_end_observed = upper_ignited.value();
    if (!upper_ignited.value())
      {
        // Predicted too low: Av* lies above the seeded top. That top is now a
        // verified non-igniting point, so it becomes the new lower end and the
        // search widens up to the global bound.
        if (result.Av_hi >= sp.Av_hi)
          return result; // genuinely no threshold below the global Av_hi
        result.Av_lo = result.Av_hi;
        result.lower_end_observed = true;
        if (ladder_enabled)
          {
            const auto climbed =
              climb(result.Av_lo * sp.bisection_probe_ratio, sp.Av_hi);
            if (!climbed.has_value() || !climbed.value())
              return result;
          }
        else
          {
            result.Av_hi = sp.Av_hi;
            upper_ignited = run_and_record(result.Av_hi);
            if (!upper_ignited.has_value() || !upper_ignited.value())
              return result;
            result.upper_end_observed = true;
          }
      }
    else if (job.verify_lower_endpoint && result.Av_lo > sp.Av_lo)
      {
        const auto lower_ignited = run_and_record(result.Av_lo);
        if (!lower_ignited.has_value())
          return result;
        if (lower_ignited.value())
          {
            // Predicted too high: Av* lies below the seeded bottom. That bottom
            // is now a verified igniting point, so it becomes the new upper end
            // and the search falls back to the global lower bound.
            result.Av_hi = result.Av_lo;
            result.upper_end_observed = true;
            result.Av_lo = sp.Av_lo;
            result.lower_end_observed = true; // global lower end, trivially non-igniting
          }
        else
          result.lower_end_observed = true;
      }
    }

  const auto effective_tol = [&](const double lo, const double hi) {
    const double mid = 0.5 * (lo + hi);
    return std::max(sp.bisection_tolerance,
                    sp.bisection_relative_tolerance * mid);
  };

  result.threshold_found = true;
  while (result.Av_hi - result.Av_lo >=
           effective_tol(result.Av_lo, result.Av_hi) &&
         result.iterations < sp.bisection_max_iterations)
    {
      const double midpoint = 0.5 * (result.Av_lo + result.Av_hi);
      const auto ignited = run_and_record(midpoint);
      if (!ignited.has_value())
        return result;
      if (ignited.value())
        {
          result.Av_hi = midpoint;
          result.upper_end_observed = true;
        }
      else
        {
          result.Av_lo = midpoint;
          result.lower_end_observed = true;
        }
      ++result.iterations;
    }

  result.critical_Av = 0.5 * (result.Av_lo + result.Av_hi);
  result.converged =
    result.Av_hi - result.Av_lo <
      effective_tol(result.Av_lo, result.Av_hi);
  return result;
}
} // namespace

caf::behavior cancellation_controller_actor(
  caf::event_based_actor *self,
  caf::actor worker,
  std::shared_ptr<worker_cancellation_state> cancellation)
{
  self->monitor(worker, [self](const caf::error &) { self->quit(); });

  return {
    [=](const std::string &message) {
      if (message == "cancel")
        request_process_cancellation(cancellation);
      else if (message == "quit")
        self->quit();
    },
    // Sample-scoped cancel (certification at readiness): ignored unless the
    // worker is still running exactly this sample.
    [=](const std::string &message, int sample_id) {
      if (message == "cancel" &&
          cancellation->active_sample_id.load(std::memory_order_acquire) ==
            sample_id)
        request_process_cancellation(cancellation);
    }
  };
}

caf::behavior worker_actor(stateful_actor<worker_state> *self,
                           actor manager,
                           radial_sim_params_t params,
                           int worker_index)
{
  self->state().manager = manager;
  self->state().params = params;
  self->state().worker_index = worker_index;
  self->state().cancellation = std::make_shared<worker_cancellation_state>();
  const auto worker = caf::actor_cast<caf::actor>(self);
  self->state().cancellation_controller = self->spawn(
    cancellation_controller_actor, worker, self->state().cancellation);
  anon_mail("cancellation_controller",
            worker,
            self->state().cancellation_controller)
    .send(self->state().manager);

  return {
    [=](radial_job_t job) {
      self->state().cancellation->requested.store(false,
                                                   std::memory_order_release);
      self->state().cancellation->active_process_group.store(
        0, std::memory_order_release);
      self->state().cancellation->active_sample_id.store(
        job.sample_id, std::memory_order_release);
      self->println(
        "[sample={}] bisecting Av=[{:.6f}, {:.6f}]{}: R={:.4f} Aw={:.4f} gamma={:.5f}",
        job.sample_id,
        job.Av_lo,
        job.Av_hi,
        job.verify_lower_endpoint ? " (GP-seeded, both ends verified)" : "",
        job.R,
        job.Aw,
        job.gamma);

      const auto t0 = std::chrono::high_resolution_clock::now();
      bool threshold_found = false;
      bool converged = false;
      bool cancelled = false;
      double critical_Av = 0.0;
      double Av_lo = job.Av_lo;
      double Av_hi = job.Av_hi;
      unsigned int iterations = 0;
      unsigned int solver_runs = 0;
      bool bracket_ends_observed = false;
      std::string run_dir;
      std::string error;

      try
        {
          const auto result = run_bisection(job,
                                            self->state().params,
                                            self->state().cancellation);
          threshold_found = result.threshold_found;
          converged = result.converged;
          cancelled = result.cancelled;
          critical_Av = result.critical_Av;
          Av_lo = result.Av_lo;
          Av_hi = result.Av_hi;
          iterations = result.iterations;
          solver_runs = result.solver_runs;
          bracket_ends_observed =
            result.lower_end_observed && result.upper_end_observed;
          run_dir = result.sample_root.string();
        }
      catch (const std::exception &e)
        {
          error = e.what();
          self->println("WORKER ERROR sample={}: {}", job.sample_id, error);
        }

      const auto t1 = std::chrono::high_resolution_clock::now();
      const double runtime_s = std::chrono::duration<double>(t1 - t0).count();
      self->state().cancellation->active_sample_id.store(
        -1, std::memory_order_release);

      anon_mail("result",
                job.sample_id,
                cancelled,
                threshold_found,
                converged,
                critical_Av,
                Av_lo,
                Av_hi,
                iterations,
                solver_runs,
                bracket_ends_observed,
                runtime_s,
                run_dir,
                error,
                self)
        .send(self->state().manager);
    },

    [=](const std::string &msg) {
      if (msg == "quit")
        {
          anon_mail("quit", self).send(self->state().manager);
          self->quit();
        }
    }
  };
}
