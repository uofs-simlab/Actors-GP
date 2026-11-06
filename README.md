# Actor-based adaptive reconstruction of ignition-threshold surfaces

Code for the paper *Actor-Based Adaptive Reconstruction of Ignition-Threshold Surfaces in Excitable
Media Using Gaussian Processes* (M. M. Moayeri, C. D. Marcotte, R. J. Spiteri).

The framework reconstructs the critical ignition amplitude A*(s, gamma) of a FitzHugh-Nagumo
reaction-diffusion system as a Gaussian-process (GP) surface. The threshold at each sample is found by a
search: a bisection over the stimulus amplitude in which every trial is a PDE solve that stops as soon as
its outcome is decided. A coordinator actor dispatches searches to worker actors under synchronous or
asynchronous scheduling. A persistent Python advisor fits the GP, chooses new locations, and decides when
to stop.

| Path | Contents |
|---|---|
| `ignition-actor/` | Launcher (`ignition_actor.sh`); `src/`: coordinator (`server.cc`), workers (`worker.cc`) and client; `gp/`: GP advisor (`gp_advisor.py`) and model (`gp_avstar_surface_botorch.py`); `tables/`: wave speeds |
| `solver/` | FitzHugh-Nagumo solver with early-out: `radial_fhn_bdf` (1-D radial) and `parallel_bdf_fhn` (2-D wave segment) |
| `jobs/` | Slurm scripts of the final runs (`*.sbatch`), their settings and the launch chain |

## Requirements

GCC 12.3, CMake, OpenMPI 4.1.5, PETSc 3.21.6, p4est 2.8.6, deal.II 9.6.0 (with MPI, PETSc, p4est, UMFPACK
and complex values), CAF 1.0.2, and Python 3.11 with torch 2.0.1, gpytorch 1.11, botorch 0.11.1 and numpy
1.25 (`ignition-actor/gp/requirements-gp.txt`). The launch chain expects the Python environment at
`./gp_botorch_env` (a virtualenv or a symlink to one).

## Build

```sh
export DEAL_II_DIR=<deal.II prefix> CAF_ROOT=<CAF prefix>
cmake -S solver -B build/solver -DDEAL_II_DIR="$DEAL_II_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build build/solver --target radial_fhn_bdf parallel_bdf_fhn -j8
cmake -S ignition-actor -B build/actor -DCAF_ROOT="$CAF_ROOT" -DCMAKE_BUILD_TYPE=Release
cmake --build build/actor -j8
```

## Run

Set `DEAL_II_DIR` and `CAF_ROOT` as above, then submit from `jobs/`. Each job builds the code and writes to
`jobs/{radial,wave}/results/`.

```sh
cd jobs
# Radial: W = 1, 5, 10, 20 workers on one node
sbatch --account=<account> radial_rep.sbatch <async|sync> <W> <informed|global> <rep>
# Wave segment: N = 1, 2, 4, 8 nodes; 16 ranks per solve gives W = 2N, 32 ranks gives W = N
sbatch --account=<account> --nodes=N --ntasks=$((32*N)) wave_rep.sbatch <async|sync> <rep> [informed|global] [16|32]
```

The settings of the final runs are in `jobs/`: problem, solver and GP settings in `*_env.sh`, method
settings in `run_arm_*.sh`, and scheduler settings in `run_scaling.sh` and `run_surface_wave.sh`.

## Site-specific settings

The Slurm scripts carry the node constraint, excluded nodes and module names of the original cluster
(32-core nodes, one core reserved for the coordinator and GP). Change them for another site: the `#SBATCH`
lines in `jobs/*.sbatch` and `ignition-actor/ignition_actor.sh`, and the `module load`
line in the latter.
