#ifndef RADIAL_IGNITION_CLIENT_H
#define RADIAL_IGNITION_CLIENT_H

#include "radial_common.h"

struct remote_state
{
  caf::actor manager;
  int n_workers = 0;
  // Cores this node gives its workers; 0 = legacy n_workers * solver_mpi_ranks.
  int worker_core_budget = 0;
};

caf::behavior remote_actor(caf::stateful_actor<remote_state> *self,
                           const std::string &hostname,
                           uint16_t port,
                           int n_workers,
                           int worker_core_budget);

#endif
