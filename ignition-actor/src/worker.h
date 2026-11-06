#ifndef RADIAL_IGNITION_WORKER_H
#define RADIAL_IGNITION_WORKER_H

#include "radial_common.h"

#include <atomic>
#include <memory>
#include <sys/types.h>

struct worker_cancellation_state
{
  std::atomic<bool>  requested{false};
  std::atomic<pid_t> active_process_group{0};
  // Sample this worker is running (-1 between jobs). A sample-scoped cancel
  // ("cancel", id) is honoured only while it matches, so a late message can
  // never kill the next job on the worker.
  std::atomic<int>   active_sample_id{-1};
};

struct worker_state
{
  caf::actor manager;
  radial_sim_params_t params;
  std::shared_ptr<worker_cancellation_state> cancellation;
  caf::actor cancellation_controller;
  // Index of this worker WITHIN ITS NODE (0..workers_per_node-1).
  int worker_index = 0;
};

caf::behavior cancellation_controller_actor(
  caf::event_based_actor *self,
  caf::actor worker,
  std::shared_ptr<worker_cancellation_state> cancellation);

caf::behavior worker_actor(caf::stateful_actor<worker_state> *self,
                           caf::actor manager,
                           radial_sim_params_t params,
                           int worker_index);

#endif
