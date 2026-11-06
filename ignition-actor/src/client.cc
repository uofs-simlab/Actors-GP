#include "client.h"
#include "worker.h"

#include "caf/io/all.hpp"

using namespace caf;

caf::behavior remote_actor(stateful_actor<remote_state> *self,
                           const std::string &hostname,
                           uint16_t port,
                           int n_workers,
                           int worker_core_budget)
{
  auto srv = self->system().middleman().remote_actor(hostname, port);
  if (!srv)
    {
      self->println("Failed to connect to server at {}:{}", hostname, port);
      self->quit();
      return {};
    }

  self->println("Connected to server at {}:{}", hostname, port);
  self->state().manager = *srv;
  self->state().n_workers = n_workers;
  self->state().worker_core_budget = worker_core_budget;
  anon_mail(self, 0).send(self->state().manager);

  return {
    [=](radial_sim_params_t sp) {
      for (int i = 0; i < self->state().n_workers; ++i)
        {
          auto wsp = sp;
          const auto slice = worker_core_slice(i,
                                               self->state().n_workers,
                                               self->state().worker_core_budget,
                                               sp.solver_mpi_ranks);
          wsp.solver_first_core = slice.first;
          wsp.solver_mpi_ranks  = slice.second;
          auto w = self->spawn(worker_actor, self->state().manager, wsp, i);
          anon_mail(w).send(self->state().manager);
        }
      self->quit();
    }
  };
}
