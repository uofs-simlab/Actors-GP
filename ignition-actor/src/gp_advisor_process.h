#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <sys/types.h>

// Actor-owned, single-request-at-a-time IPC. No background thread accesses
// CAF state, no process-wide signal handlers, and no fork of a CAF thread.
// Call only from the owning coordinator's serialized message handlers.
class gp_advisor_process
{
public:
  gp_advisor_process(const std::string &python, const std::string &script,
                     std::chrono::milliseconds timeout = std::chrono::hours(2));
  ~gp_advisor_process();
  gp_advisor_process(const gp_advisor_process &) = delete;
  gp_advisor_process &operator=(const gp_advisor_process &) = delete;
  int run(const std::string &command);
  void discard() noexcept;
  pid_t pid() const noexcept { return pid_; }

private:
  int exchange(const std::string &command);
  void close_child() noexcept;
  int fd_ = -1;
  pid_t pid_ = -1;
  std::uint64_t sequence_ = 0;
  std::chrono::milliseconds timeout_;
};
