#include "gp_advisor_process.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

gp_advisor_process::gp_advisor_process(const std::string &python,
                                       const std::string &script,
                                       std::chrono::milliseconds timeout)
  : timeout_(timeout)
{
  int sockets[2];
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0)
    throw std::runtime_error("advisor socketpair: " + std::string(std::strerror(errno)));
  // Keep both ends away from standard descriptors, even if a launcher closed
  // stdin/stdout. This also ensures dup2 clears CLOEXEC on the child endpoints.
  for (auto &fd : sockets)
    if (fd <= STDERR_FILENO)
      {
        const int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
        if (moved < 0)
          {
            ::close(sockets[0]); ::close(sockets[1]);
            throw std::runtime_error("advisor descriptor allocation failed");
          }
        ::close(fd);
        fd = moved;
      }
  posix_spawn_file_actions_t actions;
  int rc = ::posix_spawn_file_actions_init(&actions);
  if (rc == 0)
    {
      rc = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDIN_FILENO);
      if (rc == 0) rc = ::posix_spawn_file_actions_adddup2(&actions, sockets[1], STDOUT_FILENO);
      if (rc == 0) rc = ::posix_spawn_file_actions_addclose(&actions, sockets[0]);
      if (rc == 0) rc = ::posix_spawn_file_actions_addclose(&actions, sockets[1]);
      char *argv[] = {const_cast<char *>(python.c_str()), const_cast<char *>("-u"),
                      const_cast<char *>(script.c_str()),
                      const_cast<char *>("--persistent-server"), nullptr};
      if (rc == 0) rc = ::posix_spawnp(&pid_, python.c_str(), &actions, nullptr, argv, environ);
      ::posix_spawn_file_actions_destroy(&actions);
    }
  ::close(sockets[1]);
  if (rc != 0)
    {
      ::close(sockets[0]); pid_ = -1;
      throw std::runtime_error("advisor posix_spawn: " + std::string(std::strerror(rc)));
    }
  fd_ = sockets[0];
  if (::fcntl(fd_, F_SETFL, O_NONBLOCK) != 0)
    {
      close_child();
      throw std::runtime_error("advisor nonblocking transport setup failed");
    }
}

int gp_advisor_process::exchange(const std::string &command)
{
  if (fd_ < 0 || command.size() > 1024 * 1024)
    throw std::runtime_error("advisor closed or request too large");
  const auto id = ++sequence_;
  const auto deadline = std::chrono::steady_clock::now() + timeout_;
  const auto wait_ready = [&](short events) {
    for (;;)
      {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) throw std::runtime_error("advisor request timed out");
        pollfd descriptor{fd_, events, 0};
        const int rc = ::poll(&descriptor, 1, static_cast<int>(remaining));
        if (rc < 0 && errno == EINTR) continue;
        if (rc == 0) throw std::runtime_error("advisor request timed out");
        if (rc < 0 || (descriptor.revents & POLLNVAL))
          throw std::runtime_error("advisor transport polling failed");
        return; // recv/send handles EOF, HUP and ERR without spinning.
      }
  };
  const std::string message = std::to_string(id) + " " +
                              std::to_string(command.size()) + "\n" + command;
  std::size_t offset = 0;
  while (offset < message.size())
    {
      wait_ready(POLLOUT);
      const auto n = ::send(fd_, message.data() + offset, message.size() - offset,
                            MSG_NOSIGNAL);
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (n <= 0) throw std::runtime_error("advisor request pipe closed");
      offset += static_cast<std::size_t>(n);
    }
  std::string reply;
  for (;;)
    {
      wait_ready(POLLIN);
      char c;
      const auto n = ::recv(fd_, &c, 1, 0);
      if (n < 0 && (errno == EINTR || errno == EAGAIN)) continue;
      if (n <= 0) throw std::runtime_error("advisor exited before replying");
      if (c == '\n') break;
      if (reply.size() >= 128) throw std::runtime_error("advisor malformed reply");
      reply += c;
    }
  std::istringstream parser(reply);
  std::uint64_t received_id;
  int status;
  std::string extra;
  if (!(parser >> received_id >> status) || received_id != id ||
      status < 0 || status > 255 || (parser >> extra))
    throw std::runtime_error("advisor mismatched or malformed reply");
  return status;
}

int gp_advisor_process::run(const std::string &command)
{
  try { return exchange(command); }
  catch (...) { close_child(); throw; }
}

void gp_advisor_process::discard() noexcept
{
  if (fd_ < 0) return;
  try { if (exchange("DISCARD") != 0) close_child(); }
  catch (...) { close_child(); }
}

void gp_advisor_process::close_child() noexcept
{
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
  if (pid_ <= 0) return;
  // EOF normally exits the idle advisor. Never reap or signal another child.
  for (int i = 0; i < 20; ++i)
    {
      const auto rc = ::waitpid(pid_, nullptr, WNOHANG);
      if (rc == pid_ || (rc < 0 && errno == ECHILD)) { pid_ = -1; return; }
      if (i == 10) ::kill(pid_, SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  ::kill(pid_, SIGKILL);
  while (::waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
  pid_ = -1;
}

gp_advisor_process::~gp_advisor_process() { close_child(); }
