#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <taskr/taskr.hpp>

#include <modules/module.hpp>

namespace serving::modules::service
{

class Module final : public serving::modules::Module
{
  public:

  using serviceFunction_t = taskr::Service::serviceFc_t;

  Module(std::shared_ptr<taskr::Runtime> taskr)
    : serving::modules::Module(),
      _taskr(taskr)
  {}

  ~Module() override = default;

  // Ingest external service owned by another module (non-owning pointer).
  __INLINE__ void addService(const std::string &name, taskr::Service *service)
  {
    if (_services.contains(name)) HICR_THROW_LOGIC("[Service] Service '%s' is already registered.", name.c_str());
    _services[name] = service;
  }

  void initialize() override
  {
    // Required for services-only mode (no tasks): do not auto-finish immediately.
    _taskr->setFinishOnLastTask(false);
    for (const auto &[_, service] : _services) _taskr->addService(service);
    _taskr->initialize();
  }

  void run() override { _taskr->run(); }

  void terminate() override { _taskr->setFinishOnLastTask(true); }

  void await() override { _taskr->await(); }

  void finalize() override
  {
    _taskr->finalize();
    _services.clear();
  }

  protected:

  void service() override {}

  private:

  std::shared_ptr<taskr::Runtime> _taskr;

  std::unordered_map<std::string, taskr::Service *> _services;
};
} // namespace serving::modules::service
