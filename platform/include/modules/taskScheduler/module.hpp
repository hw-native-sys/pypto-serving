#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hicr/core/computeManager.hpp>
#include <taskr/taskr.hpp>

#include <modules/module.hpp>

namespace serving::modules::taskScheduler
{

class Module final : public serving::modules::Module
{
  public:

  using taskFunction_t = taskr::function_t;

  Module(std::shared_ptr<HiCR::ComputeManager> computeManager, std::shared_ptr<taskr::Runtime> taskr)
    : serving::modules::Module(),
      _computeManager(computeManager),
      _taskr(taskr)
  {}

  ~Module() override = default;

  __INLINE__ void addTask(const std::string &name, const taskFunction_t &function)
  {
    if (_taskNameToIndex.contains(name)) HICR_THROW_LOGIC("Task '%s' is already registered in taskScheduler module.", name.c_str());
    _functions.push_back(std::make_unique<taskr::Function>(_computeManager.get(), function));
    _tasks.push_back(std::make_unique<taskr::Task>(_functions.back().get()));
    _taskNameToIndex[name] = _tasks.size() - 1;
  }

  void initialize() override
  {
    _taskr->setTaskCallbackHandler(HiCR::tasking::Task::callback_t::onTaskSuspend, [&](taskr::Task *task) { _taskr->resumeTask(task); });
    _taskr->setFinishOnLastTask(false);
    for (auto &task : _tasks) _taskr->addTask(task.get());
    _taskr->initialize();
  }

  void run() override { _taskr->run(); }

  void terminate() override { _taskr->setFinishOnLastTask(true); }

  void await() override { _taskr->await(); }

  void finalize() override { _taskr->finalize(); }

  protected:

  void service() override {}

  private:

  std::shared_ptr<HiCR::ComputeManager> _computeManager;
  std::shared_ptr<taskr::Runtime>       _taskr;

  std::vector<std::unique_ptr<taskr::Function>> _functions;
  std::vector<std::unique_ptr<taskr::Task>>     _tasks;
  std::unordered_map<std::string, size_t>       _taskNameToIndex;
};
} // namespace serving::modules::taskScheduler
