#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <hicr/core/exceptions.hpp>

#include <modules/configuration/partition.hpp>
#include <modules/roles/coordinator/module.hpp>
#include <modules/roles/job.hpp>
#include <modules/roles/jobFactory.hpp>

#include "module.hpp"

namespace serving::modules::router
{

class DefaultProcessFunction final
{
  public:

  using messageId_t = serving::system::channels::Message::messageId_t;
  using jobName_t   = serving::modules::roles::JobFactory::jobName_t;
  using edgeName_t  = Module::edgeName_t;
  using input_t     = Module::input_t;
  using message_t   = Module::message_t;

  DefaultProcessFunction(serving::modules::roles::JobFactory          &jobFactory,
                         serving::modules::roles::coordinator::Module &coordinatorModule,
                         const serving::configuration::Partition      &partition)
    : _jobFactory(jobFactory),
      _coordinatorModule(coordinatorModule)
  {
    for (const auto &task : partition.getTasks())
    {
      for (const auto &inputEdgeName : task->getInputs())
      {
        if (_inputEdgeToJobName.contains(inputEdgeName)) { HICR_THROW_LOGIC("Input edge '%s' is consumed by multiple tasks.", inputEdgeName.c_str()); }
        _inputEdgeToJobName[inputEdgeName] = task->getFunctionName();
      }
    }
  }

  __INLINE__ void process(const input_t &input, const edgeName_t &edgeName, const message_t &message)
  {
    if (_inputEdgeToJobName.contains(edgeName) == false) { HICR_THROW_LOGIC("No job registered for input edge '%s'.", edgeName.c_str()); }
    const auto &jobName = _inputEdgeToJobName.at(edgeName);
    const auto  jobId   = message.getMetadata().getId();
    auto        job     = getOrCreateJob(jobId, jobName, message);
    satisfyInputDependency(*job, edgeName, message);
    if (job->isReady())
    {
      _coordinatorModule.submitJob(job);
      std::lock_guard lock(_jobsMutex);
      _jobs.erase(jobId);
    }
  }

  private:

  __INLINE__ std::shared_ptr<serving::modules::roles::Job> getOrCreateJob(const messageId_t jobId, const jobName_t &jobName, const message_t &message)
  {
    std::lock_guard lock(_jobsMutex);
    if (_jobs.contains(jobId)) { return _jobs.at(jobId); }
    auto job     = std::make_shared<serving::modules::roles::Job>(_jobFactory.createJob(jobName, message.getMetadata()));
    _jobs[jobId] = job;
    return job;
  }

  __INLINE__ void satisfyInputDependency(serving::modules::roles::Job &job, const edgeName_t &edge, const message_t &message)
  {
    auto &dependency = job.getInputDependency(edge);
    if (dependency.isSatisfied()) { HICR_THROW_LOGIC("Input dependency %s of job is already satisfied.", edge.c_str()); }
    dependency.storeData(message.getData(), message.getSize());
    dependency.setSatisfied(true);
  }

  serving::modules::roles::JobFactory                                           &_jobFactory;
  serving::modules::roles::coordinator::Module                                  &_coordinatorModule;
  std::unordered_map<edgeName_t, jobName_t>                                      _inputEdgeToJobName;
  std::unordered_map<messageId_t, std::shared_ptr<serving::modules::roles::Job>> _jobs;
  std::mutex                                                                     _jobsMutex;
};

[[nodiscard]] static __INLINE__ Module::processFc_t makeDefaultProcessFunction(serving::modules::roles::JobFactory          &jobFactory,
                                                                               serving::modules::roles::coordinator::Module &coordinatorModule,
                                                                               const serving::configuration::Partition      &partition)
{
  auto processFunction = std::make_shared<DefaultProcessFunction>(jobFactory, coordinatorModule, partition);
  return
    [processFunction](const Module::input_t &input, const Module::edgeName_t &edgeName, const Module::message_t &message) { processFunction->process(input, edgeName, message); };
}

} // namespace serving::modules::router
