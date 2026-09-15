#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>
#include <hicr/core/localMemorySlot.hpp>

#include <modules/subscription.hpp>
#include <modules/module.hpp>
#include <system/channels/input.hpp>
#include <system/channels/message.hpp>
#include <system/channels/output.hpp>

#include "../job.hpp"
#include "../jobFactory.hpp"

namespace serving::modules::roles::coordinator
{

class Module final : public modules::Module
{
  public:

  using input_t       = std::shared_ptr<serving::system::channels::Input>;
  using output_t      = std::shared_ptr<serving::system::channels::Output>;
  using message_t     = serving::system::channels::Message;
  using metadata_t    = serving::system::channels::Message::metadata_t;
  using messageId_t   = serving::system::channels::Message::messageId_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using instanceId_t  = HiCR::Instance::instanceId_t;
  using jobName_t     = serving::modules::roles::JobFactory::jobName_t;
  using edgeName_t    = std::string;

  struct JobOutput
  {
    edgeName_t                             name;
    std::shared_ptr<HiCR::LocalMemorySlot> data;
  };

  using completionCallback_t = std::function<void(const metadata_t &metadata, const std::vector<JobOutput> &outputs)>;

  Module(const size_t intervalMs)
    : modules::Module(intervalMs)
  {}

  using inputChannelMap_t  = std::unordered_map<edgeName_t, output_t>;
  using outputChannelMap_t = std::unordered_map<edgeName_t, input_t>;

  __INLINE__ void addReplica(const instanceId_t replicaId, const inputChannelMap_t &inputChannels, const outputChannelMap_t &outputChannels)
  {
    if (_replicas.contains(replicaId)) HICR_THROW_LOGIC("Replica %lu already registered.", replicaId);
    _replicas[replicaId] = ReplicaChannels{
      .inputChannels  = inputChannels,
      .outputChannels = outputChannels,
    };
    _readyReplicas.push(replicaId);
  }

  __INLINE__ void addMessageType(const messageType_t messageType) { _messageTypes.insert(messageType); }

  __INLINE__ void setCompletionCallback(completionCallback_t callback) { _completionCallback = callback; }

  __INLINE__ void submitJob(const std::shared_ptr<Job> &job)
  {
    const auto jobId = job->getMetadata().getId();
    {
      std::lock_guard lock(_jobsMutex);
      if (_jobs.contains(jobId)) { return; }
      _jobs[jobId] = job;
    }
    {
      std::lock_guard lock(_pendingJobsMutex);
      _pendingJobs.push(job);
    }
  }

  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> buildSubscriptions()
  {
    std::vector<serving::modules::Subscription> subscriptions;
    size_t                                      subscriptionCount = 0;
    for (const auto &[_, replica] : _replicas) subscriptionCount += replica.outputChannels.size() * _messageTypes.size();
    subscriptions.reserve(subscriptionCount);
    for (const auto &[replicaId, replica] : _replicas)
    {
      for (const auto &[outputName, readChannel] : replica.outputChannels)
      {
        for (const auto messageType : _messageTypes)
          subscriptions.emplace_back(
            messageType, readChannel, [this, replicaId, outputName](const input_t, const message_t &message) { this->replicaResponseHandler(replicaId, outputName, message); });
      }
    }
    return subscriptions;
  }

  [[nodiscard]] __INLINE__ std::vector<std::pair<messageType_t, input_t>> buildUnsubscriptions() const
  {
    std::vector<std::pair<messageType_t, input_t>> unsubscriptions;
    size_t                                         unsubscriptionCount = 0;
    for (const auto &[_, replica] : _replicas) unsubscriptionCount += replica.outputChannels.size() * _messageTypes.size();
    unsubscriptions.reserve(unsubscriptionCount);
    for (const auto &[_, replica] : _replicas)
    {
      for (const auto &[_, readChannel] : replica.outputChannels)
      {
        for (const auto &messageType : _messageTypes) { unsubscriptions.push_back({messageType, readChannel}); }
      }
    }
    return unsubscriptions;
  }

  void initialize() override
  {
    if (_messageTypes.empty()) HICR_THROW_LOGIC("Coordinator has no message types configured.");
    if (_replicas.empty()) HICR_THROW_LOGIC("Coordinator has no replicas.");
  }

  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override {}

  protected:

  void service() override { dispatchReadyJobs(); }

  private:

  struct ReplicaChannels
  {
    inputChannelMap_t  inputChannels;
    outputChannelMap_t outputChannels;
  };

  __INLINE__ void dispatchReadyJobs()
  {
    std::shared_ptr<Job> job = nullptr;
    {
      std::lock_guard lock(_pendingJobsMutex);
      if (_pendingJobs.empty()) return;
      job = _pendingJobs.front();
      _pendingJobs.pop();
    }

    if (job->isReady() == false)
    {
      std::lock_guard lock(_pendingJobsMutex);
      _pendingJobs.push(job);
      return;
    }

    instanceId_t replicaId;
    {
      std::lock_guard lock(_readyReplicasMutex);
      if (_readyReplicas.empty())
      {
        std::lock_guard pendingLock(_pendingJobsMutex);
        _pendingJobs.push(job);
        return;
      }
      replicaId = _readyReplicas.front();
      _readyReplicas.pop();
    }

    {
      std::lock_guard lock(_replicaJobsMutex);
      _replicaJobs[replicaId] = job->getMetadata().getId();
    }
    const auto &replica = _replicas.at(replicaId);
    for (auto &[inputName, dependency] : job->getInputDependencies())
    {
      if (replica.inputChannels.contains(inputName) == false) HICR_THROW_RUNTIME("Replica %lu has no input channel for dependency '%s'.", replicaId, inputName.c_str());
      const auto message = message_t(dependency.getDataPointer(), dependency.getDataSize(), job->getMetadata());
      replica.inputChannels.at(inputName)->pushMessageLocking(message);
      dependency.freeDataSlot();
    }
  }

  __INLINE__ void replicaResponseHandler(const instanceId_t replicaId, const edgeName_t &outputName, const message_t &message)
  {
    const auto jobId = message.getMetadata().getId();

    {
      std::lock_guard lock(_replicaJobsMutex);
      if (!_replicaJobs.contains(replicaId)) { HICR_THROW_RUNTIME("Received response from replica %lu with no assigned job.", replicaId); }
      if (_replicaJobs.at(replicaId) != jobId)
      {
        HICR_THROW_RUNTIME("Received response from replica %lu for job %lu, but expected job %lu.", replicaId, jobId, _replicaJobs.at(replicaId));
      }
    }

    std::shared_ptr<Job> job = nullptr;
    {
      std::lock_guard lock(_jobsMutex);
      job = _jobs.at(jobId);
    }

    auto &dependency = job->getOutputDependency(outputName);
    if (dependency.isSatisfied()) HICR_THROW_RUNTIME("Output dependency '%s' for job %lu is already satisfied.", outputName.c_str(), jobId);
    dependency.storeData(message.getData(), message.getSize());
    dependency.setSatisfied(true);

    if (job->isFinished() == false) return;

    std::vector<JobOutput> outputs;
    outputs.reserve(job->getOutputDependencies().size());
    for (auto &[_, outputDependency] : job->getOutputDependencies())
    {
      outputs.push_back(JobOutput{
        .name = outputDependency.getName(),
        .data = outputDependency.getData(),
      });
    }

    {
      std::lock_guard lock(_jobsMutex);
      _jobs.erase(jobId);
    }

    {
      std::lock_guard lock(_replicaJobsMutex);
      _replicaJobs.erase(replicaId);
    }

    {
      std::lock_guard lock(_readyReplicasMutex);
      _readyReplicas.push(replicaId);
    }
    if (_completionCallback != nullptr) { _completionCallback(job->getMetadata(), outputs); }

    for (auto &[_, outputDependency] : job->getOutputDependencies()) { outputDependency.freeDataSlot(); }
  }

  std::unordered_map<instanceId_t, ReplicaChannels> _replicas;

  std::unordered_set<messageType_t> _messageTypes;

  std::unordered_map<messageId_t, std::shared_ptr<Job>> _jobs;
  std::mutex                                            _jobsMutex;

  std::queue<std::shared_ptr<Job>> _pendingJobs;
  std::mutex                       _pendingJobsMutex;

  std::queue<instanceId_t> _readyReplicas;
  std::mutex               _readyReplicasMutex;

  std::unordered_map<instanceId_t, messageId_t> _replicaJobs;
  std::mutex                                    _replicaJobsMutex;

  completionCallback_t _completionCallback;
};
} // namespace serving::modules::roles::coordinator
