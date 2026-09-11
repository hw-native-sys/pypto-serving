#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
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

#include <system/runtimePlan.hpp>

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
  using drainCallback_t      = std::function<void(instanceId_t replicaId)>;

  // partitionName is recorded in RuntimePlan snapshots.
  Module(const size_t intervalMs, const std::string &partitionName = "")
    : modules::Module(intervalMs),
      _partitionName(partitionName)
  {}

  using inputChannelMap_t  = std::unordered_map<edgeName_t, output_t>;
  using outputChannelMap_t = std::unordered_map<edgeName_t, input_t>;

  __INLINE__ void addReplica(const instanceId_t replicaId, const inputChannelMap_t &inputChannels, const outputChannelMap_t &outputChannels)
  {
    std::unique_lock lock(_replicasMutex);
    if (_replicas.contains(replicaId)) HICR_THROW_LOGIC("Replica %lu already registered.", replicaId);
    _replicas[replicaId] = ReplicaEntry{
      .inputChannels  = inputChannels,
      .outputChannels = outputChannels,
      .status         = system::RuntimePlan::ReplicaStatus::Active,
    };
    _readyReplicas.push(replicaId);
    _planVersion.fetch_add(1, std::memory_order_relaxed);
  }

  // Hot-add a replica to a running coordinator.  Safe to call concurrently
  // with job dispatch and completion handling.  Returns the new subscriptions
  // to register with channelDispatcher before the replica can receive work.
  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> addReplicaLive(const instanceId_t        replicaId,
                                                                                      const inputChannelMap_t  &inputChannels,
                                                                                      const outputChannelMap_t &outputChannels)
  {
    std::vector<serving::modules::Subscription> subs;
    {
      std::unique_lock lock(_replicasMutex);
      if (_replicas.contains(replicaId)) HICR_THROW_LOGIC("addReplicaLive: replica %lu already registered.", replicaId);
      _replicas[replicaId] = ReplicaEntry{
        .inputChannels  = inputChannels,
        .outputChannels = outputChannels,
        .status         = system::RuntimePlan::ReplicaStatus::Active,
      };
      _planVersion.fetch_add(1, std::memory_order_relaxed);
      for (const auto &[outputName, readChannel] : outputChannels)
        for (const auto messageType : _messageTypes)
          subs.emplace_back(
            messageType, readChannel, [this, replicaId, outputName](const input_t, const message_t &message) { this->replicaResponseHandler(replicaId, outputName, message); });
    }
    {
      std::lock_guard readyLock(_readyReplicasMutex);
      _readyReplicas.push(replicaId);
    }
    return subs;
  }

  // Begin draining replica replicaId: no new jobs will be dispatched to it.
  // If the replica is idle when drained (no in-flight job), the drainCallback
  // fires synchronously before this method returns.  If the replica has a job
  // in flight, the drainCallback fires from replicaResponseHandler when that
  // job completes.  removeReplica is safe only after the callback fires.
  __INLINE__ void drainReplica(const instanceId_t replicaId)
  {
    {
      std::unique_lock lock(_replicasMutex);
      if (_replicas.contains(replicaId) == false) HICR_THROW_LOGIC("drainReplica: unknown replica %lu.", replicaId);
      if (_replicas.at(replicaId).status != system::RuntimePlan::ReplicaStatus::Active) HICR_THROW_LOGIC("drainReplica: replica %lu is not Active.", replicaId);
      _replicas.at(replicaId).status = system::RuntimePlan::ReplicaStatus::Draining;
      _planVersion.fetch_add(1, std::memory_order_relaxed);
    }
    // Check idle (no in-flight job) outside _replicasMutex to avoid ordering
    // inversion with _replicaJobsMutex.  A narrow race exists: if dispatch is
    // mid-assignment (status checked as Active, not yet in _replicaJobs), the
    // callback may fire before the last job completes.  Callers who drain only
    // after waiting for all pending completions do not encounter this race.
    bool idle = false;
    {
      std::lock_guard jobsLock(_replicaJobsMutex);
      idle = !_replicaJobs.contains(replicaId);
    }
    if (idle)
    {
      bool shouldNotify = false;
      {
        std::unique_lock lock(_replicasMutex);
        auto            &entry = _replicas.at(replicaId);
        if (!entry.idleNotified)
        {
          entry.idleNotified = true;
          shouldNotify       = true;
        }
      }
      if (shouldNotify && _drainCallback) _drainCallback(replicaId);
    }
  }

  // Remove a drained replica. The replica must be Draining (not Active).
  // After this call the replica's channels may be torn down safely.
  __INLINE__ void removeReplica(const instanceId_t replicaId)
  {
    std::unique_lock lock(_replicasMutex);
    if (_replicas.contains(replicaId) == false) HICR_THROW_LOGIC("removeReplica: unknown replica %lu.", replicaId);
    if (_replicas.at(replicaId).status != system::RuntimePlan::ReplicaStatus::Draining) HICR_THROW_LOGIC("removeReplica: replica %lu must be Draining before removal.", replicaId);
    _replicas.at(replicaId).status = system::RuntimePlan::ReplicaStatus::Removed;
    _planVersion.fetch_add(1, std::memory_order_relaxed);
  }

  // Return an immutable snapshot of the current deployment runtime state.
  [[nodiscard]] __INLINE__ system::RuntimePlan getRuntimePlan() const
  {
    system::RuntimePlan plan;
    plan.version = _planVersion.load(std::memory_order_relaxed);

    system::RuntimePlan::PartitionState part;
    part.name = _partitionName;
    {
      std::shared_lock lock(_replicasMutex);
      part.replicas.reserve(_replicas.size());
      for (const auto &[id, entry] : _replicas) part.replicas.push_back({id, entry.status});
    }
    plan.partitions.push_back(std::move(part));
    return plan;
  }

  __INLINE__ void addMessageType(const messageType_t messageType) { _messageTypes.insert(messageType); }

  __INLINE__ void setCompletionCallback(completionCallback_t callback) { _completionCallback = callback; }

  // Called exactly once per replica when it transitions Draining → idle (no
  // more in-flight jobs and no new jobs will be assigned).  Safe to call
  // removeReplica() and tear down channels from inside the callback.
  __INLINE__ void setDrainCallback(drainCallback_t callback) { _drainCallback = std::move(callback); }

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

  struct ReplicaEntry
  {
    inputChannelMap_t                  inputChannels;
    outputChannelMap_t                 outputChannels;
    system::RuntimePlan::ReplicaStatus status{system::RuntimePlan::ReplicaStatus::Active};
    bool                               idleNotified{false};
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

    // Find the next Active replica, discarding any Draining/Removed entries
    // that were enqueued before a status transition.  Collect drain
    // notifications so the callback fires outside all locks.
    instanceId_t              replicaId;
    bool                      found = false;
    std::vector<instanceId_t> drainNotifications;
    {
      std::lock_guard lock(_readyReplicasMutex);
      while (!_readyReplicas.empty())
      {
        const auto candidate = _readyReplicas.front();
        _readyReplicas.pop();
        std::unique_lock statusLock(_replicasMutex);
        if (!_replicas.contains(candidate)) continue;
        auto &entry = _replicas.at(candidate);
        if (entry.status == system::RuntimePlan::ReplicaStatus::Active)
        {
          replicaId = candidate;
          found     = true;
          break;
        }
        if (entry.status == system::RuntimePlan::ReplicaStatus::Draining && !entry.idleNotified)
        {
          entry.idleNotified = true;
          drainNotifications.push_back(candidate);
        }
      }
    }
    for (const auto id : drainNotifications)
      if (_drainCallback) _drainCallback(id);
    if (!found)
    {
      std::lock_guard pendingLock(_pendingJobsMutex);
      _pendingJobs.push(job);
      return;
    }

    {
      std::lock_guard lock(_replicaJobsMutex);
      _replicaJobs[replicaId] = job->getMetadata().getId();
    }
    std::shared_lock replicasLock(_replicasMutex);
    const auto      &replica = _replicas.at(replicaId);
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

    // Re-queue only if Active.  If Draining and not yet notified, fire the
    // drain callback (exactly once, guarded by idleNotified).  Both checks
    // happen under _replicasMutex; the callback and enqueue happen
    // outside all locks to avoid inversion and allow re-entrant calls.
    bool shouldRequeue = false;
    bool drainIdle     = false;
    {
      std::unique_lock statusLock(_replicasMutex);
      if (_replicas.contains(replicaId))
      {
        auto &entry = _replicas.at(replicaId);
        if (entry.status == system::RuntimePlan::ReplicaStatus::Active) { shouldRequeue = true; }
        else if (entry.status == system::RuntimePlan::ReplicaStatus::Draining && !entry.idleNotified)
        {
          entry.idleNotified = true;
          drainIdle          = true;
        }
      }
    }
    if (shouldRequeue)
    {
      std::lock_guard lock(_readyReplicasMutex);
      _readyReplicas.push(replicaId);
    }
    if (drainIdle && _drainCallback) _drainCallback(replicaId);
    if (_completionCallback != nullptr) { _completionCallback(job->getMetadata(), outputs); }

    for (auto &[_, outputDependency] : job->getOutputDependencies()) { outputDependency.freeDataSlot(); }
  }

  std::string               _partitionName;
  std::atomic<uint64_t>     _planVersion{0};
  mutable std::shared_mutex _replicasMutex;

  std::unordered_map<instanceId_t, ReplicaEntry> _replicas;

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
  drainCallback_t      _drainCallback;
};
} // namespace serving::modules::roles::coordinator
