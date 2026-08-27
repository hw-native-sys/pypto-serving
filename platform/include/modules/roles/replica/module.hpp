#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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

#include <modules/roles/taskContext.hpp>

namespace serving::modules::roles::replica
{

class Module final : public serving::modules::Module
{
  public:

  using instanceId_t  = HiCR::Instance::instanceId_t;
  using input_t       = std::shared_ptr<serving::system::channels::Input>;
  using output_t      = std::shared_ptr<serving::system::channels::Output>;
  using message_t     = serving::system::channels::Message;
  using metadata_t    = serving::system::channels::Message::metadata_t;
  using messageId_t   = serving::system::channels::Message::messageId_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using edgeName_t    = std::string;
  using inputMap_t    = std::unordered_map<edgeName_t, input_t>;
  using outputMap_t   = std::unordered_map<edgeName_t, output_t>;
  using processFc_t   = std::function<void(serving::modules::roles::TaskContext &context)>;
  using memorySlot_t  = std::shared_ptr<HiCR::LocalMemorySlot>;

  Module(const std::string &taskName, const std::vector<edgeName_t> &inputNames, const std::vector<edgeName_t> &outputNames, const processFc_t &processFc)
    : serving::modules::Module(),
      _taskName(taskName),
      _inputNames(inputNames.begin(), inputNames.end()),
      _outputNames(outputNames.begin(), outputNames.end()),
      _isCoordinatorSet(false),
      _processFc(processFc)
  {}

  ~Module() override = default;

  __INLINE__ void setCoordinator(const instanceId_t coordinatorId, const outputMap_t &outputChannels, const inputMap_t &inputChannels)
  {
    if (_isCoordinatorSet) { HICR_THROW_LOGIC("Coordinator %lu already registered.", _coordinatorId); }
    _coordinatorId    = coordinatorId;
    _outputChannels   = outputChannels;
    _inputChannels    = inputChannels;
    _isCoordinatorSet = true;
  }

  __INLINE__ void addMessageType(const messageType_t messageType) { _messageTypes.insert(messageType); }

  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> buildSubscriptions()
  {
    std::vector<serving::modules::Subscription> out;
    out.reserve(_messageTypes.size() * _inputChannels.size());
    for (const auto &[inputName, inputChannel] : _inputChannels)
    {
      for (const auto messageType : _messageTypes)
        out.emplace_back(messageType, inputChannel, [this, inputName](const input_t, const message_t &message) { this->coordinatorMessageHandler(inputName, message); });
    }
    return out;
  }

  [[nodiscard]] __INLINE__ std::vector<std::pair<messageType_t, input_t>> buildUnsubscriptions() const
  {
    std::vector<std::pair<messageType_t, input_t>> out;
    out.reserve(_messageTypes.size() * _inputChannels.size());
    for (const auto &[_, inputChannel] : _inputChannels)
    {
      for (const auto messageType : _messageTypes) out.push_back({messageType, inputChannel});
    }
    return out;
  }

  void initialize() override
  {
    if (_isCoordinatorSet == false) HICR_THROW_LOGIC("Coordinator has not been set");
    if (_messageTypes.empty()) HICR_THROW_LOGIC("Replica has no message types configured.");
    for (const auto &inputName : _inputNames)
      if (_inputChannels.contains(inputName) == false) HICR_THROW_LOGIC("Replica has no input channel for dependency '%s'.", inputName.c_str());
    for (const auto &outputName : _outputNames)
      if (_outputChannels.contains(outputName) == false) HICR_THROW_LOGIC("Replica has no output channel for dependency '%s'.", outputName.c_str());
  }

  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override {}

  protected:

  void service() override {}

  private:

  struct ActiveJob
  {
    metadata_t                                   metadata;
    std::unordered_map<edgeName_t, memorySlot_t> inputs;
  };

  __INLINE__ void coordinatorMessageHandler(const edgeName_t &inputName, const message_t &message)
  {
    if (_inputNames.contains(inputName) == false) HICR_THROW_RUNTIME("Replica received undeclared input dependency '%s'.", inputName.c_str());
    if (message.getData() == nullptr && message.getSize() > 0) HICR_THROW_RUNTIME("Replica received null input data for dependency '%s' with non-zero size.", inputName.c_str());

    // Copy data before acquiring the lock to avoid serializing memcpy/fence under contention.
    const auto copiedData = copyMessageData(inputName, message);
    const auto jobId      = message.getMetadata().getId();
    ActiveJob  activeJob;
    {
      std::lock_guard lock(_activeJobsMutex);
      if (_activeJobs.contains(jobId) == false) _activeJobs[jobId] = ActiveJob{.metadata = message.getMetadata()};
      auto &job = _activeJobs.at(jobId);
      if (job.inputs.contains(inputName))
      {
        if (copiedData != nullptr) _inputChannels.at(inputName)->getConfig().payloadMemoryManager->freeLocalMemorySlot(copiedData);
        HICR_THROW_RUNTIME("Replica received duplicate input dependency '%s' for job %lu.", inputName.c_str(), jobId);
      }
      job.inputs[inputName] = copiedData;
      if (isReady(job) == false) return;

      activeJob = std::move(job);
      _activeJobs.erase(jobId);
    }

    auto                                 outputConfigs = buildOutputConfigs();
    serving::modules::roles::TaskContext context(_taskName, activeJob.metadata, activeJob.inputs, outputConfigs);

    // Scope guard: if _processFc or any output validation throws, free owned
    // outputs and input slots before unwinding to prevent slot leaks.
    struct CleanupGuard
    {
      serving::modules::roles::TaskContext &context;
      ActiveJob                            &activeJob;
      std::function<void(ActiveJob &)>      freeInputSlotsFn;
      bool                                  dismissed = false;
      ~CleanupGuard()
      {
        if (!dismissed)
        {
          context.freeOwnedOutputs();
          freeInputSlotsFn(activeJob);
        }
      }
    } guard{context, activeJob, [this](ActiveJob &j) { freeInputSlots(j); }};

    _processFc(context);

    std::unordered_set<edgeName_t> sentOutputs;
    for (const auto &output : context.getOutputs())
    {
      if (_outputNames.contains(output.name) == false) HICR_THROW_RUNTIME("Task '%s' set undeclared output dependency '%s'.", _taskName.c_str(), output.name.c_str());
      if (sentOutputs.contains(output.name)) HICR_THROW_RUNTIME("Task '%s' set output dependency '%s' more than once.", _taskName.c_str(), output.name.c_str());
      sentOutputs.insert(output.name);
    }

    for (const auto &outputName : _outputNames)
      if (sentOutputs.contains(outputName) == false) HICR_THROW_RUNTIME("Task '%s' did not set output dependency '%s'.", _taskName.c_str(), outputName.c_str());

    // All validation passed. Dismiss the guard so outputs stay alive for the
    // coordinator completion callback, which is responsible for freeing them.
    guard.dismissed = true;

    for (const auto &output : context.getOutputs())
    {
      const auto response = message_t(
        output.data == nullptr ? nullptr : static_cast<const uint8_t *>(output.data->getPointer()), output.data == nullptr ? 0 : output.data->getSize(), activeJob.metadata);
      _outputChannels.at(output.name)->pushMessageLocking(response);
    }

    // Input slots are no longer needed once outputs have been dispatched.
    freeInputSlots(activeJob);
  }

  [[nodiscard]] __INLINE__ memorySlot_t copyMessageData(const edgeName_t &inputName, const message_t &message) const
  {
    if (message.getSize() == 0) return nullptr;

    const auto &cfg     = _inputChannels.at(inputName)->getConfig();
    const auto  srcSlot = cfg.payloadMemoryManager->registerLocalMemorySlot(cfg.payloadMemorySpace, const_cast<uint8_t *>(message.getData()), message.getSize());
    auto        dstSlot = cfg.payloadMemoryManager->allocateLocalMemorySlot(cfg.payloadMemorySpace, message.getSize());
    cfg.payloadCommunicationManager->memcpy(dstSlot, 0, srcSlot, 0, message.getSize());
    cfg.payloadCommunicationManager->fence(dstSlot, 0, 1);
    cfg.payloadMemoryManager->deregisterLocalMemorySlot(srcSlot);
    return dstSlot;
  }

  [[nodiscard]] __INLINE__ serving::modules::roles::TaskContext::outputConfigMap_t buildOutputConfigs() const
  {
    serving::modules::roles::TaskContext::outputConfigMap_t configs;
    for (const auto &[name, output] : _outputChannels) configs.emplace(name, output->getConfig());
    return configs;
  }

  __INLINE__ void freeInputSlots(ActiveJob &job) const
  {
    for (auto &[name, slot] : job.inputs)
    {
      if (slot == nullptr) continue;
      _inputChannels.at(name)->getConfig().payloadMemoryManager->freeLocalMemorySlot(slot);
      slot = nullptr;
    }
  }

  [[nodiscard]] __INLINE__ bool isReady(const ActiveJob &job) const
  {
    for (const auto &inputName : _inputNames)
      if (job.inputs.contains(inputName) == false) return false;
    return true;
  }

  const std::string                    _taskName;
  const std::unordered_set<edgeName_t> _inputNames;
  const std::unordered_set<edgeName_t> _outputNames;

  bool        _isCoordinatorSet;
  processFc_t _processFc;

  HiCR::Instance::instanceId_t               _coordinatorId;
  inputMap_t                                 _inputChannels;
  outputMap_t                                _outputChannels;
  std::unordered_set<messageType_t>          _messageTypes;
  std::unordered_map<messageId_t, ActiveJob> _activeJobs;
  std::mutex                                 _activeJobsMutex;
};
} // namespace serving::modules::roles::replica
