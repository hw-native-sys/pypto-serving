#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hicr/core/communicationManager.hpp>
#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>
#include <hicr/core/globalMemorySlot.hpp>
#include <hicr/core/instance.hpp>

#include <modules/module.hpp>
#include <system/channels/input.hpp>
#include <system/channels/output.hpp>

namespace serving::modules::channelController
{

#define __SERVING_DEFAULT_CHANNEL_CONTROLLER_EXCHANGE_TAG 0x0000A100

/**
 * Creates channels using a reconciliation loop.
 * TODO: Right now every instance should be running this module as a service for
 * a backend like MPI. use channel creator frontend to mask this.
 */
class Module final : public serving::modules::Module
{
  public:

  using edgeName_t     = std::string;
  using instanceId_t   = HiCR::Instance::instanceId_t;
  using input_t        = serving::system::channels::Input;
  using output_t       = serving::system::channels::Output;
  using channelId_t    = serving::system::channels::channelId_t;
  using keyBuilderFc_t = serving::system::channels::keyBuilderFc_t;

  Module(const instanceId_t                               instance,
         const std::vector<HiCR::CommunicationManager *> &communicationManagersInOrder,
         const HiCR::GlobalMemorySlot::tag_t              exchangeTag = __SERVING_DEFAULT_CHANNEL_CONTROLLER_EXCHANGE_TAG,
         const size_t                                     intervalMs  = 250)
    : serving::modules::Module(intervalMs),
      _instanceId(instance),
      _communicationManagersInOrder(communicationManagersInOrder),
      _exchangeTag(exchangeTag)
  {
    if (_communicationManagersInOrder.empty()) HICR_THROW_LOGIC("Channel controller requires at least one communication manager.");
  }

  ~Module() override = default;

  [[nodiscard]] __INLINE__ std::weak_ptr<output_t> addDesiredProducer(const instanceId_t                  targetId,
                                                                      const channelId_t                   channelId,
                                                                      const serving::configuration::Edge &edge,
                                                                      const keyBuilderFc_t               &keyBuilder)
  {
    std::lock_guard lock(_producerMutex);
    const auto     &edgeName = edge.getName();
    if (_desiredProducers.contains(edgeName)) HICR_THROW_LOGIC("Desired producer for edge %s to target instance %lu already exists.", edgeName.c_str(), targetId);
    _desiredProducers[edgeName] = std::make_shared<serving::system::channels::Output>(edge, channelId, _instanceId, targetId, keyBuilder);
    return _desiredProducers[edgeName];
  }

  [[nodiscard]] __INLINE__ std::weak_ptr<input_t> addDesiredConsumer(const instanceId_t                  sourceId,
                                                                     const channelId_t                   channelId,
                                                                     const serving::configuration::Edge &edge,
                                                                     const keyBuilderFc_t               &keyBuilder)
  {
    std::lock_guard lock(_consumerMutex);
    const auto     &edgeName = edge.getName();
    if (_desiredConsumers.contains(edgeName)) HICR_THROW_LOGIC("Desired consumer for edge %s from source instance %lu already exists.", edgeName.c_str(), sourceId);
    _desiredConsumers[edgeName] = std::make_shared<serving::system::channels::Input>(edge, channelId, sourceId, _instanceId, keyBuilder);
    return _desiredConsumers[edgeName];
  }

  [[nodiscard]] __INLINE__ bool hasProducer(const edgeName_t edgeName) const
  {
    std::lock_guard lock(_producerMutex);
    return _actualProducers.contains(edgeName);
  }
  [[nodiscard]] __INLINE__ bool hasConsumer(const edgeName_t edgeName) const
  {
    std::lock_guard lock(_consumerMutex);
    return _actualConsumers.contains(edgeName);
  }

  [[nodiscard]] __INLINE__ std::weak_ptr<output_t> getProducer(const edgeName_t edgeName) const
  {
    std::lock_guard lock(_producerMutex);
    if (_actualProducers.contains(edgeName) == false) HICR_THROW_RUNTIME("No actual producer found for edge %s.", edgeName.c_str());
    return _actualProducers.at(edgeName);
  }

  [[nodiscard]] __INLINE__ std::weak_ptr<input_t> getConsumer(const edgeName_t edgeName) const
  {
    std::lock_guard lock(_consumerMutex);
    if (_actualConsumers.contains(edgeName) == false) HICR_THROW_RUNTIME("No actual consumer found for edge %s.", edgeName.c_str());
    return _actualConsumers.at(edgeName);
  }

  __INLINE__ void removeDesiredProducer(const edgeName_t edgeName)
  {
    std::lock_guard lock(_producerMutex);
    _desiredProducers.erase(edgeName);
  }

  __INLINE__ void removeDesiredConsumer(const edgeName_t edgeName)
  {
    std::lock_guard lock(_consumerMutex);
    _desiredConsumers.erase(edgeName);
  }

  void initialize() override { reconcile(); }
  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override
  {
    std::scoped_lock lock(_producerMutex, _consumerMutex);
    _desiredProducers.clear();
    _actualProducers.clear();
    _desiredConsumers.clear();
    _actualConsumers.clear();
  }

  protected:

  void service() override { reconcile(); }

  private:

  __INLINE__ void reconcile()
  {
    std::unordered_map<edgeName_t, std::shared_ptr<output_t>> producersToCreate;
    std::unordered_map<edgeName_t, std::shared_ptr<input_t>>  consumersToCreate;
    std::vector<edgeName_t>                                   producersToDelete;
    std::vector<edgeName_t>                                   consumersToDelete;

    // Diff desired vs actual
    {
      std::scoped_lock lock(_producerMutex, _consumerMutex);
      for (const auto &[target, output] : _desiredProducers)
        if (_actualProducers.contains(target) == false) producersToCreate[target] = output;
      for (const auto &[source, input] : _desiredConsumers)
        if (_actualConsumers.contains(source) == false) consumersToCreate[source] = input;
      for (const auto &[target, _] : _actualProducers)
        if (_desiredProducers.contains(target) == false) producersToDelete.push_back(target);
      for (const auto &[source, _] : _actualConsumers)
        if (_desiredConsumers.contains(source) == false) consumersToDelete.push_back(source);
    }

    // Create missing channels
    if (producersToCreate.empty() == false || consumersToCreate.empty() == false)
    {
      std::vector<serving::system::channels::memorySlotExchangeInfo_t> memorySlotsToExchange;
      for (const auto &[_, output] : producersToCreate) output->getMemorySlotsToExchange(memorySlotsToExchange);
      for (const auto &[_, input] : consumersToCreate) input->getMemorySlotsToExchange(memorySlotsToExchange);

      std::map<HiCR::CommunicationManager *, std::vector<HiCR::CommunicationManager::globalKeyMemorySlotPair_t>> exchangeMap;
      for (const auto manager : _communicationManagersInOrder) exchangeMap[manager] = {};

      for (const auto &entry : memorySlotsToExchange)
      {
        if (exchangeMap.contains(entry.communicationManager) == false) HICR_THROW_LOGIC("Memory slot exchange manager is not present in configured manager order.");
        exchangeMap[entry.communicationManager].push_back({entry.globalKey, entry.memorySlot});
      }

      for (const auto manager : _communicationManagersInOrder) manager->exchangeGlobalMemorySlots(_exchangeTag, exchangeMap[manager]);
      for (const auto manager : _communicationManagersInOrder) manager->fence(_exchangeTag);

      for (const auto &[_, output] : producersToCreate) output->initialize(_exchangeTag);
      for (const auto &[_, input] : consumersToCreate) input->initialize(_exchangeTag);

      {
        std::scoped_lock lock(_producerMutex, _consumerMutex);
        for (const auto &[target, output] : producersToCreate) { _actualProducers[target] = output; }
        for (const auto &[source, input] : consumersToCreate) { _actualConsumers[source] = input; }
      }
    }

    // Delete stale channels
    std::vector<std::shared_ptr<output_t>> staleProducers;
    std::vector<std::shared_ptr<input_t>>  staleConsumers;
    {
      std::scoped_lock lock(_producerMutex, _consumerMutex);
      staleProducers.reserve(producersToDelete.size());
      for (const auto &target : producersToDelete)
      {
        auto it = _actualProducers.find(target);
        if (it == _actualProducers.end()) continue;
        staleProducers.push_back(std::move(it->second));
        _actualProducers.erase(it);
      }
      staleConsumers.reserve(consumersToDelete.size());
      for (const auto &source : consumersToDelete)
      {
        auto it = _actualConsumers.find(source);
        if (it == _actualConsumers.end()) continue;
        staleConsumers.push_back(std::move(it->second));
        _actualConsumers.erase(it);
      }
    }
  }

  const instanceId_t _instanceId;

  mutable std::mutex                              _producerMutex;
  mutable std::mutex                              _consumerMutex;
  const std::vector<HiCR::CommunicationManager *> _communicationManagersInOrder;
  const HiCR::GlobalMemorySlot::tag_t             _exchangeTag;

  // Desired state
  std::unordered_map<edgeName_t, std::shared_ptr<output_t>> _desiredProducers; // key: target instance
  std::unordered_map<edgeName_t, std::shared_ptr<input_t>>  _desiredConsumers; // key: source instance

  // Actual created state
  std::unordered_map<edgeName_t, std::shared_ptr<output_t>> _actualProducers; // key: target instance
  std::unordered_map<edgeName_t, std::shared_ptr<input_t>>  _actualConsumers; // key: source instance
};
} // namespace serving::modules::channelController
