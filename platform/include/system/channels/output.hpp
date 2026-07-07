#pragma once
#include <chrono>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/frontends/channel/fixedSize/spsc/producer.hpp>
#include <hicr/frontends/channel/variableSize/spsc/producer.hpp>

#include <modules/configuration/edge.hpp>

#include "base.hpp"
#include "message.hpp"

namespace serving::system::channels
{

class Output final : public Base
{
  public:

  Output(const serving::configuration::Edge &edgeConfig,
         const channelId_t                   channelIndex,
         const HiCR::Instance::instanceId_t  sourceIndex,
         const HiCR::Instance::instanceId_t  targetIndex,
         const keyBuilderFc_t               &keyBuilder)
    : Base(buildChannelConfig(edgeConfig), buildSlotKeys(channelIndex, sourceIndex, targetIndex, keyBuilder)),
      _targetInstance(targetIndex)
  {
    const auto &cfg                    = _config;
    _dataChannelProducerSizeInfoBuffer = cfg.coordinationMemoryManager->allocateLocalMemorySlot(cfg.coordinationMemorySpace, sizeof(size_t));
  }

  ~Output() override
  {
    const auto &cfg = _config;
    cfg.coordinationMemoryManager->freeLocalMemorySlot(_dataChannelProducerSizeInfoBuffer);
  }

  __INLINE__ void getMemorySlotsToExchange(std::vector<memorySlotExchangeInfo_t> &memorySlots) const override
  {
    const auto &cfg  = _config;
    const auto &keys = _slotKeys;
    // Data channel producer coordination slots
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.dataProducerCoordinationBufferForSizesKey,
                                                   .memorySlot           = _dataChannelLocalCoordinationBufferForSizes});
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.dataProducerCoordinationBufferForPayloadKey,
                                                   .memorySlot           = _dataChannelLocalCoordinationBufferForPayloads});
    // Metadata channel producer coordination slot
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.metadataProducerCoordinationBufferKey,
                                                   .memorySlot           = _metadataChannelLocalCoordinationBuffer});
  }

  __INLINE__ bool isFull(const size_t messageSize) const
  {
    checkReady();
    _metadataChannel->updateDepth();
    if (_metadataChannel->isFull() == true) return true;
    _dataChannel->updateDepth();
    if (_dataChannel->isFull(messageSize) == true) return true;
    return false;
  }

  __INLINE__ void pushMessageLocking(const Message message)
  {
    while (true)
    {
      std::unique_lock<std::mutex> guard(_lock);
      if (!isFull(message.getSize()))
      {
        pushMessage(message);
        return;
      }
      guard.unlock();
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }

  __INLINE__ const HiCR::Instance::instanceId_t getTargetInstance() const { return _targetInstance; }

  private:

  __INLINE__ void pushMessage(const Message message) const
  {
    if (isFull(message.getSize()) == true) HICR_THROW_RUNTIME("Trying to push a message when channel is full. This is a bug in serving.");
    const auto &cfg         = _config;
    auto        payloadSlot = cfg.payloadMemoryManager->registerLocalMemorySlot(cfg.payloadMemorySpace, (void *)message.getData(), message.getSize());
    _dataChannel->push(payloadSlot);
    auto metadataSlot = cfg.coordinationMemoryManager->registerLocalMemorySlot(cfg.coordinationMemorySpace, (void *)&message.getMetadata(), sizeof(Message::metadata_t));
    _metadataChannel->push(metadataSlot);
    cfg.payloadMemoryManager->deregisterLocalMemorySlot(payloadSlot);
    cfg.coordinationMemoryManager->deregisterLocalMemorySlot(metadataSlot);
  }

  __INLINE__ void createChannels() override
  {
    const auto &cfg = _config;
    // Producer data channel
    _dataChannel = std::make_shared<HiCR::channel::variableSize::SPSC::Producer>(*cfg.coordinationCommunicationManager,
                                                                                 *cfg.payloadCommunicationManager,
                                                                                 _dataChannelProducerSizeInfoBuffer,
                                                                                 _dataChannelConsumerPayloadBuffer,
                                                                                 _dataChannelConsumerSizesBuffer,
                                                                                 _dataChannelProducerCoordinationBufferForSizes->getSourceLocalMemorySlot(),
                                                                                 _dataChannelProducerCoordinationBufferForPayloads->getSourceLocalMemorySlot(),
                                                                                 _dataChannelConsumerCoordinationBufferForSizes,
                                                                                 _dataChannelConsumerCoordinationBufferForPayloads,
                                                                                 cfg.bufferSize,
                                                                                 sizeof(uint8_t),
                                                                                 cfg.bufferCapacity);
    // Producer metadata channel
    _metadataChannel = std::make_shared<HiCR::channel::fixedSize::SPSC::Producer>(*cfg.coordinationCommunicationManager,
                                                                                  *cfg.coordinationCommunicationManager,
                                                                                  _metadataChannelConsumerPayloadBuffer,
                                                                                  _metadataChannelProducerCoordinationBuffer->getSourceLocalMemorySlot(),
                                                                                  _metadataChannelConsumerCoordinationBuffer,
                                                                                  sizeof(Message::metadata_t),
                                                                                  cfg.bufferCapacity);
  }

  const HiCR::Instance::instanceId_t                           _targetInstance;
  std::shared_ptr<HiCR::LocalMemorySlot>                       _dataChannelProducerSizeInfoBuffer;
  std::shared_ptr<HiCR::channel::variableSize::SPSC::Producer> _dataChannel;
  std::shared_ptr<HiCR::channel::fixedSize::SPSC::Producer>    _metadataChannel;
};
} // namespace serving::system::channels