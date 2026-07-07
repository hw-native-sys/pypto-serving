#pragma once
#include <functional>
#include <memory>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/frontends/channel/fixedSize/spsc/consumer.hpp>
#include <hicr/frontends/channel/variableSize/spsc/consumer.hpp>

#include <modules/configuration/edge.hpp>

#include "base.hpp"
#include "message.hpp"

namespace serving::system::channels
{
class Input final : public Base
{
  public:

  Input(const serving::configuration::Edge &edgeConfig,
        const channelId_t                   channelIndex,
        const HiCR::Instance::instanceId_t  sourceIndex,
        const HiCR::Instance::instanceId_t  targetIndex,
        const keyBuilderFc_t               &keyBuilder)
    : Base(buildChannelConfig(edgeConfig), buildSlotKeys(channelIndex, sourceIndex, targetIndex, keyBuilder)),
      _sourceInstance(sourceIndex)
  {
    const auto &cfg = _config;
    // Allocating additional local buffers required for the consumer data channel
    const auto sizesBufferSize   = HiCR::channel::variableSize::Base::getTokenBufferSize(sizeof(size_t), cfg.bufferCapacity);
    _dataChannelSizesBuffer      = cfg.coordinationMemoryManager->allocateLocalMemorySlot(cfg.coordinationMemorySpace, sizesBufferSize);
    const auto payloadBufferSize = HiCR::channel::variableSize::SPSC::Consumer::getPayloadBufferSize(cfg.bufferSize);
    _dataChannelPayloadBuffer    = cfg.payloadMemoryManager->allocateLocalMemorySlot(cfg.payloadMemorySpace, payloadBufferSize);
    // Allocating additional local buffers required for the consumer metadata channel
    const auto metadataBufferSize = HiCR::channel::fixedSize::Base::getTokenBufferSize(sizeof(Message::metadata_t), cfg.bufferCapacity);
    _metadataChannelPayloadBuffer = cfg.coordinationMemoryManager->allocateLocalMemorySlot(cfg.coordinationMemorySpace, metadataBufferSize);
  }

  ~Input() override
  {
    const auto &cfg = _config;
    cfg.coordinationMemoryManager->freeLocalMemorySlot(_dataChannelSizesBuffer);
    cfg.payloadMemoryManager->freeLocalMemorySlot(_dataChannelPayloadBuffer);
    cfg.coordinationMemoryManager->freeLocalMemorySlot(_metadataChannelPayloadBuffer);
  }

  __INLINE__ void getMemorySlotsToExchange(std::vector<memorySlotExchangeInfo_t> &memorySlots) const override
  {
    const auto &cfg  = _config;
    const auto &keys = _slotKeys;
    // Data channel slots
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.dataConsumerCoordinationBufferForSizesKey,
                                                   .memorySlot           = _dataChannelLocalCoordinationBufferForSizes});
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.dataConsumerCoordinationBufferForPayloadKey,
                                                   .memorySlot           = _dataChannelLocalCoordinationBufferForPayloads});
    memorySlots.push_back(
      memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager, .globalKey = keys.dataConsumerSizesBufferKey, .memorySlot = _dataChannelSizesBuffer});
    memorySlots.push_back(
      memorySlotExchangeInfo_t{.communicationManager = cfg.payloadCommunicationManager, .globalKey = keys.dataConsumerPayloadBufferKey, .memorySlot = _dataChannelPayloadBuffer});
    // Metadata channel slots
    memorySlots.push_back(memorySlotExchangeInfo_t{.communicationManager = cfg.coordinationCommunicationManager,
                                                   .globalKey            = keys.metadataConsumerCoordinationBufferKey,
                                                   .memorySlot           = _metadataChannelLocalCoordinationBuffer});
    memorySlots.push_back(memorySlotExchangeInfo_t{
      .communicationManager = cfg.coordinationCommunicationManager, .globalKey = keys.metadataConsumerPayloadBufferKey, .memorySlot = _metadataChannelPayloadBuffer});
  }

  __INLINE__ bool hasMessage() const
  {
    checkReady();
    _metadataChannel->updateDepth();
    if (_metadataChannel->isEmpty() == true) return false;
    _dataChannel->updateDepth();
    if (_dataChannel->isEmpty() == true) return false;
    return true;
  }

  __INLINE__ Message getMessage() const
  {
    if (hasMessage() == false) HICR_THROW_RUNTIME("Trying to get message when there is none available. This is a bug in serving.");
    // Data payload
    const auto dataBufferPtr   = (uint8_t *)_dataChannel->getPayloadBufferMemorySlot()->getSourceLocalMemorySlot()->getPointer();
    const auto dataToken       = _dataChannel->peek();
    const auto dataMessagePos  = dataToken[0];
    const auto dataMessagePtr  = &dataBufferPtr[dataMessagePos];
    const auto dataMessageSize = dataToken[1];
    // Metadata payload
    const auto metadataBufferPtr  = (Message::metadata_t *)_metadataChannel->getTokenBuffer()->getSourceLocalMemorySlot()->getPointer();
    const auto metadataToken      = _metadataChannel->peek();
    const auto metadataMessagePos = metadataToken;
    const auto metadataMessagePtr = &metadataBufferPtr[metadataMessagePos];
    const auto metadata           = *metadataMessagePtr;
    return Message(dataMessagePtr, dataMessageSize, metadata);
  }

  __INLINE__ void popMessage()
  {
    if (hasMessage() == false) HICR_THROW_RUNTIME("Trying to pop message when there is none available. This is a bug in serving.");
    _dataChannel->pop();
    _metadataChannel->pop();
  }

  __INLINE__ const HiCR::Instance::instanceId_t &getSourceInstance() const { return _sourceInstance; }

  private:

  __INLINE__ void createChannels() override
  {
    const auto &cfg = _config;
    // Consumer data channel
    _dataChannel = std::make_shared<HiCR::channel::variableSize::SPSC::Consumer>(*cfg.coordinationCommunicationManager,
                                                                                 *cfg.payloadCommunicationManager,
                                                                                 _dataChannelConsumerPayloadBuffer,
                                                                                 _dataChannelConsumerSizesBuffer,
                                                                                 _dataChannelConsumerCoordinationBufferForSizes->getSourceLocalMemorySlot(),
                                                                                 _dataChannelConsumerCoordinationBufferForPayloads->getSourceLocalMemorySlot(),
                                                                                 _dataChannelProducerCoordinationBufferForSizes,
                                                                                 _dataChannelProducerCoordinationBufferForPayloads,
                                                                                 cfg.bufferSize,
                                                                                 cfg.bufferCapacity);
    // Consumer metadata channel
    _metadataChannel = std::make_shared<HiCR::channel::fixedSize::SPSC::Consumer>(*cfg.coordinationCommunicationManager,
                                                                                  *cfg.coordinationCommunicationManager,
                                                                                  _metadataChannelConsumerPayloadBuffer,
                                                                                  _metadataChannelConsumerCoordinationBuffer->getSourceLocalMemorySlot(),
                                                                                  _metadataChannelProducerCoordinationBuffer,
                                                                                  sizeof(Message::metadata_t),
                                                                                  cfg.bufferCapacity);
  }

  const HiCR::Instance::instanceId_t                           _sourceInstance;
  std::shared_ptr<HiCR::LocalMemorySlot>                       _dataChannelSizesBuffer;
  std::shared_ptr<HiCR::LocalMemorySlot>                       _dataChannelPayloadBuffer;
  std::shared_ptr<HiCR::LocalMemorySlot>                       _metadataChannelPayloadBuffer;
  std::shared_ptr<HiCR::channel::variableSize::SPSC::Consumer> _dataChannel;
  std::shared_ptr<HiCR::channel::fixedSize::SPSC::Consumer>    _metadataChannel;
};
} // namespace serving::system::channels