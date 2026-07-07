#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include <hicr/core/communicationManager.hpp>
#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>
#include <hicr/core/globalMemorySlot.hpp>
#include <hicr/core/instance.hpp>
#include <hicr/core/localMemorySlot.hpp>
#include <hicr/core/memoryManager.hpp>
#include <hicr/core/memorySpace.hpp>
#include <hicr/frontends/channel/base.hpp>

#include <modules/configuration/edge.hpp>

namespace serving::system::channels
{

/**
 * Configuration structure for the channel base class, containing all necessary information to create and initialize the channels.
 */
struct channelConfig_t
{
  std::string                        name                             = "";
  size_t                             bufferCapacity                   = 1;
  size_t                             bufferSize                       = 0;
  HiCR::CommunicationManager        *payloadCommunicationManager      = nullptr;
  HiCR::MemoryManager               *payloadMemoryManager             = nullptr;
  std::shared_ptr<HiCR::MemorySpace> payloadMemorySpace               = nullptr;
  HiCR::CommunicationManager        *coordinationCommunicationManager = nullptr;
  HiCR::MemoryManager               *coordinationMemoryManager        = nullptr;
  std::shared_ptr<HiCR::MemorySpace> coordinationMemorySpace          = nullptr;
};

/**
 * Structure containing the global keys for all the global memory slots that need to be exchanged for the creation of the channels.
 */
struct slotKeys_t
{
  // Data channel keys
  HiCR::GlobalMemorySlot::globalKey_t dataConsumerSizesBufferKey                  = 0;
  HiCR::GlobalMemorySlot::globalKey_t dataConsumerPayloadBufferKey                = 0;
  HiCR::GlobalMemorySlot::globalKey_t dataConsumerCoordinationBufferForSizesKey   = 0;
  HiCR::GlobalMemorySlot::globalKey_t dataConsumerCoordinationBufferForPayloadKey = 0;
  HiCR::GlobalMemorySlot::globalKey_t dataProducerCoordinationBufferForSizesKey   = 0;
  HiCR::GlobalMemorySlot::globalKey_t dataProducerCoordinationBufferForPayloadKey = 0;
  // Metadata channel keys
  HiCR::GlobalMemorySlot::globalKey_t metadataConsumerPayloadBufferKey      = 0;
  HiCR::GlobalMemorySlot::globalKey_t metadataConsumerCoordinationBufferKey = 0;
  HiCR::GlobalMemorySlot::globalKey_t metadataProducerCoordinationBufferKey = 0;
};

using channelId_t = uint64_t;
/**
 * Type definition for the key builder function, which is used to build the global keys for the global memory slots that need to be exchanged.
 */
using keyBuilderFc_t =
  std::function<slotKeys_t(const HiCR::Instance::instanceId_t sourceInstanceId, const HiCR::Instance::instanceId_t targetInstanceId, const channelId_t channelId)>;

/**
 * Structure containing the information of a memory slot that needs to be exchanged.
 */
struct memorySlotExchangeInfo_t
{
  HiCR::CommunicationManager            *communicationManager = nullptr;
  HiCR::GlobalMemorySlot::globalKey_t    globalKey            = 0;
  std::shared_ptr<HiCR::LocalMemorySlot> memorySlot           = nullptr;
};

/**
 * Base class for the channels, containing all the common information and functions for both producer and consumer channels.
 */
class Base
{
  public:

  Base(const channelConfig_t &config, const slotKeys_t &slotKeys)
    : _config(config),
      _slotKeys(slotKeys),
      _isReady(false)
  {
    validateConfig();
    const auto coordinationBufferSize              = HiCR::channel::Base::getCoordinationBufferSize();
    _dataChannelLocalCoordinationBufferForSizes    = _config.coordinationMemoryManager->allocateLocalMemorySlot(_config.coordinationMemorySpace, coordinationBufferSize);
    _dataChannelLocalCoordinationBufferForPayloads = _config.coordinationMemoryManager->allocateLocalMemorySlot(_config.coordinationMemorySpace, coordinationBufferSize);
    _metadataChannelLocalCoordinationBuffer        = _config.coordinationMemoryManager->allocateLocalMemorySlot(_config.coordinationMemorySpace, coordinationBufferSize);
    HiCR::channel::Base::initializeCoordinationBuffer(_dataChannelLocalCoordinationBufferForSizes);
    HiCR::channel::Base::initializeCoordinationBuffer(_dataChannelLocalCoordinationBufferForPayloads);
    HiCR::channel::Base::initializeCoordinationBuffer(_metadataChannelLocalCoordinationBuffer);
  }

  virtual ~Base()
  {
    _config.coordinationMemoryManager->freeLocalMemorySlot(_dataChannelLocalCoordinationBufferForSizes);
    _config.coordinationMemoryManager->freeLocalMemorySlot(_dataChannelLocalCoordinationBufferForPayloads);
    _config.coordinationMemoryManager->freeLocalMemorySlot(_metadataChannelLocalCoordinationBuffer);
  }

  virtual void getMemorySlotsToExchange(std::vector<memorySlotExchangeInfo_t> &memorySlots) const = 0;

  __INLINE__ void initialize(const HiCR::GlobalMemorySlot::tag_t tag)
  {
    // Data channel global memory slots
    _dataChannelConsumerSizesBuffer                   = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataConsumerSizesBufferKey);
    _dataChannelConsumerPayloadBuffer                 = _config.payloadCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataConsumerPayloadBufferKey);
    _dataChannelConsumerCoordinationBufferForSizes    = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataConsumerCoordinationBufferForSizesKey);
    _dataChannelConsumerCoordinationBufferForPayloads = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataConsumerCoordinationBufferForPayloadKey);
    _dataChannelProducerCoordinationBufferForSizes    = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataProducerCoordinationBufferForSizesKey);
    _dataChannelProducerCoordinationBufferForPayloads = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.dataProducerCoordinationBufferForPayloadKey);
    // Metadata channel global memory slots
    _metadataChannelConsumerPayloadBuffer      = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.metadataConsumerPayloadBufferKey);
    _metadataChannelConsumerCoordinationBuffer = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.metadataConsumerCoordinationBufferKey);
    _metadataChannelProducerCoordinationBuffer = _config.coordinationCommunicationManager->getGlobalMemorySlot(tag, _slotKeys.metadataProducerCoordinationBufferKey);
    createChannels();

    // Set the channel as ready to be used
    _isReady = true;
  }

  __INLINE__ void lock() { _lock.lock(); }
  __INLINE__ void unlock() { _lock.unlock(); }

  __INLINE__ bool isReady() const { return _isReady; }

  protected:

  __INLINE__ void checkReady() const
  {
    if (_isReady == false) HICR_THROW_LOGIC("Channel '%s' is not initialized. Call initialize() before using it.", _config.name.c_str());
  }

  std::mutex _lock;

  virtual void createChannels() = 0;

  __INLINE__ void validateConfig() const
  {
    if (_config.payloadCommunicationManager == nullptr) HICR_THROW_LOGIC("Required HiCR object 'PayloadCommunicationManager' not provided for channel '%s'", _config.name.c_str());
    if (_config.payloadMemoryManager == nullptr) HICR_THROW_LOGIC("Required HiCR object 'PayloadMemoryManager' not provided for channel '%s'", _config.name.c_str());
    if (_config.payloadMemorySpace == nullptr) HICR_THROW_LOGIC("Required HiCR object 'PayloadMemorySpace' not provided for channel '%s'", _config.name.c_str());
    if (_config.coordinationCommunicationManager == nullptr)
      HICR_THROW_LOGIC("Required HiCR object 'CoordinationCommunicationManager' not provided for channel '%s'", _config.name.c_str());
    if (_config.coordinationMemoryManager == nullptr) HICR_THROW_LOGIC("Required HiCR object 'CoordinationMemoryManager' not provided for channel '%s'", _config.name.c_str());
    if (_config.coordinationMemorySpace == nullptr) HICR_THROW_LOGIC("Required HiCR object 'CoordinationMemorySpace' not provided for channel '%s'", _config.name.c_str());
  }

  static __INLINE__ channelConfig_t buildChannelConfig(const serving::configuration::Edge &edgeConfig)
  {
    return {.name                             = edgeConfig.getName(),
            .bufferCapacity                   = edgeConfig.getBufferCapacity(),
            .bufferSize                       = edgeConfig.getBufferSize(),
            .payloadCommunicationManager      = edgeConfig.getPayloadCommunicationManager(),
            .payloadMemoryManager             = edgeConfig.getPayloadMemoryManager(),
            .payloadMemorySpace               = edgeConfig.getPayloadMemorySpace(),
            .coordinationCommunicationManager = edgeConfig.getCoordinationCommunicationManager(),
            .coordinationMemoryManager        = edgeConfig.getCoordinationMemoryManager(),
            .coordinationMemorySpace          = edgeConfig.getCoordinationMemorySpace()};
  }

  /**
   * Helper function to build the slot keys for the global memory slots that need to be exchanged.
   */
  static __INLINE__ slotKeys_t buildSlotKeys(const channelId_t                  channelIndex,
                                             const HiCR::Instance::instanceId_t sourceIndex,
                                             const HiCR::Instance::instanceId_t targetIndex,
                                             const keyBuilderFc_t              &keyBuilder)
  {
    return keyBuilder(sourceIndex, targetIndex, channelIndex);
  }

  // Config + externally-provided key material
  const channelConfig_t _config;
  const slotKeys_t      _slotKeys;

  bool _isReady;

  // Local coordination buffers
  std::shared_ptr<HiCR::LocalMemorySlot> _dataChannelLocalCoordinationBufferForSizes;
  std::shared_ptr<HiCR::LocalMemorySlot> _dataChannelLocalCoordinationBufferForPayloads;
  std::shared_ptr<HiCR::LocalMemorySlot> _metadataChannelLocalCoordinationBuffer;

  // Data channel exchanged global slots
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelConsumerSizesBuffer;
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelConsumerPayloadBuffer;
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelConsumerCoordinationBufferForSizes;
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelConsumerCoordinationBufferForPayloads;
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelProducerCoordinationBufferForSizes;
  std::shared_ptr<HiCR::GlobalMemorySlot> _dataChannelProducerCoordinationBufferForPayloads;

  // Metadata channel exchanged global slots
  std::shared_ptr<HiCR::GlobalMemorySlot> _metadataChannelConsumerPayloadBuffer;
  std::shared_ptr<HiCR::GlobalMemorySlot> _metadataChannelConsumerCoordinationBuffer;
  std::shared_ptr<HiCR::GlobalMemorySlot> _metadataChannelProducerCoordinationBuffer;
};
} // namespace serving::system::channels