#pragma once

#include <cstdint>
#include <string>
#include <hicr/core/definitions.hpp>
#include <hicr/core/communicationManager.hpp>
#include <hicr/core/memoryManager.hpp>
#include <hicr/core/memorySpace.hpp>
#include <nlohmann_json/parser.hpp>

namespace serving::configuration
{

class Edge
{
  public:

#define __SERVING_PARTITION_DEFAULT_BUFFER_CAPACITY 1

  typedef uint64_t edgeIndex_t;

  Edge(const nlohmann::json &js) { deserialize(js); }
  Edge(const std::string &name, const size_t bufferCapacity, const size_t bufferSize = 0)
    : _name(name),
      _bufferCapacity(bufferCapacity),
      _bufferSize(bufferSize)
  {}
  virtual ~Edge() = default;

  __INLINE__ void          setName(const std::string &name) { _name = name; }
  [[nodiscard]] __INLINE__ std::string getName() const { return _name; }
  [[nodiscard]] __INLINE__ std::string getProducer() const { return _producer; }
  [[nodiscard]] __INLINE__ std::string getConsumer() const { return _consumer; }
  [[nodiscard]] __INLINE__ size_t      getBufferCapacity() const { return _bufferCapacity; }
  [[nodiscard]] __INLINE__ size_t      getBufferSize() const { return _bufferSize; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Name"]            = _name;
    js["Producer"]        = _producer;
    js["Consumer"]        = _consumer;
    js["Buffer Capacity"] = _bufferCapacity;
    js["Buffer Size"]     = _bufferSize;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    _name = hicr::json::getString(js, "Name");
    if (js.contains("Producer")) _producer = hicr::json::getString(js, "Producer");
    if (js.contains("Consumer")) _consumer = hicr::json::getString(js, "Consumer");
    if (js.contains("Buffer Capacity")) _bufferCapacity = hicr::json::getNumber<size_t>(js, "Buffer Capacity");
    _bufferSize = hicr::json::getNumber<size_t>(js, "Buffer Size");
  }

  // Functions to set the HiCR elements required for the creation of edge channels
  __INLINE__ void setPayloadCommunicationManager(HiCR::CommunicationManager *const communicationManager) { _payloadCommunicationManager = communicationManager; }
  __INLINE__ void setPayloadMemoryManager(HiCR::MemoryManager *const memoryManager) { _payloadMemoryManager = memoryManager; }
  __INLINE__ void setPayloadMemorySpace(const std::shared_ptr<HiCR::MemorySpace> memorySpace) { _payloadMemorySpace = memorySpace; }
  __INLINE__ void setCoordinationCommunicationManager(HiCR::CommunicationManager *const communicationManager) { _coordinationCommunicationManager = communicationManager; }
  __INLINE__ void setCoordinationMemoryManager(HiCR::MemoryManager *const memoryManager) { _coordinationMemoryManager = memoryManager; }
  __INLINE__ void setCoordinationMemorySpace(const std::shared_ptr<HiCR::MemorySpace> memorySpace) { _coordinationMemorySpace = memorySpace; }

  __INLINE__ void setProducer(const std::string &partition) { _producer = partition; }
  __INLINE__ void setConsumer(const std::string &partition) { _consumer = partition; }
  __INLINE__ void setPromptEdge(const bool isPromptEdge) { _isPromptEdge = isPromptEdge; }
  __INLINE__ void setResultEdge(const bool isResultEdge) { _isResultEdge = isResultEdge; }
  __INLINE__ void setBufferCapacity(const size_t bufferCapacity) { _bufferCapacity = bufferCapacity; }
  __INLINE__ void setBufferSize(const size_t bufferSize) { _bufferSize = bufferSize; }

  // Functions to set the HiCR elements required for the creation of edge channels
  [[nodiscard]] __INLINE__ HiCR::CommunicationManager *getPayloadCommunicationManager() const { return _payloadCommunicationManager; }
  [[nodiscard]] __INLINE__ HiCR::MemoryManager *getPayloadMemoryManager() const { return _payloadMemoryManager; }
  [[nodiscard]] __INLINE__ std::shared_ptr<HiCR::MemorySpace> getPayloadMemorySpace() const { return _payloadMemorySpace; }
  [[nodiscard]] __INLINE__ HiCR::CommunicationManager *getCoordinationCommunicationManager() const { return _coordinationCommunicationManager; }
  [[nodiscard]] __INLINE__ HiCR::MemoryManager *getCoordinationMemoryManager() const { return _coordinationMemoryManager; }
  [[nodiscard]] __INLINE__ std::shared_ptr<HiCR::MemorySpace> getCoordinationMemorySpace() const { return _coordinationMemorySpace; }
  [[nodiscard]] __INLINE__ bool                               isPromptEdge() const { return _isPromptEdge; }
  [[nodiscard]] __INLINE__ bool                               isResultEdge() const { return _isResultEdge; }

  private:

  std::string _name;
  std::string _producer;
  std::string _consumer;
  size_t      _bufferCapacity = __SERVING_PARTITION_DEFAULT_BUFFER_CAPACITY;
  size_t      _bufferSize;

  // This flag serves to indicate whether these are edges that connect to the request manager
  bool _isPromptEdge = false;
  bool _isResultEdge = false;

  // HiCR-specific objects to create the payload buffers. These are to be set at runtime
  HiCR::CommunicationManager        *_payloadCommunicationManager = nullptr;
  HiCR::MemoryManager               *_payloadMemoryManager        = nullptr;
  std::shared_ptr<HiCR::MemorySpace> _payloadMemorySpace          = nullptr;

  // HiCR-specific objects to create the coordination buffers. These are to be set at runtime
  HiCR::CommunicationManager        *_coordinationCommunicationManager = nullptr;
  HiCR::MemoryManager               *_coordinationMemoryManager        = nullptr;
  std::shared_ptr<HiCR::MemorySpace> _coordinationMemorySpace          = nullptr;

}; // class Base

} // namespace serving::configuration