#pragma once

#include <cstdint>
#include <nlohmann_json/parser.hpp>
#include <hicr/core/definitions.hpp>
#include <hicr/core/instance.hpp>

namespace serving::configuration
{

class Replica final
{
  public:

  typedef uint64_t replicaIndex_t;

  Replica(const nlohmann::json js) { deserialize(js); };
  Replica(const HiCR::Instance::instanceId_t instanceId)
    : _instanceId(instanceId)
  {}
  ~Replica() = default;

  __INLINE__ void setInstanceId(const HiCR::Instance::instanceId_t instanceId) { _instanceId = instanceId; }

  [[nodiscard]] __INLINE__ auto getInstanceId() const { return _instanceId; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Instance Id"] = _instanceId;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    if (js.contains("Instance Id")) _instanceId = hicr::json::getNumber<HiCR::Instance::instanceId_t>(js, "Instance Id"); // Optional, as it is determined at runtime
  }

  private:

  HiCR::Instance::instanceId_t _instanceId;

}; // class Replica

} // namespace serving::configuration