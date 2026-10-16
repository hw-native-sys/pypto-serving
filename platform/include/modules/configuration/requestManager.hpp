#pragma once

#include <cstdint>
#include <nlohmann_json/parser.hpp>
#include <hicr/core/definitions.hpp>
#include <hicr/core/instance.hpp>

namespace serving::configuration
{

class RequestManager final
{
  public:

  RequestManager(const nlohmann::json js) { deserialize(js); };
  RequestManager(const std::string &input, const std::string &output)
    : _input(input),
      _output(output)
  {}
  ~RequestManager() = default;

  __INLINE__ void setInput(const std::string &input) { _input = input; }
  __INLINE__ void setOutput(const std::string &output) { _output = output; }
  __INLINE__ void setInstanceId(const HiCR::Instance::instanceId_t instanceId) { _instanceId = instanceId; }

  [[nodiscard]] __INLINE__ auto getInput() const { return _input; }
  [[nodiscard]] __INLINE__ auto getOutput() const { return _output; }
  [[nodiscard]] __INLINE__ auto getInstanceId() const { return _instanceId; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Input"]       = _input;
    js["Output"]      = _output;
    js["Instance Id"] = _instanceId;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    _input  = hicr::json::getString(js, "Input");
    _output = hicr::json::getString(js, "Output");
    if (js.contains("Instance Id")) _instanceId = hicr::json::getNumber<HiCR::Instance::instanceId_t>(js, "Instance Id"); // Optional, as it is determined at runtime
  }

  private:

  std::string                  _input;
  std::string                  _output;
  HiCR::Instance::instanceId_t _instanceId;

}; // class RequestManager

} // namespace serving::configuration