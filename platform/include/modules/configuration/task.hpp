#pragma once

#include <vector>
#include <string>
#include <nlohmann_json/parser.hpp>
#include <hicr/core/definitions.hpp>

namespace serving::configuration
{

class Task final
{
  public:

  Task(const nlohmann::json &js) { deserialize(js); };
  Task(const std::string &functionName)
    : _functionName(functionName)
  {}
  ~Task() = default;

  __INLINE__ void setFunctionName(const std::string &functionName) { _functionName = functionName; }
  __INLINE__ void addInput(const std::string &input) { _inputs.push_back(input); }
  __INLINE__ void addOutput(const std::string &output) { _outputs.push_back(output); }
  __INLINE__ void addDependency(const std::string &functionName) { _dependencies.push_back(functionName); }

  [[nodiscard]] __INLINE__ auto  getFunctionName() const { return _functionName; }
  [[nodiscard]] __INLINE__ auto &getInputs() const { return _inputs; }
  [[nodiscard]] __INLINE__ auto &getOutputs() const { return _outputs; }
  [[nodiscard]] __INLINE__ auto &getDependencies() const { return _dependencies; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Function Name"] = _functionName;
    js["Inputs"]        = _inputs;
    js["Outputs"]       = _outputs;
    js["Dependencies"]  = _dependencies;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    _functionName = hicr::json::getString(js, "Function Name");
    _inputs       = hicr::json::getArray<std::string>(js, "Inputs");
    _outputs      = hicr::json::getArray<std::string>(js, "Outputs");
    _dependencies = hicr::json::getArray<std::string>(js, "Dependencies");
  }

  private:

  std::string              _functionName;
  std::vector<std::string> _inputs;
  std::vector<std::string> _outputs;
  std::vector<std::string> _dependencies;

}; // class Task

} // namespace serving::configuration