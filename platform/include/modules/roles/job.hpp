#pragma once

#include <string>
#include <vector>
#include <nlohmann_json/json.hpp>

#include <hicr/core/definitions.hpp>

#include <modules/configuration/deployment.hpp>
#include <system/channels/message.hpp>

#include "./dependency.hpp"

namespace serving::modules::roles
{

using dependency_t = std::string;

class Job
{
  public:

  Job(const std::string                            &name,
      const system::channels::Message::metadata_t  &messageMetadata,
      std::unordered_map<dependency_t, Dependency> &inputDependencies,
      std::unordered_map<dependency_t, Dependency> &outputDependencies)
    : _name(name),
      _messageMetadata(messageMetadata),
      _inputDependencies(inputDependencies),
      _outputDependencies(outputDependencies)
  {}

  ~Job() = default;

  [[nodiscard]] __INLINE__ Dependency &getInputDependency(const dependency_t &dependencyName) { return _inputDependencies.at(dependencyName); }
  [[nodiscard]] __INLINE__ Dependency &getOutputDependency(const dependency_t &dependencyName) { return _outputDependencies.at(dependencyName); }
  [[nodiscard]] __INLINE__ std::unordered_map<dependency_t, Dependency> &getInputDependencies() { return _inputDependencies; }
  [[nodiscard]] __INLINE__ std::unordered_map<dependency_t, Dependency> &getOutputDependencies() { return _outputDependencies; }
  [[nodiscard]] __INLINE__ const std::unordered_map<dependency_t, Dependency> &getInputDependencies() const { return _inputDependencies; }
  [[nodiscard]] __INLINE__ const std::unordered_map<dependency_t, Dependency> &getOutputDependencies() const { return _outputDependencies; }
  [[nodiscard]] __INLINE__ const system::channels::Message::metadata_t &getMetadata() const { return _messageMetadata; }

  [[nodiscard]] __INLINE__ bool isReady() const
  {
    for (const auto &[_, dependency] : _inputDependencies)
    {
      if (dependency.isSatisfied() == false) return false;
    }
    return true;
  }

  [[nodiscard]] __INLINE__ bool isFinished() const
  {
    for (const auto &[_, dependency] : _outputDependencies)
    {
      if (dependency.isSatisfied() == false) return false;
    }
    return true;
  }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;
    js["Name"]     = _name;
    js["Ready"]    = isReady();
    js["Finished"] = isFinished();

    js["Message metadata"]             = nlohmann::json::object();
    js["Message metadata"]["Type"]     = _messageMetadata.type;
    js["Message metadata"]["Group"]    = _messageMetadata.groupId;
    js["Message metadata"]["Sequence"] = _messageMetadata.sequenceId;
    js["Message metadata"]["ID"]       = _messageMetadata.getId();

    js["Input Dependencies"] = nlohmann::json::array();
    for (const auto &[dependencyName, dependency] : _inputDependencies)
    {
      nlohmann::json depJs;
      depJs["Name"]            = dependency.getName();
      depJs["Dependency Name"] = dependencyName;
      depJs["Satisfied"]       = dependency.isSatisfied();
      depJs["Data Size"]       = (dependency.getData() == nullptr) ? 0 : dependency.getData()->getSize();
      js["Input Dependencies"].push_back(depJs);
    }
    js["Output Dependencies"] = nlohmann::json::array();
    for (const auto &[dependencyName, dependency] : _outputDependencies)
    {
      nlohmann::json depJs;
      depJs["Name"]            = dependency.getName();
      depJs["Dependency Name"] = dependencyName;
      depJs["Satisfied"]       = dependency.isSatisfied();
      depJs["Data Size"]       = (dependency.getData() == nullptr) ? 0 : dependency.getData()->getSize();
      js["Output Dependencies"].push_back(depJs);
    }
    return js;
  }

  protected:

  const std::string                            _name;
  const system::channels::Message::metadata_t  _messageMetadata;
  std::unordered_map<dependency_t, Dependency> _inputDependencies;
  std::unordered_map<dependency_t, Dependency> _outputDependencies;
};
} // namespace serving::modules::roles
