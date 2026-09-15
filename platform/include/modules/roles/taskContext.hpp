#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>
#include <hicr/core/localMemorySlot.hpp>

#include <system/channels/base.hpp>
#include <system/channels/message.hpp>

namespace serving::modules::roles
{

class TaskContext final
{
  public:

  using metadata_t        = serving::system::channels::Message::metadata_t;
  using memorySlot_t      = std::shared_ptr<HiCR::LocalMemorySlot>;
  using outputConfigMap_t = std::unordered_map<std::string, serving::system::channels::channelConfig_t>;

  struct Output
  {
    std::string  name;
    memorySlot_t data;
    bool         owned = false;
  };

  TaskContext(const std::string &name, const metadata_t &metadata, const std::unordered_map<std::string, memorySlot_t> &inputs, const outputConfigMap_t &outputConfigs)
    : _name(name),
      _metadata(metadata),
      _inputs(inputs),
      _outputConfigs(outputConfigs)
  {}

  [[nodiscard]] __INLINE__ const std::string &getName() const { return _name; }
  [[nodiscard]] __INLINE__ const metadata_t  &getMetadata() const { return _metadata; }

  [[nodiscard]] __INLINE__ memorySlot_t getInput(const std::string &name) const
  {
    if (_inputs.contains(name) == false) HICR_THROW_RUNTIME("Task '%s' has no input '%s'.", _name.c_str(), name.c_str());
    const auto &input = _inputs.at(name);
    if (input == nullptr) return _emptySlot;
    return input;
  }

  __INLINE__ void setOutput(const std::string &name, const memorySlot_t &data)
  {
    if (_outputIndex.contains(name)) HICR_THROW_RUNTIME("Task '%s' set output '%s' more than once.", _name.c_str(), name.c_str());
    if (_outputConfigs.contains(name) == false) HICR_THROW_RUNTIME("Task '%s' set undeclared output '%s'.", _name.c_str(), name.c_str());
    if (data == nullptr) HICR_THROW_RUNTIME("Task '%s' cannot set output '%s' from a null memory slot.", _name.c_str(), name.c_str());

    _outputIndex[name] = _outputs.size();
    _outputs.push_back(Output{.name = name, .data = data, .owned = false});
  }

  __INLINE__ void setOutput(const std::string &name, const void *data, const size_t size)
  {
    if (_outputIndex.contains(name)) HICR_THROW_RUNTIME("Task '%s' set output '%s' more than once.", _name.c_str(), name.c_str());
    if (data == nullptr && size > 0) HICR_THROW_LOGIC("Task '%s' cannot set output '%s' from null data with non-zero size.", _name.c_str(), name.c_str());
    if (_outputConfigs.contains(name) == false) HICR_THROW_RUNTIME("Task '%s' set undeclared output '%s'.", _name.c_str(), name.c_str());

    memorySlot_t outputSlot = nullptr;
    if (size > 0)
    {
      const auto &cfg     = _outputConfigs.at(name);
      const auto  srcSlot = cfg.payloadMemoryManager->registerLocalMemorySlot(cfg.payloadMemorySpace, const_cast<void *>(data), size);
      outputSlot          = cfg.payloadMemoryManager->allocateLocalMemorySlot(cfg.payloadMemorySpace, size);
      cfg.payloadCommunicationManager->memcpy(outputSlot, 0, srcSlot, 0, size);
      cfg.payloadCommunicationManager->fence(outputSlot, 0, 1);
      cfg.payloadMemoryManager->deregisterLocalMemorySlot(srcSlot);
    }

    _outputIndex[name] = _outputs.size();
    _outputs.push_back(Output{.name = name, .data = outputSlot, .owned = true});
  }

  [[nodiscard]] __INLINE__ const std::vector<Output> &getOutputs() const { return _outputs; }

  __INLINE__ void freeOwnedOutputs()
  {
    for (auto &output : _outputs)
    {
      if (output.owned == false || output.data == nullptr) continue;
      _outputConfigs.at(output.name).payloadMemoryManager->freeLocalMemorySlot(output.data);
      output.data = nullptr;
    }
  }

  private:

  inline static const memorySlot_t                     _emptySlot = std::make_shared<HiCR::LocalMemorySlot>(nullptr, 0);
  const std::string                                    _name;
  const metadata_t                                     _metadata;
  const std::unordered_map<std::string, memorySlot_t> &_inputs;
  const outputConfigMap_t                             &_outputConfigs;
  std::vector<Output>                                  _outputs;
  std::unordered_map<std::string, size_t>              _outputIndex;
};

} // namespace serving::modules::roles
