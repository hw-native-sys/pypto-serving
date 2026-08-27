#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/subscription.hpp>
#include <modules/module.hpp>
#include <modules/roles/coordinator/module.hpp>
#include <system/channels/input.hpp>
#include <system/channels/message.hpp>
#include <system/channels/output.hpp>

namespace serving::modules::router
{

class Module final : public serving::modules::Module
{
  public:

  using input_t       = std::shared_ptr<serving::system::channels::Input>;
  using output_t      = std::shared_ptr<serving::system::channels::Output>;
  using message_t     = serving::system::channels::Message;
  using metadata_t    = serving::system::channels::Message::metadata_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using instanceId_t  = HiCR::Instance::instanceId_t;
  using edgeName_t    = std::string;
  using jobOutput_t   = serving::modules::roles::coordinator::Module::JobOutput;
  using processFc_t   = std::function<void(const input_t &input, const edgeName_t &edgeName, const message_t &message)>;

  Module()
    : serving::modules::Module()
  {}

  __INLINE__ void addMessageType(const messageType_t messageType) { _messageTypes.insert(messageType); }

  __INLINE__ void addInput(const edgeName_t &edgeName, const input_t &input)
  {
    if (_inputs.contains(edgeName)) HICR_THROW_LOGIC("Graph input '%s' is already registered.", edgeName.c_str());
    _inputs[edgeName] = input;
  }

  __INLINE__ void addOutput(const edgeName_t &edgeName, const output_t &output)
  {
    if (_outputs.contains(edgeName)) HICR_THROW_LOGIC("Graph output '%s' is already registered.", edgeName.c_str());
    _outputs[edgeName] = output;
  }

  __INLINE__ void setProcessFunction(processFc_t processFc) { _processFc = processFc; }

  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> buildSubscriptions()
  {
    std::vector<serving::modules::Subscription> subscriptions;
    subscriptions.reserve(_inputs.size() * _messageTypes.size());
    for (const auto &[edgeName, input] : _inputs)
    {
      for (const auto messageType : _messageTypes)
      {
        subscriptions.emplace_back(messageType, input, [this, edgeName](const input_t input, const message_t &message) { _processFc(input, edgeName, message); });
      }
    }
    return subscriptions;
  }

  [[nodiscard]] __INLINE__ std::vector<std::pair<messageType_t, input_t>> buildUnsubscriptions() const
  {
    std::vector<std::pair<messageType_t, input_t>> unsubscriptions;
    unsubscriptions.reserve(_inputs.size() * _messageTypes.size());
    for (const auto &[_, input] : _inputs)
    {
      for (const auto messageType : _messageTypes) { unsubscriptions.push_back({messageType, input}); }
    }
    return unsubscriptions;
  }

  __INLINE__ void routeOutputs(const metadata_t &metadata, const std::vector<jobOutput_t> &outputs)
  {
    for (const auto &output : outputs)
    {
      if (_outputs.contains(output.name) == false) [[unlikely]] { HICR_THROW_LOGIC("No graph output channel registered for edge '%s'.", output.name.c_str()); }
      const auto &channel = _outputs.at(output.name);
      const auto  message =
        message_t(output.data == nullptr ? nullptr : static_cast<const uint8_t *>(output.data->getPointer()), output.data == nullptr ? 0 : output.data->getSize(), metadata);
      channel->pushMessageLocking(message);
    }
  }

  void initialize() override {}
  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override
  {
    _inputs.clear();
    _outputs.clear();
    _messageTypes.clear();
    _processFc = nullptr;
  }

  protected:

  void service() override {}

  private:

  std::unordered_map<edgeName_t, input_t>  _inputs;
  std::unordered_map<edgeName_t, output_t> _outputs;
  std::unordered_set<messageType_t>        _messageTypes;
  processFc_t                              _processFc;
};
} // namespace serving::modules::router
