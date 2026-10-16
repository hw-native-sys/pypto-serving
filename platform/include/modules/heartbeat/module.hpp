#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/module.hpp>
#include <modules/subscription.hpp>
#include <system/channels/input.hpp>
#include <system/channels/message.hpp>
#include <system/channels/messageTypeRegistry.hpp>
#include <system/channels/output.hpp>

namespace serving::modules::heartbeat
{

class Module final : public serving::modules::Module
{
  public:

  using instanceId_t  = HiCR::Instance::instanceId_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using input_t       = std::shared_ptr<serving::system::channels::Input>;
  using output_t      = std::shared_ptr<serving::system::channels::Output>;
  using message_t     = serving::system::channels::Message;

  enum class health_t : uint8_t
  {
    unknown   = 0,
    healthy   = 1,
    unhealthy = 2
  };

  inline static std::string health_tToString(health_t state)
  {
    switch (state)
    {
    case health_t::healthy: return "Healthy";
    case health_t::unhealthy: return "Unhealthy";
    case health_t::unknown: return "Unknown";
    }

    HICR_THROW_LOGIC("Invalid health state %u.", static_cast<unsigned>(state));
  }

  struct healthEvent_t
  {
    instanceId_t                          instanceId;
    health_t                              previousHealth;
    health_t                              newHealth;
    std::chrono::steady_clock::time_point timestamp;
  };

  using healthChangeCallback_t = std::function<void(const healthEvent_t &)>;

  Module(const instanceId_t                     instanceId,
         const size_t                           toleranceMs,
         const healthChangeCallback_t          &healthChangeCallback,
         system::channels::MessageTypeRegistry &messageTypeRegistry,
         const size_t                           intervalMs)
    : serving::modules::Module(intervalMs),
      _instanceId(instanceId),
      _toleranceMs(toleranceMs),
      _healthChangeCallback(healthChangeCallback),
      _messageType(messageTypeRegistry.registerType("modules.heartbeat.heartbeat"))
  {
    if (_toleranceMs < intervalMs) HICR_THROW_LOGIC("[Heartbeat] tolerance must be >= interval.");
  }

  ~Module() override = default;

  __INLINE__ void addInput(const instanceId_t instanceId, const input_t &input)
  {
    if (_inputs.contains(instanceId)) HICR_THROW_LOGIC("[Heartbeat] input already exists for instance %lu.", instanceId);
    _inputs[instanceId]   = input;
    _health[instanceId]   = health_t::unknown;
    _lastSeen[instanceId] = std::chrono::steady_clock::time_point::min();
  }

  __INLINE__ void removeInput(const instanceId_t instanceId)
  {
    _inputs.erase(instanceId);
    _health.erase(instanceId);
    _lastSeen.erase(instanceId);
  }

  __INLINE__ void addOutput(const instanceId_t instanceId, const output_t &output)
  {
    if (_outputs.contains(instanceId)) HICR_THROW_LOGIC("[Heartbeat] output already exists for instance %lu.", instanceId);
    _outputs[instanceId] = output;
  }

  __INLINE__ void removeOutput(const instanceId_t instanceId) { _outputs.erase(instanceId); }

  [[nodiscard]] __INLINE__ health_t getHealth(const instanceId_t instanceId) const
  {
    if (_health.contains(instanceId) == false) return health_t::unknown;
    return _health.at(instanceId);
  }

  [[nodiscard]] __INLINE__ const std::unordered_map<instanceId_t, health_t> &getHealthSnapshot() const { return _health; }

  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> buildSubscriptions()
  {
    std::vector<serving::modules::Subscription> out;
    out.reserve(_inputs.size());
    for (const auto &[instanceId, input] : _inputs)
    {
      out.emplace_back(_messageType, input, [this, instanceId](const input_t, const message_t &message) { this->heartbeatMessageHandler(instanceId, message); });
    }
    return out;
  }

  [[nodiscard]] __INLINE__ std::vector<std::pair<messageType_t, input_t>> buildUnsubscriptions() const
  {
    std::vector<std::pair<messageType_t, input_t>> out;
    out.reserve(_inputs.size());
    for (const auto &[_, input] : _inputs) out.push_back({_messageType, input});
    return out;
  }

  void initialize() override {}

  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override
  {
    _inputs.clear();
    _outputs.clear();
    _health.clear();
    _lastSeen.clear();
  }

  protected:

  void service() override
  {
    checkTimeouts();
    sendHeartbeats();
  }

  private:

  __INLINE__ void emitHealthEvent(const instanceId_t instanceId, const health_t previousHealth, const health_t newHealth, const std::chrono::steady_clock::time_point now)
  {
    _healthChangeCallback(healthEvent_t{.instanceId = instanceId, .previousHealth = previousHealth, .newHealth = newHealth, .timestamp = now});
  }

  __INLINE__ void heartbeatMessageHandler(const instanceId_t instanceId, const message_t &message)
  {
    if (message.getMetadata().type != _messageType) HICR_THROW_RUNTIME("[Heartbeat] unexpected message type %lu.", message.getMetadata().type);
    const auto now        = std::chrono::steady_clock::now();
    const auto previous   = _health[instanceId];
    _lastSeen[instanceId] = now;
    _health[instanceId]   = health_t::healthy;
    emitHealthEvent(instanceId, previous, _health[instanceId], now);
  }

  __INLINE__ void checkTimeouts()
  {
    const auto now     = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::milliseconds(_toleranceMs);
    for (const auto &[instanceId, _] : _inputs)
    {
      if (_lastSeen[instanceId] == std::chrono::steady_clock::time_point::min()) continue;
      if (now - _lastSeen[instanceId] > timeout)
      {
        const auto previous = _health[instanceId];
        _health[instanceId] = health_t::unhealthy;
        emitHealthEvent(instanceId, previous, _health[instanceId], now);
      }
    }
  }

  __INLINE__ void sendHeartbeats()
  {
    const uint8_t payload = 0;
    for (const auto &[_, output] : _outputs)
    {
      serving::system::channels::Message::metadata_t metadata;
      metadata.type       = _messageType;
      metadata.groupId    = static_cast<serving::system::channels::Message::groupId_t>(_instanceId);
      metadata.sequenceId = 0;
      const message_t heartbeatMessage(&payload, sizeof(payload), metadata);
      output->pushMessageLocking(heartbeatMessage);
    }
  }

  const instanceId_t _instanceId;
  const size_t       _toleranceMs;

  const healthChangeCallback_t _healthChangeCallback;

  const system::channels::MessageTypeRegistry::messageType_t _messageType;

  std::unordered_map<instanceId_t, input_t>                               _inputs;
  std::unordered_map<instanceId_t, output_t>                              _outputs;
  std::unordered_map<instanceId_t, health_t>                              _health;
  std::unordered_map<instanceId_t, std::chrono::steady_clock::time_point> _lastSeen;
};
} // namespace serving::modules::heartbeat
