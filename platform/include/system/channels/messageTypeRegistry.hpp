#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>

#include "message.hpp"

namespace serving::system::channels
{

class MessageTypeRegistry
{
  public:

  using messageType_t = system::channels::Message::messageType_t;

  MessageTypeRegistry()  = default;
  ~MessageTypeRegistry() = default;

  __INLINE__ messageType_t registerType(const std::string &messageType)
  {
    if (messageType.empty()) HICR_THROW_LOGIC("Message type cannot be empty.");
    if (_nameToType.contains(messageType)) HICR_THROW_LOGIC("Message type '%s' is already registered.", messageType.c_str());

    auto type = fnv1a(messageType);

    if (_typeToName.contains(type))
      HICR_THROW_LOGIC("Message type hash collision: '%s' and '%s' map to id %lu.", messageType.c_str(), _typeToName.at(type).c_str(), static_cast<unsigned long>(type));

    _nameToType[messageType] = type;
    _typeToName[type]        = messageType;

    return type;
  }

  [[nodiscard]] __INLINE__ messageType_t getType(const std::string &messageType) const
  {
    if (_nameToType.contains(messageType) == false) HICR_THROW_LOGIC("Message type '%s' is not registered.", messageType.c_str());
    return _nameToType.at(messageType);
  }

  [[nodiscard]] __INLINE__ const std::string &getName(const messageType_t type) const
  {
    if (_typeToName.contains(type) == false) HICR_THROW_LOGIC("Message type id %lu is not registered.", static_cast<unsigned long>(type));
    return _typeToName.at(type);
  }

  private:

  __INLINE__ static messageType_t fnv1a(const std::string_view str)
  {
    constexpr uint64_t offsetBasis = 0xcbf29ce484222325ull;
    constexpr uint64_t prime       = 0x100000001b3ull;
    uint64_t           hash        = offsetBasis;
    for (const unsigned char c : str)
    {
      hash ^= static_cast<uint64_t>(c);
      hash *= prime;
    }
    return static_cast<messageType_t>(hash & Message::MAX_TYPE);
  }

  std::unordered_map<std::string, messageType_t> _nameToType;
  std::unordered_map<messageType_t, std::string> _typeToName;
};
} // namespace serving::system::channels