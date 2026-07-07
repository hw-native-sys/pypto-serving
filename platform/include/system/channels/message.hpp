#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <hicr/core/definitions.hpp>
#include <hicr/core/memoryManager.hpp>
#include <hicr/core/memorySpace.hpp>

namespace serving::system::channels
{

class Message final
{
  public:

  // 16 bits for type (65536)
  using messageType_t = uint16_t;

  // 14 bits for group (16384)
  using groupId_t = uint16_t;
  // 34 bits for sequence (17B)
  using sequenceId_t = uint64_t;

  // Id type
  using messageId_t = uint64_t;

  static constexpr messageId_t TYPE_BITS     = 16; // 16 bits for type (65536)
  static constexpr messageId_t GROUP_BITS    = 14; // 14 bits for group (16384)
  static constexpr messageId_t SEQUENCE_BITS = 34; // 34 bits for sequence (17B)

  static constexpr messageId_t MAX_TYPE     = (messageId_t{1} << TYPE_BITS) - 1;
  static constexpr messageId_t MAX_GROUP    = (messageId_t{1} << GROUP_BITS) - 1;
  static constexpr messageId_t MAX_SEQUENCE = (messageId_t{1} << SEQUENCE_BITS) - 1;

#pragma pack(push, 1)
  struct metadata_t
  {
    messageType_t type       = 0;
    groupId_t     groupId    = 0;
    sequenceId_t  sequenceId = 0;

    bool isValid() const
    {
      return static_cast<messageId_t>(type) <= MAX_TYPE && static_cast<messageId_t>(groupId) <= MAX_GROUP && static_cast<messageId_t>(sequenceId) <= MAX_SEQUENCE;
    }

    messageId_t getId() const
    {
      assert(isValid());
      return ((static_cast<messageId_t>(type) & MAX_TYPE) << (GROUP_BITS + SEQUENCE_BITS)) | ((static_cast<messageId_t>(groupId) & MAX_GROUP) << SEQUENCE_BITS) |
             (static_cast<messageId_t>(sequenceId) & MAX_SEQUENCE);
    }
  };
#pragma pack(pop)

  Message() = delete;

  Message(const uint8_t *const data, const size_t size, const metadata_t metadata)
    : _data(data),
      _size(size),
      _metadata(metadata)
  {}

  ~Message() = default;

  [[nodiscard]] __INLINE__ const uint8_t    *getData() const { return _data; }
  [[nodiscard]] __INLINE__ size_t            getSize() const { return _size; }
  [[nodiscard]] __INLINE__ const metadata_t &getMetadata() const { return _metadata; }

  private:

  const uint8_t *const _data;
  const size_t         _size;
  const metadata_t     _metadata;
}; // class Message
} // namespace serving::system::channels