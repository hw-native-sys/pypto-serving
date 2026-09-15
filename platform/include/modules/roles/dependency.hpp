#pragma once

#include <memory>

#include <hicr/core/definitions.hpp>
#include <hicr/core/localMemorySlot.hpp>

#include <modules/configuration/edge.hpp>

namespace serving::modules::roles
{

class Dependency
{
  public:

  Dependency()  = delete;
  ~Dependency() = default;

  Dependency(const std::string &name, const configuration::Edge &edgeInfo)
    : _name(name),
      _edgeInfo(edgeInfo)
  {}

  [[nodiscard]] __INLINE__ bool isSatisfied() const { return _isSatisfied; }
  __INLINE__ void               setSatisfied(const bool satisfied = true) { _isSatisfied = satisfied; }

  [[nodiscard]] __INLINE__ const std::shared_ptr<HiCR::LocalMemorySlot> getData() const { return _data; }
  [[nodiscard]] __INLINE__ bool                                         hasData() const { return _data != nullptr; }
  [[nodiscard]] __INLINE__ size_t                                       getDataSize() const { return hasData() ? _data->getSize() : 0; }
  [[nodiscard]] __INLINE__ const uint8_t                               *getDataPointer() const { return hasData() ? static_cast<const uint8_t *>(_data->getPointer()) : nullptr; }

  __INLINE__ void freeDataSlot()
  {
    if (_data == nullptr) return;
    _edgeInfo.getPayloadMemoryManager()->freeLocalMemorySlot(_data);
    _data = nullptr;
  }

  [[nodiscard]] __INLINE__ const std::string &getName() const { return _name; }

  __INLINE__ void storeData(const uint8_t *srcPtr, const size_t size)
  {
    if (srcPtr == nullptr && size > 0) HICR_THROW_LOGIC("Dependency '%s' cannot store null data with non-zero size.", _name.c_str());

    freeDataSlot();
    if (size == 0) return;

    auto edgeMemoryManager        = _edgeInfo.getPayloadMemoryManager();
    auto edgeMemorySpace          = _edgeInfo.getPayloadMemorySpace();
    auto edgeCommunicationManager = _edgeInfo.getPayloadCommunicationManager();

    const auto srcSlot = edgeMemoryManager->registerLocalMemorySlot(edgeMemorySpace, (void *)srcPtr, size);
    auto       dstSlot = edgeMemoryManager->allocateLocalMemorySlot(edgeMemorySpace, size);
    edgeCommunicationManager->memcpy(dstSlot, 0, srcSlot, 0, size);
    edgeCommunicationManager->fence(dstSlot, 0, 1);
    edgeMemoryManager->deregisterLocalMemorySlot(srcSlot);

    _data = dstSlot;
  }

  private:

  const std::string                      _name;
  const configuration::Edge              _edgeInfo;
  std::shared_ptr<HiCR::LocalMemorySlot> _data        = nullptr;
  bool                                   _isSatisfied = false;
};
} // namespace serving::modules::roles
