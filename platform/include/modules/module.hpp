#pragma once

#include <hicr/core/definitions.hpp>
#include <taskr/service.hpp>

namespace serving::modules
{
class Module
{
  public:

  Module() = default;

  Module(const size_t intervalMs)
  {
    if (intervalMs == 0) HICR_THROW_LOGIC("[Coordinator] interval must be greater than zero.");
    _service = std::make_unique<taskr::Service>([&]() { service(); }, intervalMs);
  }

  virtual ~Module() = default;

  __INLINE__ bool hasService() const { return _service != nullptr; }
  __INLINE__ taskr::Service *getService() const { return _service.get(); }

  virtual void initialize() = 0;
  virtual void run()        = 0;
  virtual void terminate()  = 0;
  virtual void await()      = 0;
  virtual void finalize()   = 0;

  protected:

  std::unique_ptr<taskr::Service> _service = nullptr;

  virtual void service() = 0;
};

} // namespace serving::modules