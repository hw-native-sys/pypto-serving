#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <hicr/core/instance.hpp>

namespace serving::system
{

// Immutable versioned snapshot of a deployment's runtime state.
//
// Produced by coordinator::Module::getRuntimePlan().  Callers may hold a copy
// safely; the coordinator never mutates an already-returned plan.  Version is
// monotonically increasing: two plans with the same version describe the same
// state.
struct RuntimePlan
{
  enum class ReplicaStatus
  {
    Active,   // accepting new jobs
    Draining, // finishing current job; no new jobs will be assigned
    Removed   // fully idle; channels may be torn down by the caller
  };

  struct ReplicaState
  {
    HiCR::Instance::instanceId_t instanceId;
    ReplicaStatus                status;
  };

  struct PartitionState
  {
    std::string                  name;
    HiCR::Instance::instanceId_t coordinatorId;
    std::vector<ReplicaState>    replicas;

    [[nodiscard]] size_t activeCount() const noexcept
    {
      size_t n = 0;
      for (const auto &r : replicas)
        if (r.status == ReplicaStatus::Active) ++n;
      return n;
    }

    [[nodiscard]] size_t drainingCount() const noexcept
    {
      size_t n = 0;
      for (const auto &r : replicas)
        if (r.status == ReplicaStatus::Draining) ++n;
      return n;
    }

    [[nodiscard]] size_t removedCount() const noexcept
    {
      size_t n = 0;
      for (const auto &r : replicas)
        if (r.status == ReplicaStatus::Removed) ++n;
      return n;
    }
  };

  uint64_t                    version{0};
  std::vector<PartitionState> partitions;
};

} // namespace serving::system
