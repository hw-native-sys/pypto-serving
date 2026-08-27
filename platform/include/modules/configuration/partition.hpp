#pragma once

#include <cstdint>
#include <memory>
#include "task.hpp"
#include "replica.hpp"
#include <nlohmann_json/parser.hpp>
#include <hicr/core/definitions.hpp>
#include <hicr/core/instance.hpp>

namespace serving::configuration
{

class Partition final
{
  public:

  typedef uint64_t partitionIndex_t;

  Partition(const nlohmann::json js) { deserialize(js); };
  Partition(const std::string &name, const HiCR::Instance::instanceId_t instanceId)
    : _name(name),
      _coordinatorInstanceId(instanceId)
  {}
  ~Partition() = default;

  __INLINE__ void setName(const std::string &name) { _name = name; }
  __INLINE__ void setCoordinatorInstanceId(const HiCR::Instance::instanceId_t instanceId) { _coordinatorInstanceId = instanceId; }
  __INLINE__ void addTask(const std::shared_ptr<Task> task) { _tasks.push_back(task); }
  __INLINE__ void addReplica(const std::shared_ptr<Replica> replica) { _replicas.push_back(replica); }

  [[nodiscard]] __INLINE__ auto  getName() const { return _name; }
  [[nodiscard]] __INLINE__ auto  getCoordinatorInstanceId() const { return _coordinatorInstanceId; }
  [[nodiscard]] __INLINE__ auto &getTasks() const { return _tasks; }
  [[nodiscard]] __INLINE__ auto &getReplicas() const { return _replicas; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Name"]                    = _name;
    js["Coordinator Instance Id"] = _coordinatorInstanceId;

    std::vector<nlohmann::json> tasksJs;
    for (const auto &t : _tasks) tasksJs.push_back(t->serialize());
    js["Tasks"] = tasksJs;

    std::vector<nlohmann::json> replicasJs;
    for (const auto &r : _replicas) replicasJs.push_back(r->serialize());
    js["Replicas"] = replicasJs;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    // Clearing objects
    _tasks.clear();
    _replicas.clear();

    _name = hicr::json::getString(js, "Name");
    if (js.contains("Coordinator Instance Id"))
      _coordinatorInstanceId = hicr::json::getNumber<HiCR::Instance::instanceId_t>(js, "Coordinator Instance Id"); // Optional, as it is determined at runtime

    const auto &tasks = hicr::json::getArray<nlohmann::json>(js, "Tasks");
    for (const auto &t : tasks) _tasks.push_back(std::make_shared<Task>(t));

    // This entry is optional, as it can be decided at runtime
    if (js.contains("Replicas"))
    {
      const auto &replicas = hicr::json::getArray<nlohmann::json>(js, "Replicas");
      for (const auto &r : replicas) _replicas.push_back(std::make_shared<Replica>(r));
    }
  }

  private:

  std::string                           _name;
  HiCR::Instance::instanceId_t          _coordinatorInstanceId = 0;
  std::vector<std::shared_ptr<Task>>    _tasks;
  std::vector<std::shared_ptr<Replica>> _replicas;
}; // class Partition

} // namespace serving::configuration