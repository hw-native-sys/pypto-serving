#pragma once

#include <unordered_map>
#include <hicr/core/definitions.hpp>

#include <modules/configuration/deployment.hpp>
#include <modules/configuration/edge.hpp>

#include "./dependency.hpp"
#include "./job.hpp"

namespace serving::modules::roles
{

class JobFactory
{
  public:

  using jobName_t    = std::string;
  using dependency_t = std::string;

  JobFactory(const size_t partitionIndex, const configuration::Deployment &deployment)
    : _jobInputDependencies(buildTaskInputDependencies(*deployment.getPartitions().at(partitionIndex))),
      _jobOutputDependencies(buildTaskOutputDependencies(*deployment.getPartitions().at(partitionIndex))),
      _edgeInfos(buildEdgeInfos(deployment))
  {}

  [[nodiscard]] __INLINE__ Job createJob(const std::string &jobName, const system::channels::Message::metadata_t &metadata)
  {
    auto inputDependencies = std::unordered_map<dependency_t, Dependency>();
    for (const auto &[job, edges] : _jobInputDependencies)
    {
      if (job != jobName) { continue; }
      for (auto &edge : edges)
      {
        const auto &edgeInfo = _edgeInfos.at(edge);
        inputDependencies.try_emplace(edge, edge, *edgeInfo);
      }
    }

    auto outputDependencies = std::unordered_map<dependency_t, Dependency>();
    for (const auto &[job, edges] : _jobOutputDependencies)
    {
      if (job != jobName) { continue; }
      for (auto &edge : edges)
      {
        const auto &edgeInfo = _edgeInfos.at(edge);
        outputDependencies.try_emplace(edge, edge, *edgeInfo);
      }
    }

    return Job(jobName, metadata, inputDependencies, outputDependencies);
  }

  private:

  std::unordered_map<dependency_t, std::shared_ptr<configuration::Edge>> buildEdgeInfos(const configuration::Deployment &deployment)
  {
    auto edgeInfos = std::unordered_map<dependency_t, std::shared_ptr<configuration::Edge>>();
    for (const auto &edge : deployment.getEdges()) { edgeInfos[edge->getName()] = edge; }
    return edgeInfos;
  }

  std::unordered_map<jobName_t, std::vector<dependency_t>> buildTaskInputDependencies(const configuration::Partition &partition)
  {
    auto taskInfos = std::unordered_map<jobName_t, std::vector<dependency_t>>();
    for (const auto &task : partition.getTasks()) { taskInfos[task->getFunctionName()] = task->getInputs(); }
    return taskInfos;
  }

  std::unordered_map<jobName_t, std::vector<dependency_t>> buildTaskOutputDependencies(const configuration::Partition &partition)
  {
    auto taskInfos = std::unordered_map<jobName_t, std::vector<dependency_t>>();
    for (const auto &task : partition.getTasks()) { taskInfos[task->getFunctionName()] = task->getOutputs(); }
    return taskInfos;
  }

  const std::unordered_map<jobName_t, std::vector<dependency_t>>               _jobInputDependencies;
  const std::unordered_map<jobName_t, std::vector<dependency_t>>               _jobOutputDependencies;
  const std::unordered_map<dependency_t, std::shared_ptr<configuration::Edge>> _edgeInfos;
};
} // namespace serving::modules::roles
