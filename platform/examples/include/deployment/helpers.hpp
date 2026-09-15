#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <nlohmann_json/json.hpp>

#include <hicr/core/instanceManager.hpp>

#include <modules/configuration/deployment.hpp>
#include <modules/configuration/replica.hpp>
#include <system/channels/base.hpp>
#include <system/channels/input.hpp>
#include <system/channels/output.hpp>

struct localInput_t
{
  HiCR::Instance::instanceId_t                      sourceInstanceId = -1;
  HiCR::Instance::instanceId_t                      targetInstanceId = -1;
  serving::system::channels::channelId_t            channelId        = -1;
  std::shared_ptr<serving::configuration::Edge>     edge             = {};
  std::shared_ptr<serving::system::channels::Input> channel          = nullptr; // optional: used by legacy examples
};

struct localOutput_t
{
  HiCR::Instance::instanceId_t                       sourceInstanceId = -1;
  HiCR::Instance::instanceId_t                       targetInstanceId = -1;
  serving::system::channels::channelId_t             channelId        = -1;
  std::shared_ptr<serving::configuration::Edge>      edge             = {};
  std::shared_ptr<serving::system::channels::Output> channel; // optional: used by legacy examples
};

__INLINE__ serving::system::channels::slotKeys_t defaultChannelKeyBuilder(const HiCR::Instance::instanceId_t           sourceInstanceId,
                                                                          const HiCR::Instance::instanceId_t           targetInstanceId,
                                                                          const serving::system::channels::channelId_t channelId)
{
  using key_t         = HiCR::GlobalMemorySlot::globalKey_t;
  const key_t src     = (key_t(sourceInstanceId) & ((1ull << 20) - 1)) << 44;
  const key_t dst     = (key_t(targetInstanceId) & ((1ull << 20) - 1)) << 24;
  const key_t ch      = (key_t(channelId) & ((1ull << 20) - 1)) << 4;
  auto        makeKey = [&](const key_t slot) -> key_t { return src | dst | ch | (slot & 0xFull); };
  return {
    .dataConsumerSizesBufferKey                  = makeKey(0),
    .dataConsumerPayloadBufferKey                = makeKey(1),
    .dataConsumerCoordinationBufferForSizesKey   = makeKey(2),
    .dataConsumerCoordinationBufferForPayloadKey = makeKey(3),
    .dataProducerCoordinationBufferForSizesKey   = makeKey(4),
    .dataProducerCoordinationBufferForPayloadKey = makeKey(5),
    .metadataConsumerPayloadBufferKey            = makeKey(6),
    .metadataConsumerCoordinationBufferKey       = makeKey(7),
    .metadataProducerCoordinationBufferKey       = makeKey(8),
  };
}

__INLINE__ void assignEdgeManagers(serving::configuration::Deployment       &deployment,
                                   HiCR::CommunicationManager               *communicationManager,
                                   HiCR::MemoryManager                      *memoryManager,
                                   const std::shared_ptr<HiCR::MemorySpace> &memorySpace)
{
  for (const auto &edge : deployment.getEdges())
  {
    edge->setPayloadCommunicationManager(communicationManager);
    edge->setPayloadMemoryManager(memoryManager);
    edge->setPayloadMemorySpace(memorySpace);
    edge->setCoordinationCommunicationManager(communicationManager);
    edge->setCoordinationMemoryManager(memoryManager);
    edge->setCoordinationMemorySpace(memorySpace);
  }
}

__INLINE__ void inferEdgeEndpointsFromTasks(serving::configuration::Deployment &deployment)
{
  std::set<std::string> edgeNameSet;

  for (const auto &edge : deployment.getEdges())
  {
    const auto &edgeName = edge->getName();
    if (edgeNameSet.contains(edgeName)) HICR_THROW_LOGIC("Repeated edge name '%s' in deployment.", edgeName.c_str());
    edgeNameSet.insert(edgeName);
  }

  std::map<std::string, std::string> producerPartitionMap;
  std::map<std::string, std::string> consumerPartitionMap;

  for (const auto &partition : deployment.getPartitions())
  {
    const auto &partitionName = partition->getName();

    for (const auto &task : partition->getTasks())
    {
      for (const auto &output : task->getOutputs())
      {
        if (edgeNameSet.contains(output) == false) HICR_THROW_LOGIC("Task '%s' references undefined output edge '%s'.", task->getFunctionName().c_str(), output.c_str());
        if (producerPartitionMap.contains(output))
          HICR_THROW_LOGIC("Edge '%s' has multiple producer partitions ('%s' and '%s').", output.c_str(), producerPartitionMap.at(output).c_str(), partitionName.c_str());
        producerPartitionMap[output] = partitionName;
      }
      for (const auto &input : task->getInputs())
      {
        if (edgeNameSet.contains(input) == false) HICR_THROW_LOGIC("Task '%s' references undefined input edge '%s'.", task->getFunctionName().c_str(), input.c_str());
        if (consumerPartitionMap.contains(input))
          HICR_THROW_LOGIC("Edge '%s' has multiple consumer partitions ('%s' and '%s').", input.c_str(), consumerPartitionMap.at(input).c_str(), partitionName.c_str());
        consumerPartitionMap[input] = partitionName;
      }
    }
  }

  for (const auto &edge : deployment.getEdges())
  {
    const auto &edgeName = edge->getName();

    if (producerPartitionMap.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is never produced by any task.", edgeName.c_str());
    if (consumerPartitionMap.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is never consumed by any task.", edgeName.c_str());

    const auto &producer = producerPartitionMap.at(edgeName);
    const auto &consumer = consumerPartitionMap.at(edgeName);

    if (producer == consumer) HICR_THROW_LOGIC("Edge '%s' is both produced and consumed by partition '%s'.", edgeName.c_str(), producer.c_str());

    edge->setProducer(producer);
    edge->setConsumer(consumer);
  }
}

__INLINE__ void buildLocalChannelsFromDeploymentWithIds(const serving::configuration::Deployment        &deployment,
                                                        const HiCR::Instance::instanceId_t               myInstanceId,
                                                        const serving::system::channels::keyBuilderFc_t &keyBuilder,
                                                        std::vector<localInput_t>                       &inputs,
                                                        std::vector<localOutput_t>                      &outputs)
{
  std::map<std::string, HiCR::Instance::instanceId_t> partitionToInstance;

  for (const auto &partition : deployment.getPartitions()) partitionToInstance[partition->getName()] = partition->getCoordinatorInstanceId();

  for (serving::configuration::Edge::edgeIndex_t edgeIdx = 0; edgeIdx < deployment.getEdges().size(); edgeIdx++)
  {
    const auto &edge              = deployment.getEdges()[edgeIdx];
    const auto &producerPartition = edge->getProducer();
    const auto &consumerPartition = edge->getConsumer();

    if (partitionToInstance.contains(producerPartition) == false)
      HICR_THROW_LOGIC("Edge '%s' producer partition '%s' is not present in deployment partition map.", edge->getName().c_str(), producerPartition.c_str());
    if (partitionToInstance.contains(consumerPartition) == false)
      HICR_THROW_LOGIC("Edge '%s' consumer partition '%s' is not present in deployment partition map.", edge->getName().c_str(), consumerPartition.c_str());

    const auto sourceInstanceId = partitionToInstance.at(producerPartition);
    const auto targetInstanceId = partitionToInstance.at(consumerPartition);
    const auto channelId        = static_cast<serving::system::channels::channelId_t>(edgeIdx);

    if (myInstanceId == sourceInstanceId)
    {
      outputs.push_back(localOutput_t{
        .sourceInstanceId = sourceInstanceId,
        .targetInstanceId = targetInstanceId,
        .channelId        = channelId,
        .edge             = edge,
        .channel          = std::make_shared<serving::system::channels::Output>(*edge, channelId, sourceInstanceId, targetInstanceId, keyBuilder),
      });
    }

    if (myInstanceId == targetInstanceId)
    {
      inputs.push_back(localInput_t{
        .sourceInstanceId = sourceInstanceId,
        .targetInstanceId = targetInstanceId,
        .channelId        = channelId,
        .edge             = edge,
        .channel          = std::make_shared<serving::system::channels::Input>(*edge, channelId, sourceInstanceId, targetInstanceId, keyBuilder),
      });
    }
  }
}

// Compatibility helper for legacy examples that still consume raw channel vectors.
__INLINE__ void buildLocalChannelsFromDeployment(const serving::configuration::Deployment                        &deployment,
                                                 const HiCR::Instance::instanceId_t                               myInstanceId,
                                                 const serving::system::channels::keyBuilderFc_t                 &keyBuilder,
                                                 std::vector<std::shared_ptr<serving::system::channels::Input>>  &inputs,
                                                 std::vector<std::shared_ptr<serving::system::channels::Output>> &outputs)
{
  std::vector<localInput_t>  localInputs;
  std::vector<localOutput_t> localOutputs;

  buildLocalChannelsFromDeploymentWithIds(deployment, myInstanceId, keyBuilder, localInputs, localOutputs);

  inputs.reserve(inputs.size() + localInputs.size());
  outputs.reserve(outputs.size() + localOutputs.size());

  for (const auto &in : localInputs) inputs.push_back(in.channel);
  for (const auto &out : localOutputs) outputs.push_back(out.channel);
}

__INLINE__ void readAndParseConfiguration(char                                   *argv[],
                                          serving::configuration::Deployment     &deployment,
                                          std::shared_ptr<HiCR::InstanceManager> &instanceManager,
                                          const size_t                            replicasPerPartition = 0)
{
  const std::string servingConfigFilePath = std::string(argv[1]);
  std::ifstream     servingConfigFs(servingConfigFilePath);
  auto              servingConfigJs = nlohmann::json::parse(servingConfigFs);

  deployment.deserialize(servingConfigJs);

  inferEdgeEndpointsFromTasks(deployment);

  // The coordinator is co-located with the first replica, so each partition needs
  // 1 instance for the coordinator + (replicasPerPartition - 1) extra replica instances.
  const auto extraReplicasPerPartition = replicasPerPartition > 0 ? replicasPerPartition - 1 : 0;
  const auto instancesRequired         = deployment.getPartitions().size() + (deployment.getPartitions().size() * extraReplicasPerPartition);

  if (instanceManager->getInstances().size() < instancesRequired)
  {
    fprintf(stderr, "Error: %zu instances provided, but %lu are required\n", instanceManager->getInstances().size(), instancesRequired);
    instanceManager->abort(-1);
  }

  auto instance = instanceManager->getInstances().begin();

  for (auto &partition : deployment.getPartitions())
  {
    const auto partitionInstanceId = (*instance)->getId();
    ++instance;
    partition->setCoordinatorInstanceId(partitionInstanceId);
    for (size_t i = 0; i < replicasPerPartition; i++)
    {
      // First replica is co-located with the coordinator; subsequent ones get their own instance.
      const auto replicaInstanceId = (i == 0) ? partitionInstanceId : (*instance)->getId();
      if (i > 0) ++instance;
      partition->addReplica(std::make_shared<serving::configuration::Replica>(replicaInstanceId));
    }
  }
}