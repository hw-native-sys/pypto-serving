#include <stdio.h>
#include <thread>
#include <fstream>
#include <chrono>
#include <optional>
#include <vector>

#include <modules/channelController/module.hpp>
#include <modules/channelDispatcher/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/service/module.hpp>
#include <system/channels/message.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>
#include <runtime/helpers.hpp>

#include "coordinator.hpp"
#include "replica.hpp"

constexpr size_t replicasPerPartition = 1;

constexpr serving::system::channels::Message::messageType_t kMessageType = 100;

int main(int argc, char *argv[])
{
  auto        runtime    = makeRuntime(&argc, &argv);
  const auto &instance   = runtime.instanceManager->getCurrentInstance();
  const auto  instanceId = instance->getId();
  const auto  isRoot     = instance->isRootInstance();
  auto       &serving    = *runtime.serving;

  if (argc != 2)
  {
    fprintf(stderr, "Error: Must provide config file path.\n");
    runtime.instanceManager->abort(-1);
    return -1;
  }

  serving::configuration::Deployment deployment;
  readAndParseConfiguration(argv, deployment, runtime.instanceManager, replicasPerPartition);
  assignEdgeManagers(deployment, runtime.communicationManager.get(), runtime.memoryManager.get(), runtime.bufferMemorySpace);

  // Determine roles from the deployment config.
  // In the co-located topology, the same instance can be BOTH coordinator and replica.
  bool                      isCoordinator             = false;
  bool                      isReplica                 = false;
  size_t                    coordinatorPartitionIndex = 0;
  size_t                    replicaPartitionIndex     = 0;
  instanceId_t              myCoordinatorId           = 0;
  std::vector<instanceId_t> myReplicaIds;

  for (size_t i = 0; i < deployment.getPartitions().size(); i++)
  {
    const auto &partition = deployment.getPartitions()[i];
    if (partition->getCoordinatorInstanceId() == instanceId)
    {
      isCoordinator             = true;
      coordinatorPartitionIndex = i;
      for (const auto &r : partition->getReplicas()) myReplicaIds.push_back(r->getInstanceId());
    }
    for (const auto &replica : partition->getReplicas())
    {
      if (replica->getInstanceId() == instanceId)
      {
        isReplica             = true;
        replicaPartitionIndex = i;
        myCoordinatorId       = partition->getCoordinatorInstanceId();
      }
    }
  }

  if (isCoordinator) printf("[Instance %lu] Coordinator (partition %zu)\n", instanceId, coordinatorPartitionIndex);
  if (isReplica) printf("[Instance %lu] Replica (coordinator: %lu)\n", instanceId, myCoordinatorId);

  serving::system::channels::keyBuilderFc_t keyBuilder              = defaultChannelKeyBuilder;
  std::vector<HiCR::CommunicationManager *> managerOrder            = {runtime.communicationManager.get()};
  auto                                      channelControllerModule = std::make_shared<serving::modules::channelController::Module>(instanceId, managerOrder);
  auto                                      channelDispatcherModule = std::make_shared<serving::modules::channelDispatcher::Module>(20);
  auto                                      serviceModule           = std::make_shared<serving::modules::service::Module>(runtime.taskr);

  serviceModule->addService("ChannelController", channelControllerModule->getService());
  serviceModule->addService("ChannelDispatcher", channelDispatcherModule->getService());
  serving.addModule("ChannelController", channelControllerModule);
  serving.addModule("ChannelDispatcher", channelDispatcherModule);
  serving.addModule("Service", serviceModule);

  std::optional<CoordinatorRuntime> coordinatorRt;
  std::optional<ReplicaRuntime>     replicaRt;

  if (isCoordinator)
  {
    coordinatorRt = coordinator(
      *instance, coordinatorPartitionIndex, myReplicaIds, deployment, keyBuilder, channelControllerModule, channelDispatcherModule, serviceModule, serving, kMessageType);
  }
  if (isReplica)
  {
    replicaRt =
      replica(instanceId, myCoordinatorId, replicaPartitionIndex, deployment, keyBuilder, channelControllerModule, channelDispatcherModule, serviceModule, serving, kMessageType);
  }

  serving.initialize();
  serving.run();

  if (isRoot && isCoordinator && coordinatorRt.has_value())
  {
    waitUntilReady(coordinatorRt->graphChannels.output);

    const std::string                              payload = "telephone-start";
    serving::system::channels::Message::metadata_t md;
    md.type       = kMessageType;
    md.groupId    = 1;
    md.sequenceId = 1;
    serving::system::channels::Message msg(reinterpret_cast<const uint8_t *>(payload.data()), payload.size(), md);
    coordinatorRt->graphChannels.output->pushMessageLocking(msg);
    printf("[Instance %lu] Sent: %s\n", instanceId, payload.c_str());

    while (!coordinatorRt->completed->load()) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }

    printf("[Instance %lu] Message returned to root. Terminating.\n", instanceId);
    serving.terminate();
  }

  serving.await();

  // Cleanup
  if (coordinatorRt.has_value())
  {
    channelDispatcherModule->unsubscribe(coordinatorRt->graphUnsubscription.first, coordinatorRt->graphUnsubscription.second);
    for (const auto &[type, input] : coordinatorRt->coordinatorModule->buildUnsubscriptions()) { channelDispatcherModule->unsubscribe(type, input); }
    removeDesiredSingleLocalChannels(channelControllerModule, coordinatorRt->graphChannels);
    const auto &task = deployment.getPartitions().at(coordinatorPartitionIndex)->getTasks().front();
    for (const auto replicaId : myReplicaIds)
    {
      for (const auto &inputName : task->getInputs()) channelControllerModule->removeDesiredProducer(makeCoordinatorToReplicaEdgeName(inputName, instanceId, replicaId));
      for (const auto &outputName : task->getOutputs()) channelControllerModule->removeDesiredConsumer(makeReplicaToCoordinatorEdgeName(outputName, replicaId, instanceId));
    }
  }

  if (replicaRt.has_value())
  {
    for (const auto &[type, input] : replicaRt->replicaModule->buildUnsubscriptions()) { channelDispatcherModule->unsubscribe(type, input); }
    const auto &task = deployment.getPartitions().at(replicaPartitionIndex)->getTasks().front();
    for (const auto &inputName : task->getInputs()) channelControllerModule->removeDesiredConsumer(makeCoordinatorToReplicaEdgeName(inputName, myCoordinatorId, instanceId));
    for (const auto &outputName : task->getOutputs()) channelControllerModule->removeDesiredProducer(makeReplicaToCoordinatorEdgeName(outputName, instanceId, myCoordinatorId));
  }

  runtime.instanceManager->finalize();

  return 0;
}
