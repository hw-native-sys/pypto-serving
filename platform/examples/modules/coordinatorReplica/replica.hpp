#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/channelController/module.hpp>
#include <modules/channelDispatcher/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/roles/replica/module.hpp>
#include <modules/service/module.hpp>
#include <system/channels/message.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>

using instanceId_t = HiCR::Instance::instanceId_t;

struct ReplicaRuntime
{
  std::shared_ptr<serving::modules::roles::replica::Module> replicaModule;
  instanceId_t                                              coordinatorId;
};

__INLINE__ ReplicaRuntime replica(const instanceId_t                                                  instanceId,
                                  const instanceId_t                                                  coordinatorId,
                                  const size_t                                                        partitionIndex,
                                  serving::configuration::Deployment                                 &deployment,
                                  const serving::system::channels::keyBuilderFc_t                    &keyBuilder,
                                  const std::shared_ptr<serving::modules::channelController::Module> &channelControllerModule,
                                  const std::shared_ptr<serving::modules::channelDispatcher::Module> &channelDispatcherModule,
                                  const std::shared_ptr<serving::modules::service::Module>           &serviceModule,
                                  serving::system::Engine                                            &serving,
                                  const serving::system::channels::Message::messageType_t             messageType)
{
  ReplicaRuntime replicaRuntime;
  replicaRuntime.coordinatorId = coordinatorId;

  const auto &task = deployment.getPartitions().at(partitionIndex)->getTasks().front();

  auto processFc = [instanceId, inputName = task->getInputs().front(), outputName = task->getOutputs().front()](serving::modules::roles::TaskContext &context) {
    const auto  input = context.getInput(inputName);
    std::string text(reinterpret_cast<const char *>(input->getPointer()), input->getSize());
    text += " -> R";
    text += std::to_string(instanceId);
    printf("[Instance %lu] Replica processed: %s\n", instanceId, text.c_str());
    context.setOutput(outputName, text.data(), text.size());
  };

  replicaRuntime.replicaModule = std::make_shared<serving::modules::roles::replica::Module>(task->getFunctionName(), task->getInputs(), task->getOutputs(), processFc);
  replicaRuntime.replicaModule->addMessageType(messageType);

  serving.addModule("Replica", replicaRuntime.replicaModule);

  serving::modules::roles::replica::Module::inputMap_t  inputChannels;
  serving::modules::roles::replica::Module::outputMap_t outputChannels;
  size_t                                                channelOffset = 0;

  for (const auto &inputName : task->getInputs())
  {
    const auto edge             = getEdgeByName(deployment, inputName);
    const auto internalEdgeName = makeCoordinatorToReplicaEdgeName(inputName, coordinatorId, instanceId);
    const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
    const auto channelId        = static_cast<serving::system::channels::channelId_t>(1000 + coordinatorId * 100 + channelOffset++);
    inputChannels[inputName]    = channelControllerModule->addDesiredConsumer(coordinatorId, channelId, internalEdge, keyBuilder).lock();
  }
  channelOffset = 0;
  for (const auto &outputName : task->getOutputs())
  {
    const auto edge             = getEdgeByName(deployment, outputName);
    const auto internalEdgeName = makeReplicaToCoordinatorEdgeName(outputName, instanceId, coordinatorId);
    const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
    const auto channelId        = static_cast<serving::system::channels::channelId_t>(2000 + coordinatorId * 100 + channelOffset++);
    outputChannels[outputName]  = channelControllerModule->addDesiredProducer(coordinatorId, channelId, internalEdge, keyBuilder).lock();
  }
  replicaRuntime.replicaModule->setCoordinator(coordinatorId, outputChannels, inputChannels);

  for (auto &subscription : replicaRuntime.replicaModule->buildSubscriptions()) { channelDispatcherModule->subscribe(subscription); }

  return replicaRuntime;
}
