#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/channelController/module.hpp>
#include <modules/channelDispatcher/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/roles/coordinator/module.hpp>
#include <modules/roles/job.hpp>
#include <modules/roles/jobFactory.hpp>
#include <modules/service/module.hpp>
#include <system/channels/message.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>

using instanceId_t = HiCR::Instance::instanceId_t;

struct CoordinatorRuntime
{
  using messageType_t = serving::system::channels::Message::messageType_t;
  using input_t       = std::shared_ptr<serving::system::channels::Input>;

  std::shared_ptr<serving::modules::roles::coordinator::Module> coordinatorModule;
  std::shared_ptr<serving::modules::roles::JobFactory>          jobFactory;
  desiredSingleLocalChannels_t                                  graphChannels;
  std::pair<messageType_t, input_t>                             graphUnsubscription;
  std::shared_ptr<std::atomic<bool>>                            completed = std::make_shared<std::atomic<bool>>(false);
};

__INLINE__ void submitGraphJob(const std::shared_ptr<serving::modules::roles::JobFactory>          &jobFactory,
                               const std::shared_ptr<serving::modules::roles::coordinator::Module> &coordinatorModule,
                               const std::string                                                   &jobName,
                               const std::string                                                   &inputEdgeName,
                               const serving::system::channels::Message                            &message)
{
  auto  job        = std::make_shared<serving::modules::roles::Job>(jobFactory->createJob(jobName, message.getMetadata()));
  auto &dependency = job->getInputDependency(inputEdgeName);
  dependency.storeData(message.getData(), message.getSize());
  dependency.setSatisfied(true);
  coordinatorModule->submitJob(job);
}

__INLINE__ void handleGraphMessage(const bool                                                           isRoot,
                                   const std::shared_ptr<std::atomic<bool>>                            &completed,
                                   const std::shared_ptr<serving::modules::roles::JobFactory>          &jobFactory,
                                   const std::shared_ptr<serving::modules::roles::coordinator::Module> &coordinatorModule,
                                   const std::string                                                   &jobName,
                                   const std::string                                                   &inputEdgeName,
                                   const serving::system::channels::Message                            &message)
{
  if (isRoot)
  {
    completed->store(true);
    return;
  }
  submitGraphJob(jobFactory, coordinatorModule, jobName, inputEdgeName, message);
}

__INLINE__ void routeOutput(const std::shared_ptr<serving::system::channels::Output>                   &outputChannel,
                            const serving::system::channels::Message::metadata_t                       &metadata,
                            const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs)
{
  for (const auto &output : outputs)
  {
    waitUntilReady(outputChannel);
    const auto message = serving::system::channels::Message(
      output.data == nullptr ? nullptr : static_cast<const uint8_t *>(output.data->getPointer()), output.data == nullptr ? 0 : output.data->getSize(), metadata);
    outputChannel->pushMessageLocking(message);
  }
}

// partitionIndex is the index of the partition this instance coordinates.
// replicaIds lists all replica instance IDs for this partition (first may be co-located).
__INLINE__ CoordinatorRuntime coordinator(const HiCR::Instance                                               &instance,
                                          const size_t                                                        partitionIndex,
                                          const std::vector<instanceId_t>                                    &replicaIds,
                                          serving::configuration::Deployment                                 &deployment,
                                          const serving::system::channels::keyBuilderFc_t                    &keyBuilder,
                                          const std::shared_ptr<serving::modules::channelController::Module> &channelControllerModule,
                                          const std::shared_ptr<serving::modules::channelDispatcher::Module> &channelDispatcherModule,
                                          const std::shared_ptr<serving::modules::service::Module>           &serviceModule,
                                          serving::system::Engine                                            &serving,
                                          const serving::system::channels::Message::messageType_t             messageType)
{
  const auto instanceId = instance.getId();
  const auto isRoot     = instance.isRootInstance();

  CoordinatorRuntime coordinatorRuntime;

  coordinatorRuntime.jobFactory        = std::make_shared<serving::modules::roles::JobFactory>(partitionIndex, deployment);
  coordinatorRuntime.coordinatorModule = std::make_shared<serving::modules::roles::coordinator::Module>(100);
  coordinatorRuntime.coordinatorModule->addMessageType(messageType);

  serving.addModule("Coordinator", coordinatorRuntime.coordinatorModule);
  serviceModule->addService("Coordinator", coordinatorRuntime.coordinatorModule->getService());

  coordinatorRuntime.graphChannels = createDesiredSingleLocalChannels(deployment, instanceId, keyBuilder, channelControllerModule);

  const auto &task = deployment.getPartitions().at(partitionIndex)->getTasks().front();

  // Wire internal channels to every replica in this partition.
  for (const auto replicaId : replicaIds)
  {
    serving::modules::roles::coordinator::Module::inputChannelMap_t  replicaInputChannels;
    serving::modules::roles::coordinator::Module::outputChannelMap_t replicaOutputChannels;
    size_t                                                           channelOffset = 0;

    for (const auto &inputName : task->getInputs())
    {
      const auto edge                 = getEdgeByName(deployment, inputName);
      const auto internalEdgeName     = makeCoordinatorToReplicaEdgeName(inputName, instanceId, replicaId);
      const auto internalEdge         = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      const auto channelId            = static_cast<serving::system::channels::channelId_t>(1000 + instanceId * 100 + channelOffset++);
      replicaInputChannels[inputName] = channelControllerModule->addDesiredProducer(replicaId, channelId, internalEdge, keyBuilder).lock();
    }
    channelOffset = 0;
    for (const auto &outputName : task->getOutputs())
    {
      const auto edge                   = getEdgeByName(deployment, outputName);
      const auto internalEdgeName       = makeReplicaToCoordinatorEdgeName(outputName, replicaId, instanceId);
      const auto internalEdge           = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      const auto channelId              = static_cast<serving::system::channels::channelId_t>(2000 + instanceId * 100 + channelOffset++);
      replicaOutputChannels[outputName] = channelControllerModule->addDesiredConsumer(replicaId, channelId, internalEdge, keyBuilder).lock();
    }
    coordinatorRuntime.coordinatorModule->addReplica(replicaId, replicaInputChannels, replicaOutputChannels);
  }

  const auto jobName       = task->getFunctionName();
  const auto inputEdgeName = coordinatorRuntime.graphChannels.inputInfo.edge->getName();

  serving::modules::Subscription graphSubscription(
    messageType,
    coordinatorRuntime.graphChannels.input,
    [isRoot, completed = coordinatorRuntime.completed, jobFactory = coordinatorRuntime.jobFactory, coordinatorModule = coordinatorRuntime.coordinatorModule, jobName, inputEdgeName](
      const std::shared_ptr<serving::system::channels::Input>, const serving::system::channels::Message &message) {
      handleGraphMessage(isRoot, completed, jobFactory, coordinatorModule, jobName, inputEdgeName, message);
    });

  channelDispatcherModule->subscribe(graphSubscription);
  coordinatorRuntime.graphUnsubscription = {messageType, coordinatorRuntime.graphChannels.input};

  coordinatorRuntime.coordinatorModule->setCompletionCallback(
    [outputChannel = coordinatorRuntime.graphChannels.output](const serving::system::channels::Message::metadata_t                       &metadata,
                                                              const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs) {
      routeOutput(outputChannel, metadata, outputs);
    });

  for (auto &subscription : coordinatorRuntime.coordinatorModule->buildSubscriptions()) { channelDispatcherModule->subscribe(subscription); }
  return coordinatorRuntime;
}
