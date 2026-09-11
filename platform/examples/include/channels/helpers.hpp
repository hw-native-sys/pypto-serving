#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/channelController/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <system/channels/input.hpp>
#include <system/channels/output.hpp>

#include <deployment/helpers.hpp>

struct localChannels_t
{
  std::vector<localInput_t>  inputs;
  std::vector<localOutput_t> outputs;
};

struct singleLocalChannels_t
{
  localInput_t  input;
  localOutput_t output;
};

struct desiredSingleLocalChannels_t
{
  std::shared_ptr<serving::system::channels::Input>  input;
  std::shared_ptr<serving::system::channels::Output> output;
  std::string                                        inputName;
  std::string                                        outputName;
  localInput_t                                       inputInfo;
  localOutput_t                                      outputInfo;
};

__INLINE__ localChannels_t buildLocalChannelInfos(const serving::configuration::Deployment        &deployment,
                                                  const HiCR::Instance::instanceId_t               instanceId,
                                                  const serving::system::channels::keyBuilderFc_t &keyBuilder)
{
  localChannels_t channels;
  buildLocalChannelsFromDeploymentWithIds(deployment, instanceId, keyBuilder, channels.inputs, channels.outputs);
  return channels;
}

__INLINE__ singleLocalChannels_t buildSingleLocalChannelInfos(const serving::configuration::Deployment        &deployment,
                                                              const HiCR::Instance::instanceId_t               instanceId,
                                                              const serving::system::channels::keyBuilderFc_t &keyBuilder)
{
  auto channels = buildLocalChannelInfos(deployment, instanceId, keyBuilder);
  if (channels.inputs.size() != 1) HICR_THROW_LOGIC("Expected exactly one local input channel, got %lu.", channels.inputs.size());
  if (channels.outputs.size() != 1) HICR_THROW_LOGIC("Expected exactly one local output channel, got %lu.", channels.outputs.size());
  return singleLocalChannels_t{
    .input  = channels.inputs.front(),
    .output = channels.outputs.front(),
  };
}

__INLINE__ std::shared_ptr<serving::system::channels::Input> createDesiredInput(const std::shared_ptr<serving::modules::channelController::Module> &channelController,
                                                                                const localInput_t                                                 &inputInfo,
                                                                                const serving::system::channels::keyBuilderFc_t                    &keyBuilder)
{
  auto input = channelController->addDesiredConsumer(inputInfo.sourceInstanceId, inputInfo.channelId, *inputInfo.edge, keyBuilder).lock();
  return input;
}

__INLINE__ std::shared_ptr<serving::system::channels::Output> createDesiredOutput(const std::shared_ptr<serving::modules::channelController::Module> &channelController,
                                                                                  const localOutput_t                                                &outputInfo,
                                                                                  const serving::system::channels::keyBuilderFc_t                    &keyBuilder)
{
  auto output = channelController->addDesiredProducer(outputInfo.targetInstanceId, outputInfo.channelId, *outputInfo.edge, keyBuilder).lock();
  return output;
}

__INLINE__ desiredSingleLocalChannels_t createDesiredSingleLocalChannels(const serving::configuration::Deployment                           &deployment,
                                                                         const HiCR::Instance::instanceId_t                                  instanceId,
                                                                         const serving::system::channels::keyBuilderFc_t                    &keyBuilder,
                                                                         const std::shared_ptr<serving::modules::channelController::Module> &channelController)
{
  auto infos  = buildSingleLocalChannelInfos(deployment, instanceId, keyBuilder);
  auto input  = createDesiredInput(channelController, infos.input, keyBuilder);
  auto output = createDesiredOutput(channelController, infos.output, keyBuilder);
  return desiredSingleLocalChannels_t{
    .input      = input,
    .output     = output,
    .inputName  = infos.input.edge->getName(),
    .outputName = infos.output.edge->getName(),
    .inputInfo  = infos.input,
    .outputInfo = infos.output,
  };
}

__INLINE__ void removeDesiredInputs(const std::shared_ptr<serving::modules::channelController::Module> &channelController, const std::vector<localInput_t> &inputs)
{
  for (const auto &input : inputs) { channelController->removeDesiredConsumer(input.edge->getName()); }
}

__INLINE__ void removeDesiredOutputs(const std::shared_ptr<serving::modules::channelController::Module> &channelController, const std::vector<localOutput_t> &outputs)
{
  for (const auto &output : outputs) { channelController->removeDesiredProducer(output.edge->getName()); }
}

__INLINE__ void removeDesiredSingleLocalChannels(const std::shared_ptr<serving::modules::channelController::Module> &channelController, desiredSingleLocalChannels_t &channels)
{
  channelController->removeDesiredConsumer(channels.inputName);
  channelController->removeDesiredProducer(channels.outputName);
  channels.input  = {};
  channels.output = {};
}

__INLINE__ void waitUntilReady(const std::shared_ptr<serving::system::channels::Base> &channel, const size_t intervalMs = 500)
{
  while (!channel->isReady()) { std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs)); }
}

__INLINE__ std::shared_ptr<serving::configuration::Edge> getEdgeByName(serving::configuration::Deployment &deployment, const std::string &edgeName)
{
  for (const auto &edge : deployment.getEdges())
    if (edge->getName() == edgeName) return edge;
  HICR_THROW_LOGIC("Deployment has no edge named '%s'.", edgeName.c_str());
}

__INLINE__ std::string makeCoordinatorToReplicaEdgeName(const std::string &edgeName, const HiCR::Instance::instanceId_t coordinatorId, const HiCR::Instance::instanceId_t replicaId)
{
  return edgeName + "-coordinator" + std::to_string(coordinatorId) + "-replica" + std::to_string(replicaId);
}

__INLINE__ std::string makeReplicaToCoordinatorEdgeName(const std::string &edgeName, const HiCR::Instance::instanceId_t replicaId, const HiCR::Instance::instanceId_t coordinatorId)
{
  return edgeName + "-replica" + std::to_string(replicaId) + "-coordinator" + std::to_string(coordinatorId);
}

__INLINE__ serving::configuration::Edge makeInternalEdgeFromTemplate(const std::string &name, const serving::configuration::Edge &edgeTemplate)
{
  serving::configuration::Edge edge(name, edgeTemplate.getBufferCapacity(), edgeTemplate.getBufferSize());
  edge.setPayloadCommunicationManager(edgeTemplate.getPayloadCommunicationManager());
  edge.setPayloadMemoryManager(edgeTemplate.getPayloadMemoryManager());
  edge.setPayloadMemorySpace(edgeTemplate.getPayloadMemorySpace());
  edge.setCoordinationCommunicationManager(edgeTemplate.getCoordinationCommunicationManager());
  edge.setCoordinationMemoryManager(edgeTemplate.getCoordinationMemoryManager());
  edge.setCoordinationMemorySpace(edgeTemplate.getCoordinationMemorySpace());
  return edge;
}