#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/channelController/module.hpp>
#include <modules/channelDispatcher/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/configuration/replica.hpp>
#include <modules/requestManager/module.hpp>
#include <modules/roles/coordinator/module.hpp>
#include <modules/roles/jobFactory.hpp>
#include <modules/roles/replica/module.hpp>
#include <modules/router/defaultProcessFunction.hpp>
#include <modules/router/module.hpp>
#include <modules/service/module.hpp>
#include <modules/subscription.hpp>
#include <system/channels/message.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>
#include <runtime/helpers.hpp>

class ModuleDeploymentRunner
{
  public:

  using instanceId_t  = HiCR::Instance::instanceId_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using channelId_t   = serving::system::channels::channelId_t;
  using processFc_t   = serving::modules::roles::replica::Module::processFc_t;

  ModuleDeploymentRunner(Runtime &runtime, serving::configuration::Deployment &deployment, const messageType_t messageType, processFc_t processFc)
    : _runtime(runtime),
      _deployment(deployment),
      _messageType(messageType),
      _processFc(std::move(processFc)),
      _instanceId(runtime.instanceManager->getCurrentInstance()->getId()),
      _deployerId(runtime.instanceManager->getRootInstanceId()),
      _partitionCount(deployment.getPartitions().size()),
      _requestManagerId(static_cast<instanceId_t>(_partitionCount * 2))
  {}

  [[nodiscard]] bool isCoordinator() const { return _instanceId < _partitionCount; }
  [[nodiscard]] bool isReplica() const { return _instanceId >= _partitionCount && _instanceId < _requestManagerId; }
  [[nodiscard]] bool isRequestManager() const { return _instanceId == _requestManagerId; }
  [[nodiscard]] bool isDeployer() const { return _instanceId == _deployerId; }

  [[nodiscard]] auto getRequestManager() const { return _requestManagerModule; }

  void addCoreModules()
  {
    std::vector<HiCR::CommunicationManager *> managerOrder = {_runtime.communicationManager.get()};
    _channelControllerModule                               = std::make_shared<serving::modules::channelController::Module>(_instanceId, managerOrder);
    _channelDispatcherModule                               = std::make_shared<serving::modules::channelDispatcher::Module>(1);
    _serviceModule                                         = std::make_shared<serving::modules::service::Module>(_runtime.taskr);

    _serviceModule->addService("ChannelController", _channelControllerModule->getService());
    _serviceModule->addService("ChannelDispatcher", _channelDispatcherModule->getService());
    _runtime.serving->addModule("ChannelController", _channelControllerModule);
    _runtime.serving->addModule("ChannelDispatcher", _channelDispatcherModule);
    _runtime.serving->addModule("Service", _serviceModule);
  }

  void wireGraph()
  {
    const auto partitionInstanceMap = buildPartitionInstanceMap();
    _promptEdgeName                 = _deployment.getRequestManager()->getInput();
    _resultEdgeName                 = _deployment.getRequestManager()->getOutput();
    const auto promptPartitionId    = partitionInstanceMap.at(getEdgeByName(_deployment, _promptEdgeName)->getConsumer());
    const auto resultPartitionId    = partitionInstanceMap.at(getEdgeByName(_deployment, _resultEdgeName)->getProducer());

    for (serving::configuration::Edge::edgeIndex_t edgeIdx = 0; edgeIdx < _deployment.getEdges().size(); ++edgeIdx)
    {
      const auto  &edge     = _deployment.getEdges().at(edgeIdx);
      const auto  &edgeName = edge->getName();
      instanceId_t sourceId = 0;
      instanceId_t targetId = 0;
      if (edgeName == _promptEdgeName)
      {
        sourceId = _requestManagerId;
        targetId = promptPartitionId;
      }
      else if (edgeName == _resultEdgeName)
      {
        sourceId = resultPartitionId;
        targetId = _requestManagerId;
      }
      else
      {
        sourceId = partitionInstanceMap.at(edge->getProducer());
        targetId = partitionInstanceMap.at(edge->getConsumer());
      }
      _graphChannels[edgeName] = makeGraphChannel(edgeName, sourceId, targetId, edgeIdx);
    }

    wireRequestManager();
    wireCoordinator();
    wireReplica();
  }

  void enableRemoteShutdown(const std::string &edgeName, const channelId_t channelId)
  {
    _doneEdgeName               = edgeName;
    const auto doneEdgeTemplate = makeInternalEdgeFromTemplate(edgeName, *getEdgeByName(_deployment, _resultEdgeName));
    if (isRequestManager()) _doneOutput = _channelControllerModule->addDesiredProducer(_deployerId, channelId, doneEdgeTemplate, defaultChannelKeyBuilder).lock();
    if (isDeployer()) _doneInput = _channelControllerModule->addDesiredConsumer(_requestManagerId, channelId, doneEdgeTemplate, defaultChannelKeyBuilder).lock();
  }

  void subscribeAfterInitialize()
  {
    if (isRequestManager())
    {
      waitUntilReady(_graphChannels.at(_promptEdgeName).output);
      waitUntilReady(_graphChannels.at(_resultEdgeName).input);
      if (_doneOutput != nullptr) waitUntilReady(_doneOutput);
      for (auto &subscription : _requestManagerModule->buildSubscriptions()) _channelDispatcherModule->subscribe(subscription);
    }

    if (_doneInput != nullptr)
    {
      waitUntilReady(_doneInput);
      _channelDispatcherModule->subscribe(serving::modules::Subscription(
        _messageType, _doneInput, [this](const std::shared_ptr<serving::system::channels::Input>, const serving::system::channels::Message &) { _runtime.serving->terminate(); }));
    }

    if (isCoordinator())
    {
      const auto &task = _deployment.getPartitions().at(static_cast<size_t>(_instanceId))->getTasks().front();
      for (const auto &inputName : task->getInputs()) waitUntilReady(_graphChannels.at(inputName).input);
      for (const auto &outputName : task->getOutputs()) waitUntilReady(_graphChannels.at(outputName).output);
      for (auto &subscription : _routerModule->buildSubscriptions()) _channelDispatcherModule->subscribe(subscription);
      for (auto &subscription : _coordinatorModule->buildSubscriptions()) _channelDispatcherModule->subscribe(subscription);
    }

    if (isReplica())
    {
      for (auto &subscription : _replicaModule->buildSubscriptions()) _channelDispatcherModule->subscribe(subscription);
    }
  }

  void signalShutdown()
  {
    if (_doneOutput == nullptr) HICR_THROW_LOGIC("Remote shutdown is not enabled on this instance.");
    serving::system::channels::Message::metadata_t metadata;
    metadata.type         = _messageType;
    metadata.groupId      = 0;
    metadata.sequenceId   = 1;
    const uint8_t payload = 1;
    _doneOutput->pushMessageLocking(serving::system::channels::Message(&payload, sizeof(payload), metadata));
  }

  void cleanup()
  {
    if (isRequestManager())
    {
      for (const auto &[type, input] : _requestManagerModule->buildUnsubscriptions()) _channelDispatcherModule->unsubscribe(type, input);
      if (_doneOutput != nullptr) _channelControllerModule->removeDesiredProducer(_doneEdgeName);
    }

    if (_doneInput != nullptr)
    {
      _channelDispatcherModule->unsubscribe(_messageType, _doneInput);
      _channelControllerModule->removeDesiredConsumer(_doneEdgeName);
    }

    if (isCoordinator())
    {
      for (const auto &[type, input] : _routerModule->buildUnsubscriptions()) _channelDispatcherModule->unsubscribe(type, input);
      for (const auto &[type, input] : _coordinatorModule->buildUnsubscriptions()) _channelDispatcherModule->unsubscribe(type, input);
      const auto &task      = _deployment.getPartitions().at(static_cast<size_t>(_instanceId))->getTasks().front();
      const auto  replicaId = static_cast<instanceId_t>(_instanceId + _partitionCount);
      for (const auto &inputName : task->getInputs()) _channelControllerModule->removeDesiredProducer(makeCoordinatorToReplicaEdgeName(inputName, _instanceId, replicaId));
      for (const auto &outputName : task->getOutputs()) _channelControllerModule->removeDesiredConsumer(makeReplicaToCoordinatorEdgeName(outputName, replicaId, _instanceId));
    }

    if (isReplica())
    {
      for (const auto &[type, input] : _replicaModule->buildUnsubscriptions()) _channelDispatcherModule->unsubscribe(type, input);
      const auto  coordinatorId = static_cast<instanceId_t>(_instanceId - _partitionCount);
      const auto &task          = _deployment.getPartitions().at(static_cast<size_t>(coordinatorId))->getTasks().front();
      for (const auto &inputName : task->getInputs()) _channelControllerModule->removeDesiredConsumer(makeCoordinatorToReplicaEdgeName(inputName, coordinatorId, _instanceId));
      for (const auto &outputName : task->getOutputs()) _channelControllerModule->removeDesiredProducer(makeReplicaToCoordinatorEdgeName(outputName, _instanceId, coordinatorId));
    }

    for (const auto &[edgeName, channel] : _graphChannels)
    {
      if (channel.output != nullptr) _channelControllerModule->removeDesiredProducer(edgeName);
      if (channel.input != nullptr) _channelControllerModule->removeDesiredConsumer(edgeName);
    }
  }

  private:

  struct GraphChannel
  {
    std::shared_ptr<serving::configuration::Edge>      edge;
    channelId_t                                        channelId = 0;
    instanceId_t                                       sourceId  = 0;
    instanceId_t                                       targetId  = 0;
    std::shared_ptr<serving::system::channels::Input>  input;
    std::shared_ptr<serving::system::channels::Output> output;
  };

  std::unordered_map<std::string, instanceId_t> buildPartitionInstanceMap() const
  {
    std::unordered_map<std::string, instanceId_t> out;
    for (const auto &partition : _deployment.getPartitions()) out[partition->getName()] = partition->getCoordinatorInstanceId();
    return out;
  }

  GraphChannel makeGraphChannel(const std::string &edgeName, const instanceId_t sourceId, const instanceId_t targetId, const channelId_t channelId)
  {
    GraphChannel channel;
    channel.edge      = getEdgeByName(_deployment, edgeName);
    channel.channelId = channelId;
    channel.sourceId  = sourceId;
    channel.targetId  = targetId;
    if (_instanceId == sourceId) channel.output = _channelControllerModule->addDesiredProducer(targetId, channelId, *channel.edge, defaultChannelKeyBuilder).lock();
    if (_instanceId == targetId) channel.input = _channelControllerModule->addDesiredConsumer(sourceId, channelId, *channel.edge, defaultChannelKeyBuilder).lock();
    return channel;
  }

  void wireRequestManager()
  {
    if (!isRequestManager()) return;
    _requestManagerModule = std::make_shared<serving::modules::roles::requestManager::Module>();
    _requestManagerModule->addMessageType(_messageType);
    _requestManagerModule->setPromptOutput(_graphChannels.at(_promptEdgeName).output);
    _requestManagerModule->setResultInput(_graphChannels.at(_resultEdgeName).input);
    _runtime.serving->addModule("RequestManager", _requestManagerModule);
  }

  void wireCoordinator()
  {
    if (!isCoordinator()) return;
    const auto &partition = *_deployment.getPartitions().at(static_cast<size_t>(_instanceId));
    const auto &task      = partition.getTasks().front();
    const auto  replicaId = static_cast<instanceId_t>(_instanceId + _partitionCount);

    _jobFactory        = std::make_shared<serving::modules::roles::JobFactory>(_instanceId, _deployment);
    _coordinatorModule = std::make_shared<serving::modules::roles::coordinator::Module>(1);
    _routerModule      = std::make_shared<serving::modules::router::Module>();
    _coordinatorModule->addMessageType(_messageType);
    _routerModule->addMessageType(_messageType);
    _runtime.serving->addModule("Coordinator", _coordinatorModule);
    _runtime.serving->addModule("Router", _routerModule);
    _serviceModule->addService("Coordinator", _coordinatorModule->getService());

    serving::modules::roles::coordinator::Module::inputChannelMap_t  replicaInputChannels;
    serving::modules::roles::coordinator::Module::outputChannelMap_t replicaOutputChannels;
    size_t                                                           channelOffset = 0;
    for (const auto &inputName : task->getInputs())
    {
      const auto edge             = getEdgeByName(_deployment, inputName);
      const auto internalEdgeName = makeCoordinatorToReplicaEdgeName(inputName, _instanceId, replicaId);
      const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      replicaInputChannels[inputName] =
        _channelControllerModule->addDesiredProducer(replicaId, static_cast<channelId_t>(1000 + _instanceId * 100 + channelOffset++), internalEdge, defaultChannelKeyBuilder).lock();
      _routerModule->addInput(inputName, _graphChannels.at(inputName).input);
    }
    channelOffset = 0;
    for (const auto &outputName : task->getOutputs())
    {
      const auto edge             = getEdgeByName(_deployment, outputName);
      const auto internalEdgeName = makeReplicaToCoordinatorEdgeName(outputName, replicaId, _instanceId);
      const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      replicaOutputChannels[outputName] =
        _channelControllerModule->addDesiredConsumer(replicaId, static_cast<channelId_t>(2000 + _instanceId * 100 + channelOffset++), internalEdge, defaultChannelKeyBuilder).lock();
      _routerModule->addOutput(outputName, _graphChannels.at(outputName).output);
    }
    _coordinatorModule->addReplica(replicaId, replicaInputChannels, replicaOutputChannels);

    auto defaultProcess = serving::modules::router::makeDefaultProcessFunction(*_jobFactory, *_coordinatorModule, partition);
    _routerModule->setProcessFunction(defaultProcess);
    _coordinatorModule->setCompletionCallback(
      [this](const serving::system::channels::Message::metadata_t &metadata, const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs) {
        _routerModule->routeOutputs(metadata, outputs);
      });
  }

  void wireReplica()
  {
    if (!isReplica()) return;
    const auto  coordinatorId = static_cast<instanceId_t>(_instanceId - _partitionCount);
    const auto &task          = _deployment.getPartitions().at(static_cast<size_t>(coordinatorId))->getTasks().front();
    _replicaModule            = std::make_shared<serving::modules::roles::replica::Module>(task->getFunctionName(), task->getInputs(), task->getOutputs(), _processFc);
    _replicaModule->addMessageType(_messageType);
    _runtime.serving->addModule("Replica", _replicaModule);

    serving::modules::roles::replica::Module::inputMap_t  inputChannels;
    serving::modules::roles::replica::Module::outputMap_t outputChannels;
    size_t                                                channelOffset = 0;
    for (const auto &inputName : task->getInputs())
    {
      const auto edge             = getEdgeByName(_deployment, inputName);
      const auto internalEdgeName = makeCoordinatorToReplicaEdgeName(inputName, coordinatorId, _instanceId);
      const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      inputChannels[inputName] =
        _channelControllerModule->addDesiredConsumer(coordinatorId, static_cast<channelId_t>(1000 + coordinatorId * 100 + channelOffset++), internalEdge, defaultChannelKeyBuilder)
          .lock();
    }
    channelOffset = 0;
    for (const auto &outputName : task->getOutputs())
    {
      const auto edge             = getEdgeByName(_deployment, outputName);
      const auto internalEdgeName = makeReplicaToCoordinatorEdgeName(outputName, _instanceId, coordinatorId);
      const auto internalEdge     = makeInternalEdgeFromTemplate(internalEdgeName, *edge);
      outputChannels[outputName] =
        _channelControllerModule->addDesiredProducer(coordinatorId, static_cast<channelId_t>(2000 + coordinatorId * 100 + channelOffset++), internalEdge, defaultChannelKeyBuilder)
          .lock();
    }
    _replicaModule->setCoordinator(coordinatorId, outputChannels, inputChannels);
  }

  Runtime                            &_runtime;
  serving::configuration::Deployment &_deployment;
  messageType_t                       _messageType;
  processFc_t                         _processFc;
  instanceId_t                        _instanceId;
  instanceId_t                        _deployerId;
  size_t                              _partitionCount;
  instanceId_t                        _requestManagerId;
  std::string                         _promptEdgeName;
  std::string                         _resultEdgeName;
  std::string                         _doneEdgeName;

  std::unordered_map<std::string, GraphChannel> _graphChannels;

  std::shared_ptr<serving::modules::channelController::Module>     _channelControllerModule;
  std::shared_ptr<serving::modules::channelDispatcher::Module>     _channelDispatcherModule;
  std::shared_ptr<serving::modules::service::Module>               _serviceModule;
  std::shared_ptr<serving::modules::roles::requestManager::Module> _requestManagerModule;
  std::shared_ptr<serving::modules::roles::coordinator::Module>    _coordinatorModule;
  std::shared_ptr<serving::modules::router::Module>                _routerModule;
  std::shared_ptr<serving::modules::roles::JobFactory>             _jobFactory;
  std::shared_ptr<serving::modules::roles::replica::Module>        _replicaModule;
  std::shared_ptr<serving::system::channels::Output>               _doneOutput;
  std::shared_ptr<serving::system::channels::Input>                _doneInput;
};
