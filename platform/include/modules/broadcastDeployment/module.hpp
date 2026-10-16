#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <hicr/core/computeManager.hpp>
#include <hicr/core/instanceManager.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <modules/configuration/deployment.hpp>
#include <modules/module.hpp>

namespace serving::modules::broadcastDeployment
{

#define __SERVING_REQUEST_DEPLOYMENT_CONFIGURATION_RPC_NAME "[serving/modules/broadcastDeployment] Request Deployment Configuration RPC"

class Module final : public modules::Module
{
  public:

  Module(std::shared_ptr<HiCR::InstanceManager>     instanceManager,
         std::shared_ptr<HiCR::ComputeManager>      computeManager,
         std::shared_ptr<HiCR::frontend::RPCEngine> rpcEngine,
         const HiCR::Instance::instanceId_t         deployerInstanceId,
         const HiCR::Instance::instanceId_t         instanceId,
         const configuration::Deployment           &deployment)
    : modules::Module(),
      _instanceManager(instanceManager),
      _computeManager(computeManager),
      _rpcEngine(rpcEngine),
      _deployerInstanceId(deployerInstanceId),
      _instanceId(instanceId),
      _deployment(deployment)
  {
    _rpcEngine->addRPCTarget(__SERVING_REQUEST_DEPLOYMENT_CONFIGURATION_RPC_NAME, _computeManager->createExecutionUnit([this](void *) { sendDeploymentConfiguration(); }));
  }

  Module(std::shared_ptr<HiCR::InstanceManager>     instanceManager,
         std::shared_ptr<HiCR::ComputeManager>      computeManager,
         std::shared_ptr<HiCR::frontend::RPCEngine> rpcEngine,
         const HiCR::Instance::instanceId_t         deployerInstanceId,
         const HiCR::Instance::instanceId_t         instanceId)
    : modules::Module(),
      _instanceManager(instanceManager),
      _computeManager(computeManager),
      _rpcEngine(rpcEngine),
      _deployerInstanceId(deployerInstanceId),
      _instanceId(instanceId)
  {
    _rpcEngine->addRPCTarget(__SERVING_REQUEST_DEPLOYMENT_CONFIGURATION_RPC_NAME, _computeManager->createExecutionUnit([this](void *) { sendDeploymentConfiguration(); }));
  }

  ~Module() override = default;

  __INLINE__ const configuration::Deployment &getDeployment() const { return _deployment; }

  void initialize() override
  {
    auto _instances = _instanceManager->getInstances();
    // If I am not the deployer instance, simply request the deployment information from the deployer
    if (_instanceId == _deployerInstanceId)
    {
      for (const auto &instance : _instances)
      {
        if (instance->getId() == _deployerInstanceId) { continue; }
        _rpcEngine->listen();
      }
    }
    else { retrieveDeployment(); }
  }

  // Init-only module (no periodic work)
  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override {}

  protected:

  void service() override {}

  private:

  std::shared_ptr<HiCR::InstanceManager> _instanceManager;

  std::shared_ptr<HiCR::ComputeManager> _computeManager;

  std::shared_ptr<HiCR::frontend::RPCEngine> _rpcEngine;

  const HiCR::Instance::instanceId_t _deployerInstanceId;

  const HiCR::Instance::instanceId_t _instanceId;

  serving::configuration::Deployment _deployment;

  __INLINE__ void sendDeploymentConfiguration()
  {
    // Serializing
    const auto serializedDeployment = _deployment.serialize().dump();

    // Returning serialized topology
    _rpcEngine->submitReturnValue((void *)serializedDeployment.c_str(), serializedDeployment.size() + 1);
  }

  /**
   * Retrieves the deployment configuration from the deployer instance.
   */
  void retrieveDeployment()
  {
    // Send request RPC
    _rpcEngine->requestRPC(_deployerInstanceId, __SERVING_REQUEST_DEPLOYMENT_CONFIGURATION_RPC_NAME);

    // Wait for serialized information
    auto returnValue = _rpcEngine->getReturnValue();

    // Receiving raw serialized topology information from the worker
    std::string deploymentString = (char *)returnValue->getPointer();

    // Parsing serialized raw topology into a json object
    auto deploymentJs = nlohmann::json::parse(deploymentString);

    // Creating the deployment object from the json
    _deployment = configuration::Deployment(deploymentJs);

    // Freeing return value
    _rpcEngine->getMemoryManager()->freeLocalMemorySlot(returnValue);
  }
};
} // namespace serving::modules::broadcastDeployment