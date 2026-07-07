#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>
#include <hicr/core/instanceManager.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <modules/module.hpp>
#include <system/channels/messageTypeRegistry.hpp>

namespace serving::system
{

#define __SERVING_SYSTEM_START_RPC_NAME "[serving/system] Start Instances RPC"
#define __SERVING_SYSTEM_STOP_RPC_NAME "[serving/system] Stop Instances RPC"

class Engine final
{
  public:

  Engine(std::shared_ptr<HiCR::InstanceManager>     instanceManager,
         std::shared_ptr<HiCR::ComputeManager>      computeManager,
         std::shared_ptr<HiCR::frontend::RPCEngine> rpcEngine,
         const HiCR::Instance::instanceId_t         deployerInstanceId)
    : _instanceManager(instanceManager),
      _computeManager(computeManager),
      _rpcEngine(rpcEngine),
      _instanceId(instanceManager->getCurrentInstance()->getId()),
      _deployerInstanceId(deployerInstanceId)
  {
    // Lifetime contract: Engine must outlive the RPCEngine. HiCR does not
    // expose an unregister API, so these lambdas cannot be removed in the
    // destructor. Callers must call terminate()/await() before destroying
    // the Engine to ensure no RPC dispatch races the destructor.
    _rpcEngine->addRPCTarget(__SERVING_SYSTEM_START_RPC_NAME, _computeManager->createExecutionUnit([this](void *) { start(); }));
    _rpcEngine->addRPCTarget(__SERVING_SYSTEM_STOP_RPC_NAME, _computeManager->createExecutionUnit([this](void *) { stop(); }));
  }

  ~Engine() = default;

  __INLINE__ void addModule(const std::string &name, std::shared_ptr<modules::Module> module)
  {
    if (_modules.contains(name)) HICR_THROW_LOGIC("Trying to add a module with a name that already exists in the system.");
    _modules[name] = module;
  }

  __INLINE__ void initialize()
  {
    printf("[Instance %lu] Initializing engine\n", _instanceId);

    _isRunning.store(false);

    for (const auto &[name, module] : _modules) { module->initialize(); }
  }

  __INLINE__ void run()
  {
    printf("[Instance %lu] Running engine\n", _instanceId);

    if (_instanceId == _deployerInstanceId)
    {
      for (const auto &instance : _instanceManager->getInstances())
      {
        if (instance->getId() == _instanceId) continue;
        _rpcEngine->requestRPC(instance->getId(), __SERVING_SYSTEM_START_RPC_NAME);
      }
      // Start myself
      start();
      return;
    }
    // Worker: wait for start RPC
    _rpcEngine->listen();
  }

  __INLINE__ void await()
  {
    printf("[Instance %lu] Awaiting engine\n", _instanceId);
    while (_isRunning.load() == true)
    {
      if (_rpcEngine->tryListen()) { _rpcEngine->parseAndExecuteRPC(); }
      else { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }

    for (const auto &[name, module] : _modules) { module->terminate(); }
    for (const auto &[name, module] : _modules) { module->await(); }
    for (const auto &[name, module] : _modules) { module->finalize(); }

    printf("[Instance %lu] Modules finalized\n", _instanceId);
  }

  __INLINE__ void terminate()
  {
    printf("[Instance %lu] Terminating engine\n", _instanceId);
    if (_instanceId == _deployerInstanceId)
    {
      for (const auto &instance : _instanceManager->getInstances())
      {
        if (instance->getId() == _instanceId) continue;
        _rpcEngine->requestRPC(instance->getId(), __SERVING_SYSTEM_STOP_RPC_NAME);
      }
      // Stop myself
      stop();
    }
  }

  __INLINE__ void createInstance() { _instanceManager->createInstance(); }

  __INLINE__ channels::MessageTypeRegistry &getMessageTypeRegistry() { return _messageTypeRegistry; }

  private:

  __INLINE__ void start()
  {
    if (_isRunning.load() == true) { return; }

    _isRunning.store(true);

    for (const auto &[name, module] : _modules) { module->run(); }
  }

  __INLINE__ void stop()
  {
    if (_isRunning.load() == false) { return; }

    _isRunning.store(false);
    printf("[Instance %lu] Instance stopped\n", _instanceId);
  }

  std::shared_ptr<HiCR::InstanceManager>     _instanceManager;
  std::shared_ptr<HiCR::ComputeManager>      _computeManager;
  std::shared_ptr<HiCR::frontend::RPCEngine> _rpcEngine;

  const HiCR::Instance::instanceId_t _instanceId;
  const HiCR::Instance::instanceId_t _deployerInstanceId;

  std::atomic<bool> _isRunning = false;

  std::map<std::string, std::shared_ptr<modules::Module>> _modules;

  channels::MessageTypeRegistry _messageTypeRegistry;
};
} // namespace serving::system