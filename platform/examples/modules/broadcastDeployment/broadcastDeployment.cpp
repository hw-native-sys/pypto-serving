#include <stdio.h>
#include <thread>
#include <fstream>
#include <random>

#include <hicr/backends/hwloc/memoryManager.hpp>
#include <hicr/backends/mpi/memoryManager.hpp>
#include <hicr/backends/hwloc/topologyManager.hpp>
#include <hicr/backends/mpi/instanceManager.hpp>
#include <hicr/backends/mpi/communicationManager.hpp>
#include <hicr/backends/pthreads/computeManager.hpp>
#include <hicr/backends/pthreads/communicationManager.hpp>
#include <hicr/backends/boost/computeManager.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <modules/configuration/deployment.hpp>
#include <modules/broadcastDeployment/module.hpp>
#include <system/engine.hpp>

#include <deployment/helpers.hpp>
#include <runtime/helpers.hpp>

int main(int argc, char *argv[])
{
  auto        runtime  = makeRuntime(&argc, &argv);
  const auto &instance = runtime.instanceManager->getCurrentInstance();
  auto       &serving  = *runtime.serving;

  // Check whether the instance is root
  const auto isRoot             = runtime.instanceManager->getCurrentInstance()->isRootInstance();
  const auto instanceId         = runtime.instanceManager->getCurrentInstance()->getId();
  const auto deployerInstanceId = runtime.instanceManager->getRootInstanceId();

  ///// Configuration parsing
  serving::configuration::Deployment deployment;

  // If I am root, checking arguments.
  // Do not assume other instances will have the correct arguments set (e.g., file that exists only on root instance)
  if (isRoot == true)
  {
    if (argc != 2)
    {
      fprintf(stderr, "Error: Must provide the config file path.\n");
      runtime.instanceManager->abort(-1);
    }
    // Read and parse config file
    readAndParseConfiguration(argv, deployment, runtime.instanceManager);
  }

  std::shared_ptr<serving::modules::broadcastDeployment::Module> broadcastDeploymentModule;
  if (instanceId == deployerInstanceId)
  {
    broadcastDeploymentModule = std::make_shared<serving::modules::broadcastDeployment::Module>(
      runtime.instanceManager, runtime.taskComputeManager, runtime.rpcEngine, deployerInstanceId, instanceId, deployment);
  }
  else
  {
    broadcastDeploymentModule =
      std::make_shared<serving::modules::broadcastDeployment::Module>(runtime.instanceManager, runtime.taskComputeManager, runtime.rpcEngine, deployerInstanceId, instanceId);
  }

  const auto &receivedDeployment = broadcastDeploymentModule->getDeployment();
  // Adding broadcast deployment module to serving
  serving.addModule("BroadcastDeployment", broadcastDeploymentModule);

  // Initializing serving
  serving.initialize();

  // Running serving
  serving.run();

  // Finalizing serving
  serving.terminate();

  // Awaiting serving termination
  serving.await();

  // Printing deployment information to verify it was correctly received
  std::this_thread::sleep_for(std::chrono::seconds(instanceId)); // Sleep a bit to ensure the output is not mixed
  printf("[Instance %lu] Received deployment configuration:\n%s\n", instanceId, receivedDeployment.serialize().dump(2).c_str());

  // Finalize Instance Manager
  runtime.instanceManager->finalize();
}
