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
#include <modules/channelController/module.hpp>
#include <modules/service/module.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>
#include <runtime/helpers.hpp>
#include "telephoneGame.hpp"

int main(int argc, char *argv[])
{
  auto        runtime    = makeRuntime(&argc, &argv);
  const auto &instance   = runtime.instanceManager->getCurrentInstance();
  const auto  instanceId = instance->getId();
  const auto  isRoot     = instance->isRootInstance();
  auto       &serving    = *runtime.serving;

  serving::configuration::Deployment deployment;

  if (argc != 2)
  {
    fprintf(stderr, "Error: Must provide the config file path.\n");
    runtime.instanceManager->abort(-1);
    return -1;
  }

  readAndParseConfiguration(argv, deployment, runtime.instanceManager);
  assignEdgeManagers(deployment, runtime.communicationManager.get(), runtime.memoryManager.get(), runtime.bufferMemorySpace);

  std::vector<HiCR::CommunicationManager *> managerOrder            = {runtime.communicationManager.get()};
  auto                                      channelControllerModule = std::make_shared<serving::modules::channelController::Module>(instanceId, managerOrder);

  auto channels = createDesiredSingleLocalChannels(deployment, instanceId, defaultChannelKeyBuilder, channelControllerModule);

  auto serviceModule = std::make_shared<serving::modules::service::Module>(runtime.taskr);
  serviceModule->addService("ChannelController", channelControllerModule->getService());

  serving.addModule("ChannelController", channelControllerModule);
  serving.addModule("Service", serviceModule);

  serving.initialize();

  serving.run();

  waitUntilReady(channels.input);
  waitUntilReady(channels.output);

  telephoneGame(*channels.input, *channels.output, instanceId, isRoot);

  removeDesiredSingleLocalChannels(channelControllerModule, channels);

  if (isRoot) serving.terminate();

  serving.await();

  runtime.instanceManager->finalize();

  return 0;
}
