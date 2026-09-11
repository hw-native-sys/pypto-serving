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
#include <taskr/runtime.hpp>

#include <modules/service/module.hpp>
#include <system/engine.hpp>

int main(int argc, char *argv[])
{
  // Creating HWloc topology object
  hwloc_topology_t hwlocTopologyObject;

  // Reserving memory for hwloc
  hwloc_topology_init(&hwlocTopologyObject);

  // Initializing host (CPU) topology manager
  HiCR::backend::hwloc::TopologyManager hwlocTopologyManager(&hwlocTopologyObject);

  // Gathering topology from the topology manager
  const auto topology = hwlocTopologyManager.queryTopology();

  auto d         = *topology.getDevices().begin();
  auto memSpaces = d->getMemorySpaceList();
  if (memSpaces.empty()) HICR_THROW_RUNTIME("No memory spaces found on the queried device");
  auto bufferMemorySpace = *memSpaces.begin();

  const auto &availableComputeResources = d->getComputeResourceList();
  if (availableComputeResources.size() < 2) HICR_THROW_RUNTIME("Fewer than 2 compute resources available");
  auto computeResourcesIt = availableComputeResources.begin();

  // Use only 2 cores
  std::vector<std::shared_ptr<HiCR::ComputeResource>> computeResources;
  for (int i = 0; i < 2; i++)
  {
    computeResources.push_back(*computeResourcesIt);
    computeResourcesIt++;
  }
  auto computeResource = *computeResources.begin();

  // Getting managers
  auto instanceManager      = std::shared_ptr<HiCR::InstanceManager>(HiCR::backend::mpi::InstanceManager::createDefault(&argc, &argv));
  auto communicationManager = std::make_shared<HiCR::backend::mpi::CommunicationManager>();
  auto memoryManager        = std::make_shared<HiCR::backend::mpi::MemoryManager>();
  auto workerComputeManager = std::make_shared<HiCR::backend::pthreads::ComputeManager>();
  auto taskComputeManager   = std::make_shared<HiCR::backend::boost::ComputeManager>();

  // Instantiate RPC Engine
  auto rpcEngine = std::make_shared<HiCR::frontend::RPCEngine>(*communicationManager, *instanceManager, *memoryManager, *workerComputeManager, bufferMemorySpace, computeResource);

  // Initialize RPC Engine
  rpcEngine->initialize();

  // Creating taskr object
  nlohmann::json taskrConfig;
  taskrConfig["Task Worker Inactivity Time (Ms)"] = 100;  // Suspend workers if a certain time of inactivity elapses
  taskrConfig["Task Suspend Interval Time (Ms)"]  = 100;  // Workers suspend for this time before checking back
  taskrConfig["Minimum Active Task Workers"]      = 1;    // Have at least one worker active at all times
  taskrConfig["Service Worker Count"]             = 1;    // Have one dedicated service workers at all times to listen for incoming messages
  taskrConfig["Make Task Workers Run Services"]   = true; // Workers will check for meta messages in between executions
  auto taskr                                      = std::make_shared<taskr::Runtime>(taskComputeManager.get(), workerComputeManager.get(), computeResources, taskrConfig);

  // Creating serving Engine object
  serving::system::Engine serving(instanceManager, taskComputeManager, rpcEngine, instanceManager->getRootInstanceId());

  // Check whether the instance is root
  const auto isRoot     = instanceManager->getCurrentInstance()->isRootInstance();
  const auto instanceId = instanceManager->getCurrentInstance()->getId();

  auto serviceModule = std::make_unique<serving::modules::service::Module>(taskr);

  // Adding a simple service that prints "Hello world" one time. Here we call terminate from within the task
  // to keep the application simple
  int  reps         = 5;
  int  count        = 0;
  auto helloWorldFc = [&]() {
    if (count > reps) { return; }

    printf("[Instance %lu] Hello World %d!\n", instanceId, count);
    count++;
  };
  auto helloWorldService = taskr::Service(helloWorldFc, 10);
  serviceModule->addService("helloWorld", &helloWorldService);

  // Adding service module to serving
  serving.addModule("service", std::move(serviceModule));

  // Initializing serving
  serving.initialize();

  // Running serving
  serving.run();

  if (isRoot)
  {
    printf("[Instance %lu] issuing termination\n", instanceId);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    // Finalizing serving
    serving.terminate();
  }

  // Awaiting serving termination
  serving.await();

  // Finalize Instance Manager
  instanceManager->finalize();
}
