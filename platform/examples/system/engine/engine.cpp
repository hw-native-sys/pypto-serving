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

  // Selecting first device
  const auto &devices = topology.getDevices();
  if (devices.empty())
  {
    fprintf(stderr, "No HWloc devices found.\n");
    return 1;
  }
  auto d = *devices.begin();

  // Getting memory space list from device
  const auto &memSpaces = d->getMemorySpaceList();
  if (memSpaces.empty())
  {
    fprintf(stderr, "No memory spaces on device.\n");
    return 1;
  }
  auto bufferMemorySpace = *memSpaces.begin();

  // Grabbing first compute resource for computing incoming RPCs
  const auto &computeResources = d->getComputeResourceList();
  if (computeResources.empty())
  {
    fprintf(stderr, "No compute resources on device.\n");
    return 1;
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

  // Creating serving Engine object
  serving::system::Engine serving(instanceManager, taskComputeManager, rpcEngine, instanceManager->getRootInstanceId());

  // Initializing serving
  serving.initialize();

  // Running serving
  serving.run();

  // Finalizing serving
  serving.terminate();

  // Awaiting serving termination
  serving.await();

  // Finalize Instance Manager
  instanceManager->finalize();
}
