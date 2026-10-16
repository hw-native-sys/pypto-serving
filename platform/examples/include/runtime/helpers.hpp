#pragma once
#include <memory>
#include <vector>
#include <nlohmann_json/json.hpp>

#include <hicr/backends/boost/computeManager.hpp>
#include <hicr/backends/hwloc/topologyManager.hpp>
#include <hicr/backends/mpi/communicationManager.hpp>
#include <hicr/backends/mpi/instanceManager.hpp>
#include <hicr/backends/mpi/memoryManager.hpp>
#include <hicr/backends/pthreads/computeManager.hpp>
#include <hicr/core/communicationManager.hpp>
#include <hicr/core/computeResource.hpp>
#include <hicr/core/instanceManager.hpp>
#include <hicr/core/memoryManager.hpp>
#include <hicr/core/memorySpace.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <system/engine.hpp>
#include <taskr/runtime.hpp>

struct Runtime
{
  hwloc_topology_t                                          hwlocTopologyObject;
  std::shared_ptr<HiCR::InstanceManager>                    instanceManager;
  std::shared_ptr<HiCR::backend::mpi::CommunicationManager> communicationManager;
  std::shared_ptr<HiCR::backend::mpi::MemoryManager>        memoryManager;
  std::shared_ptr<HiCR::backend::pthreads::ComputeManager>  workerComputeManager;
  std::shared_ptr<HiCR::backend::boost::ComputeManager>     taskComputeManager;
  std::shared_ptr<HiCR::frontend::RPCEngine>                rpcEngine;
  std::shared_ptr<taskr::Runtime>                           taskr;
  std::shared_ptr<HiCR::MemorySpace>                        bufferMemorySpace;
  std::vector<std::shared_ptr<HiCR::ComputeResource>>       computeResources;
  std::unique_ptr<serving::system::Engine>                  serving;
};

__INLINE__ nlohmann::json makeDefaultTaskrConfig()
{
  nlohmann::json taskrConfig;
  taskrConfig["Task Worker Inactivity Time (Ms)"] = 500;
  taskrConfig["Task Suspend Interval Time (Ms)"]  = 500;
  taskrConfig["Minimum Active Task Workers"]      = 1;
  taskrConfig["Service Worker Count"]             = 1;
  taskrConfig["Make Task Workers Run Services"]   = true;
  return taskrConfig;
}

__INLINE__ Runtime makeRuntime(int *argc, char ***argv, const size_t computeResourceCount = 2)
{
  Runtime runtime;

  hwloc_topology_init(&runtime.hwlocTopologyObject);
  HiCR::backend::hwloc::TopologyManager hwlocTopologyManager(&runtime.hwlocTopologyObject);

  const auto topology       = hwlocTopologyManager.queryTopology();
  auto       device         = *topology.getDevices().begin();
  auto       memorySpaces   = device->getMemorySpaceList();
  runtime.bufferMemorySpace = *memorySpaces.begin();
  auto computeResourcesIt   = device->getComputeResourceList().begin();
  for (size_t i = 0; i < computeResourceCount; i++)
  {
    runtime.computeResources.push_back(*computeResourcesIt);
    ++computeResourcesIt;
  }
  auto rpcComputeResource = *runtime.computeResources.begin();

  runtime.instanceManager      = std::shared_ptr<HiCR::InstanceManager>(HiCR::backend::mpi::InstanceManager::createDefault(argc, argv));
  runtime.communicationManager = std::make_shared<HiCR::backend::mpi::CommunicationManager>();
  runtime.memoryManager        = std::make_shared<HiCR::backend::mpi::MemoryManager>();
  runtime.workerComputeManager = std::make_shared<HiCR::backend::pthreads::ComputeManager>();
  runtime.taskComputeManager   = std::make_shared<HiCR::backend::boost::ComputeManager>();

  runtime.rpcEngine = std::make_shared<HiCR::frontend::RPCEngine>(
    *runtime.communicationManager, *runtime.instanceManager, *runtime.memoryManager, *runtime.workerComputeManager, runtime.bufferMemorySpace, rpcComputeResource);
  runtime.rpcEngine->initialize();

  if (runtime.computeResources.size() != 2) { HICR_THROW_RUNTIME("Too many CR"); }
  runtime.taskr = std::make_shared<taskr::Runtime>(runtime.taskComputeManager.get(), runtime.workerComputeManager.get(), runtime.computeResources, makeDefaultTaskrConfig());

  runtime.serving = std::make_unique<serving::system::Engine>(runtime.instanceManager, runtime.taskComputeManager, runtime.rpcEngine, runtime.instanceManager->getRootInstanceId());
  return runtime;
}