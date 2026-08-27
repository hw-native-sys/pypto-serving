#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include <modules/channelController/module.hpp>
#include <modules/channelDispatcher/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/configuration/edge.hpp>
#include <modules/configuration/partition.hpp>
#include <modules/configuration/replica.hpp>
#include <modules/configuration/task.hpp>
#include <modules/roles/coordinator/module.hpp>
#include <modules/roles/jobFactory.hpp>
#include <modules/roles/replica/module.hpp>
#include <modules/roles/taskContext.hpp>
#include <modules/service/module.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <runtime/helpers.hpp>

// ---------------------------------------------------------------------------
// Runtime parameters (set from argv)
// ---------------------------------------------------------------------------
static size_t   gNumRequests = 200; // jobs to submit
static uint32_t gComputeUs   = 0;   // synthetic compute µs in processFc

static constexpr size_t                                            kPayloadBytes = 4096;
static constexpr serving::system::channels::Message::messageType_t kMessageType  = 100;

// ---------------------------------------------------------------------------
// processFc: the "model inference" hook.
// Reads input bytes, accumulates a checksum (simulates per-token work),
// writes the result back.  Same logic is used for the baseline measurement.
// ---------------------------------------------------------------------------
static uint64_t computeChecksum(const uint8_t *data, size_t size)
{
  uint64_t acc = 0;
  for (size_t i = 0; i < size; ++i) acc += data[i];
  return acc;
}

static void processFc(serving::modules::roles::TaskContext &ctx)
{
  auto       slot   = ctx.getInput("tokens");
  const auto data   = static_cast<const uint8_t *>(slot == nullptr ? nullptr : slot->getPointer());
  const auto size   = slot == nullptr ? 0 : slot->getSize();
  uint64_t   result = computeChecksum(data, size);

  // Simulate inference compute (configurable via gComputeUs)
  if (gComputeUs > 0)
  {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(gComputeUs);
    while (std::chrono::steady_clock::now() < deadline) {}
  }

  ctx.setOutput("logits", &result, sizeof(result));
}

// ---------------------------------------------------------------------------
// Build a minimal single-partition deployment in memory (no JSON).
// One task "infer", input edge "tokens", output edge "logits".
// Coordinator and replica are both on instance 0 (co-located).
// ---------------------------------------------------------------------------
static serving::configuration::Deployment buildDeployment(const HiCR::Instance::instanceId_t        instanceId,
                                                          HiCR::CommunicationManager               *commMgr,
                                                          HiCR::MemoryManager                      *memMgr,
                                                          const std::shared_ptr<HiCR::MemorySpace> &memSpace)
{
  serving::configuration::Deployment deployment;

  auto tokensEdge = std::make_shared<serving::configuration::Edge>("tokens", /*capacity=*/4, kPayloadBytes);
  tokensEdge->setPayloadCommunicationManager(commMgr);
  tokensEdge->setPayloadMemoryManager(memMgr);
  tokensEdge->setPayloadMemorySpace(memSpace);
  tokensEdge->setCoordinationCommunicationManager(commMgr);
  tokensEdge->setCoordinationMemoryManager(memMgr);
  tokensEdge->setCoordinationMemorySpace(memSpace);

  auto logitsEdge = std::make_shared<serving::configuration::Edge>("logits", /*capacity=*/4, sizeof(uint64_t));
  logitsEdge->setPayloadCommunicationManager(commMgr);
  logitsEdge->setPayloadMemoryManager(memMgr);
  logitsEdge->setPayloadMemorySpace(memSpace);
  logitsEdge->setCoordinationCommunicationManager(commMgr);
  logitsEdge->setCoordinationMemoryManager(memMgr);
  logitsEdge->setCoordinationMemorySpace(memSpace);

  deployment.addChannel(tokensEdge);
  deployment.addChannel(logitsEdge);

  auto task = std::make_shared<serving::configuration::Task>(std::string("infer"));
  task->addInput("tokens");
  task->addOutput("logits");
  auto partition = std::make_shared<serving::configuration::Partition>("P0", instanceId);
  partition->addTask(task);
  partition->addReplica(std::make_shared<serving::configuration::Replica>(instanceId));
  deployment.addPartition(partition);

  return deployment;
}

int main(int argc, char *argv[])
{
  if (argc >= 2) gNumRequests = static_cast<size_t>(std::stoull(argv[1]));
  if (argc >= 3) gComputeUs = static_cast<uint32_t>(std::stoull(argv[2]));

  auto        runtime    = makeRuntime(&argc, &argv);
  const auto &instance   = runtime.instanceManager->getCurrentInstance();
  const auto  instanceId = instance->getId();
  auto       &serving    = *runtime.serving;

  // ----- Build deployment -----
  auto deployment = buildDeployment(instanceId, runtime.communicationManager.get(), runtime.memoryManager.get(), runtime.bufferMemorySpace);

  // ----- Core modules -----
  std::vector<HiCR::CommunicationManager *> managerOrder      = {runtime.communicationManager.get()};
  auto                                      channelController = std::make_shared<serving::modules::channelController::Module>(instanceId, managerOrder);
  auto                                      channelDispatcher = std::make_shared<serving::modules::channelDispatcher::Module>(1);
  auto                                      serviceModule     = std::make_shared<serving::modules::service::Module>(runtime.taskr);

  serviceModule->addService("ChannelController", channelController->getService());
  serviceModule->addService("ChannelDispatcher", channelDispatcher->getService());
  serving.addModule("ChannelController", channelController);
  serving.addModule("ChannelDispatcher", channelDispatcher);
  serving.addModule("Service", serviceModule);

  // ----- Coordinator -----
  const auto &partition = deployment.getPartitions().front();
  const auto &task      = partition->getTasks().front();

  auto jobFactory  = serving::modules::roles::JobFactory(0, deployment);
  auto coordinator = std::make_shared<serving::modules::roles::coordinator::Module>(1);
  coordinator->addMessageType(kMessageType);
  serving.addModule("Coordinator", coordinator);
  serviceModule->addService("Coordinator", coordinator->getService());

  // Coordinator→Replica internal channels (co-located: same rank, loopback)
  const auto internalTokensEdge = makeInternalEdgeFromTemplate("tokens-coord-replica", *deployment.getEdges()[0]);
  const auto internalLogitsEdge = makeInternalEdgeFromTemplate("logits-replica-coord", *deployment.getEdges()[1]);

  auto tokensOut = channelController->addDesiredProducer(instanceId, 1000, internalTokensEdge, defaultChannelKeyBuilder).lock();
  auto logitsIn  = channelController->addDesiredConsumer(instanceId, 2000, internalLogitsEdge, defaultChannelKeyBuilder).lock();

  serving::modules::roles::coordinator::Module::inputChannelMap_t  replicaInputs  = {{"tokens", tokensOut}};
  serving::modules::roles::coordinator::Module::outputChannelMap_t replicaOutputs = {{"logits", logitsIn}};
  coordinator->addReplica(instanceId, replicaInputs, replicaOutputs);

  // ----- Replica -----
  auto replica = std::make_shared<serving::modules::roles::replica::Module>("infer", task->getInputs(), task->getOutputs(), processFc);
  replica->addMessageType(kMessageType);
  serving.addModule("Replica", replica);

  auto tokensIn  = channelController->addDesiredConsumer(instanceId, 1000, internalTokensEdge, defaultChannelKeyBuilder).lock();
  auto logitsOut = channelController->addDesiredProducer(instanceId, 2000, internalLogitsEdge, defaultChannelKeyBuilder).lock();
  replica->setCoordinator(instanceId, {{"logits", logitsOut}}, {{"tokens", tokensIn}});

  // ----- Completion tracking -----
  std::atomic<size_t>   completed{0};
  std::vector<uint64_t> results(gNumRequests, 0);

  coordinator->setCompletionCallback(
    [&](const serving::system::channels::Message::metadata_t &meta, const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs) {
      const size_t idx = meta.sequenceId - 1;
      if (!outputs.empty() && outputs[0].data != nullptr) results[idx] = *static_cast<const uint64_t *>(outputs[0].data->getPointer());
      completed.fetch_add(1, std::memory_order_release);
    });

  // ----- Subscribe & start -----
  for (auto &sub : coordinator->buildSubscriptions()) channelDispatcher->subscribe(sub);
  for (auto &sub : replica->buildSubscriptions()) channelDispatcher->subscribe(sub);

  serving.initialize();
  serving.run();

  // Wait for channels to be ready
  waitUntilReady(tokensOut);
  waitUntilReady(logitsIn);
  waitUntilReady(tokensIn);
  waitUntilReady(logitsOut);

  // -----------------------------------------------------------------------
  // Baseline: N direct processFc-equivalent calls, no framework overhead.
  // -----------------------------------------------------------------------
  double baselineMs = 0.0;
  {
    std::vector<uint8_t> buf(kPayloadBytes, 0xAB);
    uint64_t             sink = 0;
    auto                 t0   = std::chrono::steady_clock::now();
    for (size_t i = 0; i < gNumRequests; ++i)
    {
      sink += computeChecksum(buf.data(), buf.size());
      if (gComputeUs > 0)
      {
        auto dl = std::chrono::steady_clock::now() + std::chrono::microseconds(gComputeUs);
        while (std::chrono::steady_clock::now() < dl) {}
      }
    }
    auto t1    = std::chrono::steady_clock::now();
    baselineMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[Baseline]  %zu calls | %.3f ms total | %.4f ms/call  (sink=%lu)\n", gNumRequests, baselineMs, baselineMs / gNumRequests, sink);
  }

  // -----------------------------------------------------------------------
  // Platform path: same N jobs through coordinator→replica→processFc.
  // -----------------------------------------------------------------------
  {
    std::vector<uint8_t> payload(kPayloadBytes, 0xAB);
    auto                 t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < gNumRequests; ++i)
    {
      serving::system::channels::Message::metadata_t md;
      md.type       = kMessageType;
      md.groupId    = 1;
      md.sequenceId = static_cast<uint64_t>(i + 1);

      auto job = std::make_shared<serving::modules::roles::Job>(jobFactory.createJob("infer", md));
      job->getInputDependency("tokens").storeData(payload.data(), payload.size());
      job->getInputDependency("tokens").setSatisfied(true);
      coordinator->submitJob(job);
    }

    while (completed.load(std::memory_order_acquire) < gNumRequests) std::this_thread::sleep_for(std::chrono::microseconds(10));

    auto   t1         = std::chrono::steady_clock::now();
    double platformMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double overheadMs = platformMs - baselineMs;
    double overheadPc = overheadMs / baselineMs * 100.0;

    printf("[Platform]  %zu jobs  | %.3f ms total | %.4f ms/call\n", gNumRequests, platformMs, platformMs / gNumRequests);
    printf("[Overhead]  +%.3f ms/call (%.1f%% of compute time)\n", overheadMs / gNumRequests, overheadPc);
    printf("            compute=%.0f us/call → framework %s critical path\n",
           static_cast<double>(gComputeUs),
           overheadPc < 5.0    ? "NOT in"
           : overheadPc < 20.0 ? "borderline"
                               : "IS in");
  }

  serving.terminate();
  serving.await();

  // Cleanup
  for (auto &[type, ch] : coordinator->buildUnsubscriptions()) channelDispatcher->unsubscribe(type, ch);
  for (auto &[type, ch] : replica->buildUnsubscriptions()) channelDispatcher->unsubscribe(type, ch);
  channelController->removeDesiredProducer("tokens-coord-replica");
  channelController->removeDesiredConsumer("logits-replica-coord");
  channelController->removeDesiredConsumer("tokens-coord-replica");
  channelController->removeDesiredProducer("logits-replica-coord");

  runtime.instanceManager->finalize();
  return 0;
}
