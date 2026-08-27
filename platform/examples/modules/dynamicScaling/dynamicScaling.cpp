// Dynamic replica scaling example.
//
// Topology: single MPI rank, coordinator co-located with all replicas.
//
// Sequence:
//   Phase 1 — two replicas (A, B) handle 30 jobs.
//   Phase 2 — drain B (fires callback immediately since B is idle after
//              Phase 1); 20 jobs flow only through A.
//   Phase 3 — hot-add replica C via addReplicaLive; 30 jobs shared by A + C.
//   Shutdown — drain A and C (both idle), clean up.
//
// drainReplica() fires the drainCallback synchronously when the replica has
// no in-flight job.  If a job is in flight, the callback fires from
// replicaResponseHandler when that job completes.  Either way, removeReplica
// is safe to call only after the callback.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
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

using coord_t = serving::modules::roles::coordinator::Module;

static constexpr serving::system::channels::Message::messageType_t kMsgType      = 100;
static constexpr size_t                                            kPayloadBytes = 64;
static constexpr size_t                                            kResultBytes  = sizeof(uint64_t);

static void processFc(serving::modules::roles::TaskContext &ctx)
{
  auto       slot  = ctx.getInput("tokens");
  const auto first = slot == nullptr ? uint8_t{0} : *static_cast<const uint8_t *>(slot->getPointer());
  uint64_t   out   = first;
  ctx.setOutput("logits", &out, sizeof(out));
}

// ---------------------------------------------------------------------------

static std::shared_ptr<serving::configuration::Edge> makeEdge(const std::string                        &name,
                                                              size_t                                    capacity,
                                                              size_t                                    bytes,
                                                              HiCR::CommunicationManager               *commMgr,
                                                              HiCR::MemoryManager                      *memMgr,
                                                              const std::shared_ptr<HiCR::MemorySpace> &memSpace)
{
  auto e = std::make_shared<serving::configuration::Edge>(name, capacity, bytes);
  e->setPayloadCommunicationManager(commMgr);
  e->setPayloadMemoryManager(memMgr);
  e->setPayloadMemorySpace(memSpace);
  e->setCoordinationCommunicationManager(commMgr);
  e->setCoordinationMemoryManager(memMgr);
  e->setCoordinationMemorySpace(memSpace);
  return e;
}

struct ReplicaChannels
{
  std::string                                        tokensKey;
  std::string                                        logitsKey;
  std::shared_ptr<serving::system::channels::Output> tokensOut;
  std::shared_ptr<serving::system::channels::Input>  logitsIn;
  std::shared_ptr<serving::system::channels::Input>  tokensIn;
  std::shared_ptr<serving::system::channels::Output> logitsOut;
};

static ReplicaChannels wireReplica(HiCR::Instance::instanceId_t                 id,
                                   serving::system::channels::channelId_t       base,
                                   const serving::configuration::Edge          &tokensEdge,
                                   const serving::configuration::Edge          &logitsEdge,
                                   serving::modules::channelController::Module &cc)
{
  const auto tk = "tokens-" + std::to_string(base);
  const auto lk = "logits-" + std::to_string(base);
  const auto te = makeInternalEdgeFromTemplate(tk, tokensEdge);
  const auto le = makeInternalEdgeFromTemplate(lk, logitsEdge);

  ReplicaChannels ch;
  ch.tokensKey = tk;
  ch.logitsKey = lk;
  ch.tokensOut = cc.addDesiredProducer(id, base, te, defaultChannelKeyBuilder).lock();
  ch.logitsIn  = cc.addDesiredConsumer(id, base + 1, le, defaultChannelKeyBuilder).lock();
  ch.tokensIn  = cc.addDesiredConsumer(id, base, te, defaultChannelKeyBuilder).lock();
  ch.logitsOut = cc.addDesiredProducer(id, base + 1, le, defaultChannelKeyBuilder).lock();
  return ch;
}

static void waitReady(const ReplicaChannels &ch)
{
  waitUntilReady(ch.tokensOut);
  waitUntilReady(ch.logitsIn);
  waitUntilReady(ch.tokensIn);
  waitUntilReady(ch.logitsOut);
}

static void submitJobs(size_t n, size_t seqBase, serving::modules::roles::JobFactory &factory, coord_t &coord)
{
  serving::system::channels::Message::metadata_t md;
  md.type    = kMsgType;
  md.groupId = 1;
  for (size_t i = 0; i < n; ++i)
  {
    md.sequenceId            = static_cast<uint64_t>(seqBase + i + 1);
    auto                 job = std::make_shared<serving::modules::roles::Job>(factory.createJob("infer", md));
    std::vector<uint8_t> payload(kPayloadBytes, static_cast<uint8_t>(md.sequenceId & 0xFF));
    job->getInputDependency("tokens").storeData(payload.data(), payload.size());
    job->getInputDependency("tokens").setSatisfied(true);
    coord.submitJob(job);
  }
}

static void waitJobs(const std::atomic<size_t> &completed, size_t target)
{
  while (completed.load(std::memory_order_acquire) < target) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  auto        rt       = makeRuntime(&argc, &argv);
  const auto  id       = rt.instanceManager->getCurrentInstance()->getId();
  auto       &serving  = *rt.serving;
  auto        commMgr  = rt.communicationManager.get();
  auto        memMgr   = rt.memoryManager.get();
  const auto &memSpace = rt.bufferMemorySpace;

  auto tokensEdge = makeEdge("tokens", 8, kPayloadBytes, commMgr, memMgr, memSpace);
  auto logitsEdge = makeEdge("logits", 8, kResultBytes, commMgr, memMgr, memSpace);

  serving::configuration::Deployment deployment;
  deployment.addChannel(tokensEdge);
  deployment.addChannel(logitsEdge);

  auto task = std::make_shared<serving::configuration::Task>(std::string("infer"));
  task->addInput("tokens");
  task->addOutput("logits");
  auto partition = std::make_shared<serving::configuration::Partition>("P0", id);
  partition->addTask(task);
  partition->addReplica(std::make_shared<serving::configuration::Replica>(id));
  deployment.addPartition(partition);

  std::vector<HiCR::CommunicationManager *> mgrs = {commMgr};
  auto                                      cc   = std::make_shared<serving::modules::channelController::Module>(id, mgrs);
  auto                                      cd   = std::make_shared<serving::modules::channelDispatcher::Module>(1);
  auto                                      svc  = std::make_shared<serving::modules::service::Module>(rt.taskr);
  svc->addService("ChannelController", cc->getService());
  svc->addService("ChannelDispatcher", cd->getService());
  serving.addModule("ChannelController", cc);
  serving.addModule("ChannelDispatcher", cd);
  serving.addModule("Service", svc);

  auto factory = serving::modules::roles::JobFactory(0, deployment);
  auto coord   = std::make_shared<coord_t>(1, "P0");
  coord->addMessageType(kMsgType);
  serving.addModule("Coordinator", coord);
  svc->addService("Coordinator", coord->getService());

  // Replica A — channel base 1000
  auto chA  = wireReplica(id, 1000, *tokensEdge, *logitsEdge, *cc);
  auto repA = std::make_shared<serving::modules::roles::replica::Module>("infer", task->getInputs(), task->getOutputs(), processFc);
  repA->addMessageType(kMsgType);
  serving.addModule("ReplicaA", repA);
  repA->setCoordinator(id, {{"logits", chA.logitsOut}}, {{"tokens", chA.tokensIn}});
  coord->addReplica(id, {{"tokens", chA.tokensOut}}, {{"logits", chA.logitsIn}});

  // Replica B — channel base 1002; will be drained after Phase 1
  auto chB  = wireReplica(id, 1002, *tokensEdge, *logitsEdge, *cc);
  auto repB = std::make_shared<serving::modules::roles::replica::Module>("infer", task->getInputs(), task->getOutputs(), processFc);
  repB->addMessageType(kMsgType);
  serving.addModule("ReplicaB", repB);
  repB->setCoordinator(id, {{"logits", chB.logitsOut}}, {{"tokens", chB.tokensIn}});
  coord->addReplica(id + 1, {{"tokens", chB.tokensOut}}, {{"logits", chB.logitsIn}});

  std::atomic<size_t>     completed{0};
  std::mutex              drainMtx;
  std::condition_variable drainCv;
  std::atomic<int>        drainCount{0};

  coord->setCompletionCallback(
    [&](const serving::system::channels::Message::metadata_t &, const std::vector<coord_t::JobOutput> &) { completed.fetch_add(1, std::memory_order_release); });

  coord->setDrainCallback([&](HiCR::Instance::instanceId_t rid) {
    printf("[drain] replica %lu is idle\n", rid);
    drainCount.fetch_add(1, std::memory_order_release);
    drainCv.notify_all();
  });

  for (auto &s : coord->buildSubscriptions()) cd->subscribe(s);
  for (auto &s : repA->buildSubscriptions()) cd->subscribe(s);
  for (auto &s : repB->buildSubscriptions()) cd->subscribe(s);

  serving.initialize();
  serving.run();

  waitReady(chA);
  waitReady(chB);

  // ── Phase 1: A + B, 30 jobs ──────────────────────────────────────────────
  printf("\n=== Phase 1: replicas A + B, 30 jobs ===\n");
  submitJobs(30, 0, factory, *coord);
  waitJobs(completed, 30);
  printf("[phase1] %zu completed\n", completed.load());

  // ── Phase 2: drain B (fires immediately, B is idle), 20 jobs on A ────────
  printf("\n=== Phase 2: drain B, 20 jobs on A only ===\n");
  coord->drainReplica(id + 1); // callback fires synchronously (B idle)
  {
    std::unique_lock lk(drainMtx);
    drainCv.wait(lk, [&] { return drainCount.load() >= 1; });
  }
  coord->removeReplica(id + 1);
  submitJobs(20, 30, factory, *coord);
  waitJobs(completed, 50);
  printf("[phase2] %zu completed, B removed\n", completed.load());

  // ── Phase 3: hot-add C (base 1004), 30 jobs on A + C ────────────────────
  printf("\n=== Phase 3: hot-add replica C, 30 jobs on A + C ===\n");

  auto chC = wireReplica(id, 1004, *tokensEdge, *logitsEdge, *cc);
  waitReady(chC); // wait before subscribing so dispatcher never polls uninitialized channels
  auto repC = std::make_shared<serving::modules::roles::replica::Module>("infer", task->getInputs(), task->getOutputs(), processFc);
  repC->addMessageType(kMsgType);
  serving.addModule("ReplicaC", repC);
  repC->setCoordinator(id, {{"logits", chC.logitsOut}}, {{"tokens", chC.tokensIn}});
  for (auto &s : repC->buildSubscriptions()) cd->subscribe(s);

  // addReplicaLive returns subscriptions for coordinator→replica response path
  auto newSubs = coord->addReplicaLive(id + 2, {{"tokens", chC.tokensOut}}, {{"logits", chC.logitsIn}});
  for (auto &s : newSubs) cd->subscribe(s);

  submitJobs(30, 50, factory, *coord);
  waitJobs(completed, 80);
  printf("[phase3] %zu completed\n", completed.load());

  // ── Shutdown: drain A + C (both idle after Phase 3) ───────────────────────
  printf("\n=== Shutdown: drain A + C ===\n");
  coord->drainReplica(id);     // fires synchronously (A idle)
  coord->drainReplica(id + 2); // fires synchronously (C idle)
  {
    std::unique_lock lk(drainMtx);
    drainCv.wait(lk, [&] { return drainCount.load() >= 3; });
  }
  coord->removeReplica(id);
  coord->removeReplica(id + 2);

  serving.terminate();
  serving.await();

  for (auto &[t, ch] : coord->buildUnsubscriptions()) cd->unsubscribe(t, ch);
  for (auto &[t, ch] : repA->buildUnsubscriptions()) cd->unsubscribe(t, ch);
  for (auto &[t, ch] : repB->buildUnsubscriptions()) cd->unsubscribe(t, ch);
  for (auto &[t, ch] : repC->buildUnsubscriptions()) cd->unsubscribe(t, ch);

  for (const auto &[tk, lk] :
       std::initializer_list<std::pair<std::string, std::string>>{{chA.tokensKey, chA.logitsKey}, {chB.tokensKey, chB.logitsKey}, {chC.tokensKey, chC.logitsKey}})
  {
    cc->removeDesiredProducer(tk);
    cc->removeDesiredConsumer(lk);
    cc->removeDesiredConsumer(tk);
    cc->removeDesiredProducer(lk);
  }

  rt.instanceManager->finalize();
  printf("\nDone. %zu jobs processed.\n", completed.load());
  return 0;
}
