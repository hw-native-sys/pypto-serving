// Single-rank coordinator+replica bridge exposed to Python via pybind11.
//
// The Python callback (processFc) is invoked by the replica for each job.
// It receives raw input bytes (typically a pickled SchedulerOutput) and must
// return raw output bytes (typically a pickled StepOutput).
//
// Usage from Python:
//   import serving_platform
//   bridge = serving_platform.Bridge(processFc, payloadCapacity=4,
//                                    payloadBytes=262144, resultBytes=65536)
//   seq_id = bridge.submit(pickled_bytes)
//   seq_id, result_bytes = bridge.getResult(timeoutMs=5000.0)
//   bridge.shutdown()
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

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

namespace py = pybind11;

static constexpr serving::system::channels::Message::messageType_t kMsgType     = 200;
static constexpr HiCR::Instance::instanceId_t                      kChanPayload = 3000;
static constexpr HiCR::Instance::instanceId_t                      kChanResult  = 4000;

// ─── Bridge ──────────────────────────────────────────────────────────────────

class Bridge
{
  public:

  // processFcPy: Python callable(bytes) -> bytes
  // payloadCapacity: max jobs in flight
  // payloadBytes: max serialized input size per job
  // resultBytes: max serialized output size per job
  Bridge(py::object processFcPy, size_t payloadCapacity, size_t payloadBytes, size_t resultBytes);
  ~Bridge();

  // Submit one job; returns a sequence ID that matches the one in getResult.
  uint64_t submit(py::bytes payload);

  // Block until one result arrives (releases GIL while waiting).
  // Returns (seqId, result_bytes). Throws std::runtime_error on timeout.
  std::pair<uint64_t, py::bytes> getResult(double timeoutMs = 5000.0);

  void shutdown();

  private:

  static void processFcImpl(serving::modules::roles::TaskContext &ctx, py::object &pyFn);

  Runtime _rt;

  serving::configuration::Deployment                            _deployment;
  std::shared_ptr<serving::modules::channelController::Module>  _cc;
  std::shared_ptr<serving::modules::channelDispatcher::Module>  _cd;
  std::shared_ptr<serving::modules::service::Module>            _svc;
  std::shared_ptr<serving::modules::roles::coordinator::Module> _coord;
  std::shared_ptr<serving::modules::roles::replica::Module>     _replica;
  std::unique_ptr<serving::modules::roles::JobFactory>          _jobFactory;

  std::shared_ptr<serving::system::channels::Output> _payloadOut;
  std::shared_ptr<serving::system::channels::Input>  _resultIn;
  std::shared_ptr<serving::system::channels::Input>  _payloadIn;
  std::shared_ptr<serving::system::channels::Output> _resultOut;

  struct Result
  {
    uint64_t    seqId;
    std::string bytes;
  };
  std::mutex              _mtx;
  std::condition_variable _cv;
  std::queue<Result>      _queue;

  std::atomic<uint64_t> _nextSeqId{1};
  py::object            _processFcPy;
};

// ─── processFcImpl ──────────────────────────────────────────────────────────

void Bridge::processFcImpl(serving::modules::roles::TaskContext &ctx, py::object &pyFn)
{
  auto slot = ctx.getInput("payload");
  if (slot == nullptr) return;

  std::string result;
  {
    py::gil_scoped_acquire gil;
    auto                   pyInput  = py::bytes(static_cast<const char *>(slot->getPointer()), slot->getSize());
    auto                   pyOutput = pyFn(pyInput);
    result                          = static_cast<std::string>(pyOutput.cast<py::bytes>());
  }
  ctx.setOutput("result", result.data(), result.size());
}

// ─── Bridge constructor ──────────────────────────────────────────────────────

Bridge::Bridge(py::object processFcPy, size_t payloadCapacity, size_t payloadBytes, size_t resultBytes)
  : _processFcPy(std::move(processFcPy))
{
  int    fakeArgc = 0;
  char  *fakeArgv = nullptr;
  char **pArgv    = &fakeArgv;
  // makeRuntime calls MPI_Init internally.  Guard against double-init
  // if another Bridge was previously created and torn down in this process.
  int alreadyInit = 0;
  MPI_Initialized(&alreadyInit);
  if (alreadyInit)
    throw std::runtime_error("serving_platform.Bridge: MPI is already initialized. "
                             "Only one Bridge may exist per process lifetime.");
  _rt = makeRuntime(&fakeArgc, &pArgv);

  const auto instanceId = _rt.instanceManager->getCurrentInstance()->getId();
  auto      &engine     = *_rt.serving;

  // ── Deployment ────────────────────────────────────────────────────────────
  auto makeEdge = [&](const std::string &name, size_t cap, size_t sz) {
    auto e = std::make_shared<serving::configuration::Edge>(name, cap, sz);
    e->setPayloadCommunicationManager(_rt.communicationManager.get());
    e->setPayloadMemoryManager(_rt.memoryManager.get());
    e->setPayloadMemorySpace(_rt.bufferMemorySpace);
    e->setCoordinationCommunicationManager(_rt.communicationManager.get());
    e->setCoordinationMemoryManager(_rt.memoryManager.get());
    e->setCoordinationMemorySpace(_rt.bufferMemorySpace);
    return e;
  };

  _deployment.addChannel(makeEdge("payload", payloadCapacity, payloadBytes));
  _deployment.addChannel(makeEdge("result", payloadCapacity, resultBytes));

  auto task = std::make_shared<serving::configuration::Task>(std::string("serve"));
  task->addInput("payload");
  task->addOutput("result");
  auto partition = std::make_shared<serving::configuration::Partition>("P0", instanceId);
  partition->addTask(task);
  partition->addReplica(std::make_shared<serving::configuration::Replica>(instanceId));
  _deployment.addPartition(partition);

  // ── Core modules ──────────────────────────────────────────────────────────
  std::vector<HiCR::CommunicationManager *> mgrs = {_rt.communicationManager.get()};
  _cc                                            = std::make_shared<serving::modules::channelController::Module>(instanceId, mgrs);
  _cd                                            = std::make_shared<serving::modules::channelDispatcher::Module>(1);
  _svc                                           = std::make_shared<serving::modules::service::Module>(_rt.taskr);

  _svc->addService("ChannelController", _cc->getService());
  _svc->addService("ChannelDispatcher", _cd->getService());
  engine.addModule("ChannelController", _cc);
  engine.addModule("ChannelDispatcher", _cd);
  engine.addModule("Service", _svc);

  // ── Coordinator ───────────────────────────────────────────────────────────
  _coord = std::make_shared<serving::modules::roles::coordinator::Module>(1);
  _coord->addMessageType(kMsgType);
  engine.addModule("Coordinator", _coord);
  _svc->addService("Coordinator", _coord->getService());

  const auto intPayloadEdge = makeInternalEdgeFromTemplate("payload-coord-replica", *_deployment.getEdges()[0]);
  const auto intResultEdge  = makeInternalEdgeFromTemplate("result-replica-coord", *_deployment.getEdges()[1]);

  _payloadOut = _cc->addDesiredProducer(instanceId, kChanPayload, intPayloadEdge, defaultChannelKeyBuilder).lock();
  _resultIn   = _cc->addDesiredConsumer(instanceId, kChanResult, intResultEdge, defaultChannelKeyBuilder).lock();

  serving::modules::roles::coordinator::Module::inputChannelMap_t  replicaInputs  = {{"payload", _payloadOut}};
  serving::modules::roles::coordinator::Module::outputChannelMap_t replicaOutputs = {{"result", _resultIn}};
  _coord->addReplica(instanceId, replicaInputs, replicaOutputs);

  // ── Replica ───────────────────────────────────────────────────────────────
  auto myProcessFc = [this](serving::modules::roles::TaskContext &ctx) { Bridge::processFcImpl(ctx, _processFcPy); };

  const auto &savedTask = _deployment.getPartitions().front()->getTasks().front();
  _replica              = std::make_shared<serving::modules::roles::replica::Module>("serve", savedTask->getInputs(), savedTask->getOutputs(), myProcessFc);
  _replica->addMessageType(kMsgType);
  engine.addModule("Replica", _replica);

  _payloadIn = _cc->addDesiredConsumer(instanceId, kChanPayload, intPayloadEdge, defaultChannelKeyBuilder).lock();
  _resultOut = _cc->addDesiredProducer(instanceId, kChanResult, intResultEdge, defaultChannelKeyBuilder).lock();
  _replica->setCoordinator(instanceId, {{"result", _resultOut}}, {{"payload", _payloadIn}});

  // ── JobFactory ────────────────────────────────────────────────────────────
  _jobFactory = std::make_unique<serving::modules::roles::JobFactory>(0, _deployment);

  // ── Completion callback ───────────────────────────────────────────────────
  _coord->setCompletionCallback(
    [this](const serving::system::channels::Message::metadata_t &meta, const std::vector<serving::modules::roles::coordinator::Module::JobOutput> &outputs) {
      std::string bytes;
      if (!outputs.empty() && outputs[0].data != nullptr)
      {
        const auto *ptr = static_cast<const char *>(outputs[0].data->getPointer());
        bytes.assign(ptr, outputs[0].data->getSize());
      }
      {
        std::lock_guard<std::mutex> lk(_mtx);
        _queue.push({meta.sequenceId, std::move(bytes)});
      }
      _cv.notify_one();
    });

  // ── Subscribe & start ─────────────────────────────────────────────────────
  for (auto &sub : _coord->buildSubscriptions()) _cd->subscribe(sub);
  for (auto &sub : _replica->buildSubscriptions()) _cd->subscribe(sub);

  {
    py::gil_scoped_release release;
    engine.initialize();
    engine.run();
    waitUntilReady(_payloadOut);
    waitUntilReady(_resultIn);
    waitUntilReady(_payloadIn);
    waitUntilReady(_resultOut);
  }
}

// ─── Bridge::submit ──────────────────────────────────────────────────────────

uint64_t Bridge::submit(py::bytes payload)
{
  const uint64_t seqId = _nextSeqId.fetch_add(1, std::memory_order_relaxed);

  serving::system::channels::Message::metadata_t md;
  md.type       = kMsgType;
  md.groupId    = 1;
  md.sequenceId = seqId;

  auto job = std::make_shared<serving::modules::roles::Job>(_jobFactory->createJob("serve", md));

  auto sv = static_cast<std::string>(payload);
  job->getInputDependency("payload").storeData(reinterpret_cast<const uint8_t *>(sv.data()), sv.size());
  job->getInputDependency("payload").setSatisfied(true);

  _coord->submitJob(job);
  return seqId;
}

// ─── Bridge::getResult ───────────────────────────────────────────────────────

std::pair<uint64_t, py::bytes> Bridge::getResult(double timeoutMs)
{
  using clock    = std::chrono::steady_clock;
  const auto ddl = clock::now() + std::chrono::duration<double, std::milli>(timeoutMs);

  std::unique_lock<std::mutex> lock(_mtx);
  bool                         gotResult = false;
  {
    py::gil_scoped_release release;
    gotResult = _cv.wait_until(lock, ddl, [this] { return !_queue.empty(); });
  }
  if (!gotResult) throw std::runtime_error("Bridge::getResult timed out");

  auto r = std::move(_queue.front());
  _queue.pop();
  return {r.seqId, py::bytes(r.bytes)};
}

// ─── Bridge::shutdown ────────────────────────────────────────────────────────

void Bridge::shutdown()
{
  auto &engine = *_rt.serving;
  {
    py::gil_scoped_release release;
    engine.terminate();
    engine.await();
  }
  for (auto &[type, ch] : _coord->buildUnsubscriptions()) _cd->unsubscribe(type, ch);
  for (auto &[type, ch] : _replica->buildUnsubscriptions()) _cd->unsubscribe(type, ch);
  _cc->removeDesiredProducer("payload-coord-replica");
  _cc->removeDesiredConsumer("result-replica-coord");
  _cc->removeDesiredConsumer("payload-coord-replica");
  _cc->removeDesiredProducer("result-replica-coord");
  // MPI_Finalize is called once at process exit via atexit, not here,
  // because MPI cannot be re-initialized in the same process after finalization.
}

Bridge::~Bridge()
{
  try
  {
    shutdown();
  }
  catch (...)
  {}
}

// ─── pybind11 module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(serving_platform, m)
{
  m.doc() = "PyPTO Serving Platform — Python bindings for coordinator/replica bridge";

  // MPI_Finalize must be called exactly once per process.  Register it at
  // module import time so it fires on clean Python exit regardless of how
  // many Bridge objects are created and destroyed.
  std::atexit([] {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  });

  py::class_<Bridge>(m, "Bridge")
    .def(py::init<py::object, size_t, size_t, size_t>(),
         py::arg("processFc"),
         py::arg("payloadCapacity") = 4,
         py::arg("payloadBytes")    = 262144,
         py::arg("resultBytes")     = 65536,
         "Create a co-located coordinator+replica bridge.\n\n"
         "processFc(bytes) -> bytes is called by the replica for every submitted job.\n"
         "payloadCapacity: max concurrent jobs in flight.\n"
         "payloadBytes: max serialized input size.\n"
         "resultBytes: max serialized output size.")
    .def("submit", &Bridge::submit, py::arg("payload"), "Submit a job. Returns the sequence ID (uint64). Thread-safe.")
    .def("getResult",
         &Bridge::getResult,
         py::arg("timeoutMs") = 5000.0,
         "Block until a result arrives. Returns (seqId, result_bytes).\n"
         "Raises RuntimeError on timeout. Releases the GIL while waiting.")
    .def("shutdown", &Bridge::shutdown, "Terminate the engine and release resources.");
}
