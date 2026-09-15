#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <nlohmann_json/parser.hpp>

#include <hicr/core/definitions.hpp>
#include <hicr/core/instance.hpp>

#include "partition.hpp"
#include "edge.hpp"
#include "requestManager.hpp"

namespace serving::configuration
{

class Deployment final
{
  public:

  /**
   * Heartbeat is a runtime check mechanism to verify neighbors and replicas are still alive. 
   * Every 'interval' milliseconds, the coordinator sends a heartbeat message to its own replicas and vice-versa
   * If the 'tolerance' interval passes without a heartbeat, then the other side is deemed non-responsive
   * Recovery or fail tolerance mechanisms can be activated upon this happening
   */
  struct heartbeat_t
  {
    // Whether to enable the heartbeat service or not
    bool enabled;

    // Whether to print the hearbeat arriving on screen
    bool visible;

    // Heartbeat interval (every how many ms do we send a heartbeat message to replicas and neighboring coordinators)
    size_t interval;

    // Tolerance is how long to wait before alerting that we haven't received a heartbeat
    size_t tolerance;
  };

  /**
   * The control buffer is the edge between coordinators and replicas to exchange control messages.
   * Control messages are not related to the application, but instead to managing deployment aspects
   * (e.g., load balancing, fail detection, migration)
   */
  static constexpr size_t _defaultControlBufferCapacity = 256;
  static constexpr size_t _defaultControlBufferSize     = _defaultControlBufferCapacity * 1024;
  struct controlBuffer_t
  {
    // Capacity (the maximum number of pending tokens in the buffer)
    size_t capacity = _defaultControlBufferCapacity;

    // Size  (the size in bytes that the buffer can hold)
    size_t size = _defaultControlBufferSize;

    // HiCR-specific objects to create the control buffers. These are to be set at runtime
    HiCR::CommunicationManager        *communicationManager = nullptr;
    HiCR::MemoryManager               *memoryManager        = nullptr;
    std::shared_ptr<HiCR::MemorySpace> memorySpace          = nullptr;
  };

  struct settings_t
  {
    heartbeat_t     heartbeat;
    controlBuffer_t controlBuffer;
  };

  Deployment(const std::string &name)
    : _name(name){};
  Deployment(const nlohmann::json &js) { deserialize(js); };
  Deployment()  = default;
  ~Deployment() = default;

  __INLINE__ void setName(const std::string &name) { _name = name; }
  __INLINE__ void addPartition(const std::shared_ptr<Partition> partition) { _partitions.push_back(partition); }
  __INLINE__ void addChannel(const std::shared_ptr<Edge> channel) { _edges.push_back(channel); }

  [[nodiscard]] __INLINE__ std::string getName() const { return _name; }
  [[nodiscard]] __INLINE__ auto       &getPartitions() const { return _partitions; }
  [[nodiscard]] __INLINE__ auto       &getEdges() const { return _edges; }
  [[nodiscard]] __INLINE__ auto       &getHeartbeat() const { return _settings.heartbeat; }
  [[nodiscard]] __INLINE__ const auto &getControlBuffer() const { return _settings.controlBuffer; }
  [[nodiscard]] __INLINE__ auto       &getControlBuffer() { return _settings.controlBuffer; }
  [[nodiscard]] __INLINE__ auto       &getControlBufferConst() const { return _settings.controlBuffer; }
  [[nodiscard]] __INLINE__ auto       &getRequestManager() const { return _requestManager; }

  [[nodiscard]] __INLINE__ nlohmann::json serialize() const
  {
    nlohmann::json js;

    js["Name"] = _name;

    std::vector<nlohmann::json> partitionsJs;
    for (const auto &p : _partitions) partitionsJs.push_back(p->serialize());
    js["Partitions"] = partitionsJs;

    std::vector<nlohmann::json> edgesJs;
    for (const auto &e : _edges) edgesJs.push_back(e->serialize());
    js["Edges"] = edgesJs;

    js["Request Manager"] = _requestManager->serialize();

    ////////////////////// Parsing settings
    auto settings = std::map<std::string, nlohmann::json>();

    // Heartbeat
    auto heartbeat         = std::map<std::string, nlohmann::json>();
    heartbeat["Enabled"]   = _settings.heartbeat.enabled;
    heartbeat["Visible"]   = _settings.heartbeat.visible;
    heartbeat["Interval"]  = _settings.heartbeat.interval;
    heartbeat["Tolerance"] = _settings.heartbeat.tolerance;
    settings["Heartbeat"]  = heartbeat;

    // Control Buffer
    auto controlBuffer         = std::map<std::string, nlohmann::json>();
    controlBuffer["Capacity"]  = _settings.controlBuffer.capacity;
    controlBuffer["Size"]      = _settings.controlBuffer.size;
    settings["Control Buffer"] = controlBuffer;

    js["Settings"] = settings;

    return js;
  }

  __INLINE__ void deserialize(const nlohmann::json &js)
  {
    // Clearing objects
    _partitions.clear();
    _edges.clear();

    _name = hicr::json::getString(js, "Name");

    const auto &partitions = hicr::json::getArray<nlohmann::json>(js, "Partitions");
    for (const auto &p : partitions) _partitions.push_back(std::make_shared<Partition>(p));

    const auto &edges = hicr::json::getArray<nlohmann::json>(js, "Edges");
    for (const auto &e : edges) _edges.push_back(std::make_shared<Edge>(e));

    const auto &requestManagerJs = hicr::json::getObject(js, "Request Manager");
    _requestManager              = std::make_shared<RequestManager>(requestManagerJs);

    // Getting settings
    nlohmann::json settingsJs = hicr::json::getObject(js, "Settings");

    nlohmann::json heartbeatJs    = hicr::json::getObject(settingsJs, "Heartbeat");
    _settings.heartbeat.enabled   = hicr::json::getBoolean(heartbeatJs, "Enabled");
    _settings.heartbeat.visible   = hicr::json::getBoolean(heartbeatJs, "Visible");
    _settings.heartbeat.interval  = hicr::json::getNumber<size_t>(heartbeatJs, "Interval");
    _settings.heartbeat.tolerance = hicr::json::getNumber<size_t>(heartbeatJs, "Tolerance");

    nlohmann::json controlBufferJs   = hicr::json::getObject(settingsJs, "Control Buffer");
    _settings.controlBuffer.capacity = hicr::json::getNumber<size_t>(controlBufferJs, "Capacity");
    _settings.controlBuffer.size     = hicr::json::getNumber<size_t>(controlBufferJs, "Size");
  }

  // Includes all kinds of sanity checks relevant to a deployment
  __INLINE__ void verify() const
  {
    // Getting rqeuest manager input and output edges
    const auto &requestManagerInputEdge  = _requestManager->getInput();
    const auto &requestManagerOutputEdge = _requestManager->getOutput();

    // Getting all partition's edges in a set
    std::set<std::string> partitionNameSet;
    for (const auto &partition : _partitions) partitionNameSet.insert(partition->getName());

    // Getting a list of all edges to check the tasks specifying inputs/outputs actually refer to one of these, or user interface
    std::set<std::string> edgeNameSet;

    // Getting a list of all the inputs and output names to verify they correspond to each other at least once
    std::set<std::string> inputSet;
    std::set<std::string> outputSet;

    // Storage for those partitions with user interface input and output
    std::shared_ptr<serving::configuration::Partition> userInterfaceInputPartition  = nullptr;
    std::shared_ptr<serving::configuration::Partition> userInterfaceOutputPartition = nullptr;

    // Now checking all dependencies belong in the same partition
    std::set<std::string> tasksWithDependencies;
    for (const auto &partition : _partitions)
    {
      // Creating set of task names of this partition
      std::set<std::string> _taskNames;
      for (const auto &task : partition->getTasks()) _taskNames.insert(task->getFunctionName());

      for (const auto &task : partition->getTasks())
        for (const auto &dependency : task->getDependencies())
        {
          // Check the task doesn't depend on itself
          if (dependency == task->getFunctionName()) HICR_THROW_LOGIC("Task %s has a dependency on itself\n", task->getFunctionName().c_str());

          // Check the dependency has a task that is referenced by this dependency
          if (_taskNames.contains(dependency) == false)
            HICR_THROW_LOGIC("Task %s has a dependency on '%s' which is not defined in this partition\n", task->getFunctionName().c_str(), dependency.c_str());

          // Adding dependency to tasks with dependencies
          tasksWithDependencies.insert(dependency);
        }
    }

    // First, the user interface input/outputs are counted as edges for the purposes of this check
    for (const auto &edge : _edges)
    {
      const auto &edgeName = edge->getName();
      if (edgeNameSet.contains(edgeName)) HICR_THROW_LOGIC("Deployment specifies repeated edge or user input '%s'\n", edgeName.c_str());
      edgeNameSet.insert(edgeName);
    }

    // For each input / output, remember which partition is its consumer / producer
    std::map<std::string, std::string> consumerPartitionMap;
    std::map<std::string, std::string> producerPartitionMap;
    for (const auto &partition : _partitions)
      for (const auto &task : partition->getTasks())
      {
        // Make sure all tasks have at least one input
        if (task->getInputs().size() == 0 && task->getDependencies().empty() == true)
          HICR_THROW_LOGIC("Deployment specifies task in partition '%s' with function name '%s' without any inputs or dependencies\n",
                           partition->getName().c_str(),
                           task->getFunctionName().c_str());

        // Check that the edge is not used as input more than once. All edges must be 1-to-1
        for (const auto &input : task->getInputs())
        {
          if (inputSet.contains(input)) HICR_THROW_LOGIC("Deployment specifies input '%s' used more than once\n", input.c_str());
          if (edgeNameSet.contains(input) == false)
            HICR_THROW_LOGIC("Deployment specifies task '%s' with an undefined input '%s'\n", task->getFunctionName().c_str(), input.c_str());
          consumerPartitionMap[input] = partition->getName();
          inputSet.insert(input);

          // Check the task that contains the user interface input does not receive any other inputs
          if (input == requestManagerInputEdge) userInterfaceInputPartition = partition;
          if (input == requestManagerOutputEdge)
            HICR_THROW_LOGIC("Deployment specifies task '%s' with user interface output '%s' which is being used as input\n", task->getFunctionName().c_str(), input.c_str());
          if (input == requestManagerInputEdge && task->getInputs().size() > 1)
            HICR_THROW_LOGIC(
              "Deployment specifies task '%s' with user interface input '%s' which is not the only input of that task\n", task->getFunctionName().c_str(), input.c_str());
        }

        // Make sure all tasks have at least one output
        if (task->getOutputs().size() == 0 && tasksWithDependencies.contains(task->getFunctionName()) == false)
          HICR_THROW_LOGIC("Deployment specifies task in partition '%s' with function name '%s' without any outputs or dependents\n",
                           partition->getName().c_str(),
                           task->getFunctionName().c_str());

        // Check that the output is not used as input more than once. All edges must be 1-to-1
        for (const auto &output : task->getOutputs())
        {
          if (outputSet.contains(output)) HICR_THROW_LOGIC("Deployment specifies output '%s' used more than once\n", output.c_str());
          if (edgeNameSet.contains(output) == false)
            HICR_THROW_LOGIC("Deployment specifies task '%s' with an undefined output '%s'\n", task->getFunctionName().c_str(), output.c_str());
          producerPartitionMap[output] = partition->getName();
          outputSet.insert(output);

          // Check the task that contains the user interface input does not receive any other inputs
          if (output == requestManagerOutputEdge) userInterfaceOutputPartition = partition;
          if (output == requestManagerInputEdge)
            HICR_THROW_LOGIC("Deployment specifies task '%s' with user interface input '%s' which is being used as output\n", task->getFunctionName().c_str(), output.c_str());
          if (output == requestManagerOutputEdge && task->getOutputs().size() > 1)
            HICR_THROW_LOGIC(
              "Deployment specifies task '%s' with user interface output '%s' which is not the only output of task\n", task->getFunctionName().c_str(), output.c_str());
        }
      }

    // Check whether all edges have consumer+producer partitions that do exist
    for (const auto &edge : _edges)
    {
      const auto &edgeName = edge->getName();
      if (edgeName != requestManagerInputEdge)
        if (edgeName != requestManagerOutputEdge)
          if (consumerPartitionMap.contains(edgeName) == false || producerPartitionMap.contains(edgeName) == false)
            HICR_THROW_LOGIC("Deployment specifies edge '%s' but it is either not used as input or output (or neither)\n", edge->getName().c_str());

      // Getting producer and consumer partitions
      const auto &producerPartition = producerPartitionMap[edge->getName()];
      const auto &consumerPartition = consumerPartitionMap[edge->getName()];

      // Checking the edge connects two different partitions
      if (producerPartition == consumerPartition)
        HICR_THROW_LOGIC("Deployment specifies edge '%s' that is both produced and consumed by partition %s\n", edge->getName().c_str(), producerPartition.c_str());

      // Setting producer and consumer partitions for the edge
      edge->setProducer(producerPartition);
      edge->setConsumer(consumerPartition);
    }

    // Make sure all inputs are also used as outputs, as long as it is not the user interface input
    for (const auto &input : inputSet)
      if (input != requestManagerInputEdge)
        if (outputSet.contains(input) == false) HICR_THROW_LOGIC("Deployment input '%s' is not associated to any output\n", input.c_str());
    for (const auto &output : outputSet)
      if (output != requestManagerOutputEdge)
        if (inputSet.contains(output) == false) HICR_THROW_LOGIC("Deployment output '%s' is not associated to any input\n", output.c_str());

    // Check the user interface input/output are being used
    if (userInterfaceInputPartition == nullptr) HICR_THROW_LOGIC("User interface input '%s' is not associated to any partition\n", requestManagerInputEdge.c_str());
    if (userInterfaceOutputPartition == nullptr) HICR_THROW_LOGIC("User interface output '%s' is not associated to any partition\n", requestManagerOutputEdge.c_str());

    // Make sure the partition which contains the user interface input does not contain any cross-partition inputs
    for (const auto &task : userInterfaceInputPartition->getTasks())
      for (const auto &input : task->getInputs())
        if (input != requestManagerInputEdge)
          if (producerPartitionMap.at(input) != userInterfaceInputPartition->getName())
            HICR_THROW_LOGIC("Partition %s consumes the user interface input '%s' but also has other inter-partition inputs (e.g.,: '%s')\n",
                             userInterfaceInputPartition->getName().c_str(),
                             requestManagerInputEdge.c_str(),
                             input.c_str());

    // Make sure the partition which contains the user interface output does not contain any cross-partition outputs
    for (const auto &task : userInterfaceOutputPartition->getTasks())
      for (const auto &output : task->getOutputs())
        if (output != requestManagerOutputEdge)
          if (consumerPartitionMap.at(output) != userInterfaceOutputPartition->getName())
            HICR_THROW_LOGIC("Partition %s produces the user interface output '%s' but also has other inter-partition inputs (e.g.,: '%s')\n",
                             userInterfaceOutputPartition->getName().c_str(),
                             requestManagerOutputEdge.c_str(),
                             output.c_str());

    // Setting producer partition for user interface input to be the same as the consumer
    for (const auto &edge : _edges)
    {
      const auto &edgeName = edge->getName();

      if (edgeName == requestManagerInputEdge)
      {
        edge->setProducer(userInterfaceInputPartition->getName());
        edge->setConsumer(userInterfaceInputPartition->getName());
        edge->setPromptEdge(true);
      }

      if (edgeName == requestManagerOutputEdge)
      {
        edge->setProducer(userInterfaceOutputPartition->getName());
        edge->setConsumer(userInterfaceOutputPartition->getName());
        edge->setResultEdge(true);
      }
    }
  }

  private:

  std::string                             _name;
  std::vector<std::shared_ptr<Partition>> _partitions;
  std::vector<std::shared_ptr<Edge>>      _edges;
  std::shared_ptr<RequestManager>         _requestManager;
  settings_t                              _settings;

}; // class Deployment

} // namespace serving::configuration