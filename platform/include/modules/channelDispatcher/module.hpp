#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include <hicr/core/definitions.hpp>
#include <hicr/core/exceptions.hpp>

#include <modules/module.hpp>

#include <system/channels/input.hpp>
#include <system/channels/message.hpp>

#include <modules/subscription.hpp>

namespace serving::modules::channelDispatcher
{

class Module final : public modules::Module
{
  public:

  Module(const size_t intervalMs)
    : modules::Module(intervalMs)
  {}

  ~Module() override = default;

  __INLINE__ void subscribe(const Subscription &subscription)
  {
    const auto &edge = subscription.getEdge();

    std::lock_guard<std::mutex> guard(_subscriptionMutex);

    _subscribedEdges.insert(edge);

    const auto key = std::make_pair(edge.get(), subscription.getType());
    if (_subscriptionToHandlerMap.contains(key)) HICR_THROW_LOGIC("A handler is already subscribed for this input edge and message type.");
    _subscriptionToHandlerMap.insert({key, subscription.getHandler()});
  }

  __INLINE__ void unsubscribe(const channels::Message::messageType_t type, const std::shared_ptr<channels::Input> edge)
  {
    std::lock_guard<std::mutex> guard(_subscriptionMutex);

    const auto key = std::make_pair(edge.get(), type);
    if (_subscriptionToHandlerMap.contains(key) == false) return;
    _subscriptionToHandlerMap.erase(key);

    bool hasRemainingHandlers = false;
    for (const auto &[channelMessagePair, _] : _subscriptionToHandlerMap)
      if (channelMessagePair.first == edge.get())
      {
        hasRemainingHandlers = true;
        break;
      }
    if (hasRemainingHandlers == false) _subscribedEdges.erase(edge);
  }

  __INLINE__ void poll()
  {
    // reduce contention
    _pollEdges.clear();
    {
      std::lock_guard<std::mutex> guard(_subscriptionMutex);
      _pollEdges.reserve(_subscribedEdges.size());
      _pollEdges.assign(_subscribedEdges.begin(), _subscribedEdges.end());
    }

    for (const auto &edge : _pollEdges)
    {
      edge->lock();
      struct EdgeUnlockGuard
      {
        const std::shared_ptr<channels::Input> &edge;
        ~EdgeUnlockGuard() { edge->unlock(); }
      } unlockGuard{edge};

      if (edge->hasMessage() == false) { continue; }

      const auto       message     = edge->getMessage();
      const auto       messageType = message.getMetadata().type;
      const auto       key         = std::make_pair(edge.get(), messageType);
      messageHandler_t handler;
      {
        std::lock_guard<std::mutex> guard(_subscriptionMutex);
        if (_subscriptionToHandlerMap.contains(key) == false)
        {
          printf("[ChannelDispatcher] No handler found for message type %u. Message will be ignored.\n", messageType);
          edge->popMessage();
          continue;
        }
        handler = _subscriptionToHandlerMap.at(key);
      }
      handler(edge, message);
      edge->popMessage();
    }
  }

  void initialize() override {}
  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override
  {
    _subscribedEdges.clear();
    _subscriptionToHandlerMap.clear();
  }

  protected:

  void service() override { poll(); }

  private:

  std::mutex                                                                                       _subscriptionMutex;
  std::set<std::shared_ptr<channels::Input>>                                                       _subscribedEdges;
  std::map<std::pair<const channels::Input *, channels::Message::messageType_t>, messageHandler_t> _subscriptionToHandlerMap;

  // Used to reduce contention in poll()
  std::vector<std::shared_ptr<channels::Input>> _pollEdges;
};
} // namespace serving::modules::channelDispatcher
