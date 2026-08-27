#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hicr/core/exceptions.hpp>
#include <hicr/core/instance.hpp>

#include <modules/module.hpp>
#include <modules/subscription.hpp>
#include <system/channels/input.hpp>
#include <system/channels/message.hpp>
#include <system/channels/output.hpp>

namespace serving::modules::roles::requestManager
{

class Module final : public serving::modules::Module
{
  public:

  using input_t       = std::shared_ptr<serving::system::channels::Input>;
  using output_t      = std::shared_ptr<serving::system::channels::Output>;
  using message_t     = serving::system::channels::Message;
  using metadata_t    = serving::system::channels::Message::metadata_t;
  using messageId_t   = serving::system::channels::Message::messageId_t;
  using messageType_t = serving::system::channels::Message::messageType_t;
  using groupId_t     = serving::system::channels::Message::groupId_t;
  using sequenceId_t  = serving::system::channels::Message::sequenceId_t;

  class Prompt final
  {
    public:

    Prompt(const metadata_t metadata, const uint8_t *data, const size_t size)
      : _metadata(metadata),
        _input((size > 0 && data != nullptr) ? std::vector<uint8_t>(data, data + size) : std::vector<uint8_t>{})
    {}

    [[nodiscard]] const metadata_t           &getMetadata() const { return _metadata; }
    [[nodiscard]] messageId_t                 getId() const { return _metadata.getId(); }
    [[nodiscard]] const std::vector<uint8_t> &getInput() const { return _input; }
    [[nodiscard]] const std::vector<uint8_t> &getResponse() const { return _response; }
    [[nodiscard]] bool                        hasResponse() const { return _hasResponse.load(); }
    [[nodiscard]] message_t                   getInputMessage() const { return message_t(_input.data(), _input.size(), _metadata); }

    private:

    friend class Module;

    void setResponse(const message_t &message)
    {
      if (message.getMetadata().getId() != getId()) { HICR_THROW_LOGIC("Response metadata does not match prompt metadata."); }
      _response.assign(message.getData(), message.getData() + message.getSize());
      _hasResponse.store(true);
    }
    metadata_t           _metadata;
    std::vector<uint8_t> _input;
    std::vector<uint8_t> _response;
    std::atomic<bool>    _hasResponse = false;
  };

  Module()
    : serving::modules::Module()
  {}

  __INLINE__ void addMessageType(const messageType_t messageType)
  {
    if (_hasMessageType) [[unlikely]]
      HICR_THROW_LOGIC("Request manager message type is already configured.");
    _messageType    = messageType;
    _hasMessageType = true;
  }

  __INLINE__ void setPromptOutput(const output_t &output)
  {
    if (output == nullptr) [[unlikely]]
      HICR_THROW_LOGIC("Request manager prompt output cannot be null.");
    _promptOutput = output;
  }

  __INLINE__ void setResultInput(const input_t &input)
  {
    if (input == nullptr) [[unlikely]]
      HICR_THROW_LOGIC("Request manager result input cannot be null.");
    _resultInput = input;
  }

  [[nodiscard]] __INLINE__ std::shared_ptr<Prompt> submit(const uint8_t *data, const size_t size)
  {
    if (_hasMessageType == false) HICR_THROW_LOGIC("Request manager has no message type configured.");
    if (_promptOutput == nullptr) HICR_THROW_LOGIC("Request manager prompt output has not been set.");
    if (data == nullptr && size > 0) HICR_THROW_LOGIC("Cannot submit a null prompt payload with non-zero size.");

    metadata_t metadata;
    metadata.type       = _messageType;
    metadata.groupId    = _defaultSessionId;
    metadata.sequenceId = _nextPromptId.fetch_add(1);

    auto prompt = std::make_shared<Prompt>(metadata, data, size);
    {
      std::lock_guard lock(_activePromptsMutex);
      const auto      promptKey = metadata.getId();
      if (_activePrompts.contains(promptKey)) HICR_THROW_LOGIC("Prompt %lu is already active.", promptKey);
      _activePrompts[promptKey] = prompt;
    }

    const auto message = message_t(prompt->getInput().data(), prompt->getInput().size(), metadata);
    _promptOutput->pushMessageLocking(message);
    return prompt;
  }

  [[nodiscard]] __INLINE__ std::vector<serving::modules::Subscription> buildSubscriptions()
  {
    if (_hasMessageType == false) HICR_THROW_LOGIC("Request manager has no message type configured.");
    if (_resultInput == nullptr) HICR_THROW_LOGIC("Request manager result input has not been set.");
    std::vector<serving::modules::Subscription> subscriptions;
    subscriptions.emplace_back(_messageType, _resultInput, [this](const input_t input, const message_t &message) { this->resultMessageHandler(input, message); });
    return subscriptions;
  }

  [[nodiscard]] __INLINE__ std::vector<std::pair<messageType_t, input_t>> buildUnsubscriptions() const
  {
    std::vector<std::pair<messageType_t, input_t>> unsubscriptions;
    if (_hasMessageType && _resultInput != nullptr) { unsubscriptions.push_back({_messageType, _resultInput}); }
    return unsubscriptions;
  }

  void initialize() override {}
  void run() override {}
  void terminate() override {}
  void await() override {}
  void finalize() override {}

  protected:

  void service() override {}

  private:

  __INLINE__ void resultMessageHandler(const input_t, const message_t &message)
  {
    if (message.getMetadata().type != _messageType) { HICR_THROW_RUNTIME("Request manager received unexpected message type %u.", message.getMetadata().type); }

    std::shared_ptr<Prompt> prompt;
    {
      std::lock_guard lock(_activePromptsMutex);
      const auto      promptKey = message.getMetadata().getId();
      if (_activePrompts.contains(promptKey) == false) { HICR_THROW_RUNTIME("Request manager received result for unknown prompt %lu.", promptKey); }
      prompt = _activePrompts.at(promptKey);
      _activePrompts.erase(promptKey);
    }
    prompt->setResponse(message);
  }

  static constexpr groupId_t                               _defaultSessionId = 0;
  output_t                                                 _promptOutput     = nullptr;
  input_t                                                  _resultInput      = nullptr;
  messageType_t                                            _messageType      = 0;
  bool                                                     _hasMessageType   = false;
  std::atomic<sequenceId_t>                                _nextPromptId     = 1;
  std::mutex                                               _activePromptsMutex;
  std::unordered_map<messageId_t, std::shared_ptr<Prompt>> _activePrompts;
};
} // namespace serving::modules::roles::requestManager
